"""Main training entry point for SF-Fuse.

This script handles:
- Configuration loading via Hydra
- Dataset and dataloader creation
- Model/Task instantiation
- Optional MLM pre-training phase for encoder
- Callback instantiation (including validation, checkpointing, timing, etc.)
- WandB logging integration
- Training with checkpoint resumption
- Validation and testing
"""

# ---------------------------------------------------------------------------
# Compatibility shim: mamba_ssm may import legacy symbols from transformers
# that were removed in transformers >= 4.39. Patch them before anything else.
# ---------------------------------------------------------------------------
import transformers.generation as _tg

_LEGACY_ALIASES = {
    "GreedySearchDecoderOnlyOutput":  "GenerateDecoderOnlyOutput",
    "GreedySearchEncoderDecoderOutput": "GenerateEncoderDecoderOutput",
    "SampleDecoderOnlyOutput":        "GenerateDecoderOnlyOutput",
    "SampleEncoderDecoderOutput":     "GenerateEncoderDecoderOutput",
    "BeamSearchDecoderOnlyOutput":    "GenerateDecoderOnlyOutput",
    "BeamSearchEncoderDecoderOutput": "GenerateEncoderDecoderOutput",
    "BeamSampleDecoderOnlyOutput":    "GenerateDecoderOnlyOutput",
    "BeamSampleEncoderDecoderOutput": "GenerateEncoderDecoderOutput",
}
for _old, _new in _LEGACY_ALIASES.items():
    if not hasattr(_tg, _old) and hasattr(_tg, _new):
        setattr(_tg, _old, getattr(_tg, _new))
del _old, _new, _LEGACY_ALIASES, _tg
# ---------------------------------------------------------------------------

import os
import random
import time
from functools import wraps
from typing import Callable, List

import fsspec
import hydra
import pytorch_lightning as pl
import torch
# PyTorch 2.6+ changed torch.load default to weights_only=True, which breaks
# checkpoint loading for Lightning checkpoints containing OmegaConf objects.
_original_load = torch.load
def _patched_load(*args, weights_only=_original_load.__code__.co_consts[-1] if hasattr(_original_load, '__code__') else False, **kwargs):
    return _original_load(*args, weights_only=False, **kwargs)
torch.load = _patched_load
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from torch.utils.data import DataLoader

from src.dataloaders.genomic_dataset import GenomicTracksDataset, worker_init_fn
from src.dataloaders.genome_mlm_dataset import ReferenceGenomeMLMDataset, mlm_worker_init_fn
from src.utils.config import (
    get_logger,
    print_config,
    print_model_info,
    process_config,
)

log = get_logger(__name__)

# Enable TensorFloat32 for faster training on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ----------- Custom WandB Logger with Retry Logic -----------

class DummyExperiment:
    """Dummy experiment for non-rank-zero processes."""
    def nop(self, *args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx) -> "DummyExperiment":
        return self

    def __setitem__(self, *args, **kwargs) -> None:
        pass


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""
    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)
        return get_experiment() or DummyExperiment()
    return experiment


class CustomWandbLogger(WandbLogger):
    """WandB logger with retry logic for network issues."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                self._experiment = wandb._attach(attach_id)
            else:
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        self._experiment = wandb.init(**self._wandb_init)
                        break
                    except Exception as e:
                        log.error(f"wandb Exception (attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            t = random.randint(30, 60)
                            log.warning(f"Sleeping for {t} seconds before retry")
                            time.sleep(t)
                        else:
                            log.error("Failed to initialize wandb after all retries")
                            raise

                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric(
                        "*", step_metric="trainer/global_step", step_sync=True
                    )

        return self._experiment


# ----------- Trainer Creation -----------

def create_trainer(config: DictConfig) -> pl.Trainer:
    """Create PyTorch Lightning Trainer with all callbacks and logger.

    Args:
        config: Hydra configuration.

    Returns:
        Configured pl.Trainer instance.
    """
    callbacks: List[pl.Callback] = []
    logger = None

    # WandB Logging
    if config.get("wandb") is not None and config.wandb.get("mode") != "disabled":
        log.info("Initializing WandB logger...")
        try:
            logger = CustomWandbLogger(
                config=OmegaConf.to_container(config, resolve=True),
                settings=wandb.Settings(start_method="fork"),
                **config.wandb,
            )
        except Exception as e:
            log.warning(f"Failed to initialize WandB logger: {e}. Continuing without WandB.")
            logger = None

    # Instantiate callbacks
    if "callbacks" in config:
        for cb_name, cb_conf in config.callbacks.items():
            if cb_conf is None:
                continue
            if logger is None and cb_name in ["learning_rate_monitor"]:
                log.info(f"Skipping callback <{cb_name}> because WandB is disabled")
                continue
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                try:
                    callbacks.append(hydra.utils.instantiate(cb_conf))
                except Exception as e:
                    log.warning(f"Failed to instantiate callback {cb_name}: {e}")

    # Process trainer config
    trainer_conf = OmegaConf.to_container(config.trainer, resolve=True)

    if "_target_" in trainer_conf:
        del trainer_conf["_target_"]

    # Handle strategy configuration
    strategy_conf = trainer_conf.get("strategy")
    if strategy_conf == "ddp":
        log.info("Using DDPStrategy with find_unused_parameters=True")
        from datetime import timedelta
        trainer_conf["strategy"] = DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(seconds=1200),
            start_method="fork",
        )
    elif isinstance(strategy_conf, dict) and "_target_" in strategy_conf:
        log.info(f"Instantiating strategy: {strategy_conf['_target_']}")
        trainer_conf["strategy"] = hydra.utils.instantiate(strategy_conf)

    log.info("Creating PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        **trainer_conf,
        callbacks=callbacks,
        logger=logger,
    )

    return trainer


def create_pretrain_trainer(config: DictConfig, pretrain_config) -> pl.Trainer:
    """Create a simplified trainer for the MLM pre-training phase.

    Args:
        config: Full Hydra configuration (for hardware/strategy settings).
        pretrain_config: Pre-training specific configuration dict.

    Returns:
        Configured pl.Trainer for pre-training.
    """
    from src.callbacks.timer import Timer

    # Minimal callbacks for pre-training
    callbacks: List[pl.Callback] = [
        Timer(step=True, inter_step=False, epoch=True, val=True),
        pl.callbacks.RichProgressBar(refresh_rate=1),
    ]

    # Logger
    logger = None
    if config.get("wandb") is not None and config.wandb.get("mode") != "disabled":
        try:
            # Create a wandb logger for the pre-training phase
            wandb_cfg = OmegaConf.to_container(config.wandb, resolve=True)
            wandb_cfg["name"] = wandb_cfg.get("name", "") + "_pretrain"
            wandb_cfg["group"] = wandb_cfg.get("group", "") or "pretrain"
            logger = CustomWandbLogger(
                config=OmegaConf.to_container(config, resolve=True),
                settings=wandb.Settings(start_method="fork"),
                **wandb_cfg,
            )
            callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))
        except Exception as e:
            log.warning(f"Failed to initialize WandB logger for pre-training: {e}")

    # Validation callback
    pretrain_val_every = pretrain_config.get("val_every_n_steps", 1000)
    try:
        from src.callbacks.validation import ValEveryNGlobalSteps
        callbacks.append(ValEveryNGlobalSteps(every_n=pretrain_val_every, num_validations=None))
    except Exception:
        pass

    # Build trainer config from base config (inherit hardware settings)
    trainer_conf = OmegaConf.to_container(config.trainer, resolve=True)
    if "_target_" in trainer_conf:
        del trainer_conf["_target_"]

    # Override with pre-training specific settings
    pretrain_steps = pretrain_config.get("steps", 10000)
    trainer_conf["max_steps"] = pretrain_steps
    trainer_conf["max_epochs"] = pretrain_config.get("max_epochs", 999)
    trainer_conf["gradient_clip_val"] = pretrain_config.get("gradient_clip_val", 1.0)

    # Disable epoch-based validation (use step-based from callback)
    trainer_conf["val_check_interval"] = None
    trainer_conf["check_val_every_n_epoch"] = 99999

    # Handle strategy
    strategy_conf = trainer_conf.get("strategy")
    if strategy_conf == "ddp":
        trainer_conf["strategy"] = DDPStrategy(find_unused_parameters=True)
    elif isinstance(strategy_conf, dict) and "_target_" in strategy_conf:
        trainer_conf["strategy"] = hydra.utils.instantiate(strategy_conf)

    log.info(
        f"Creating pre-training Trainer: "
        f"max_steps={pretrain_steps}, val_every={pretrain_val_every}"
    )

    trainer = pl.Trainer(
        **trainer_conf,
        callbacks=callbacks,
        logger=logger,
    )

    return trainer


# ----------- Main Training Function -----------

def train(config: DictConfig):
    """Main training function.

    Supports two-phase training:
    1. (Optional) MLM pre-training phase for encoder
    2. Main fine-tuning phase

    Args:
        config: Hydra configuration.
    """
    # Process config
    config = process_config(config)

    # Print config
    print_config(config, resolve=True)

    # Set seed
    seed = config.get("seed", 42)
    if config.get("train", {}).get("seed") is not None:
        seed = config.train.seed
    log.info(f"Setting random seed: {seed}")
    pl.seed_everything(seed, workers=True)

    # Create dataloaders
    log.info("Creating datasets and dataloaders...")
    batch_size = config.task.dataset.get("batch_size", 1)
    num_workers = config.task.dataset.get("num_workers", 0)

    # --- Auto-compute gradient accumulation from global_batch_size ---
    global_batch_size = config.get("train", {}).get("global_batch_size", None)
    num_devices = config.trainer.get("devices", 1)
    if isinstance(num_devices, (list, tuple)):
        num_devices = len(num_devices)

    if global_batch_size is not None and global_batch_size > 0:
        per_step_total = batch_size * num_devices
        if per_step_total == 0:
            raise ValueError("batch_size * num_devices = 0, cannot compute gradient accumulation.")
        if global_batch_size % per_step_total != 0:
            raise ValueError(
                f"global_batch_size ({global_batch_size}) must be divisible by "
                f"micro_batch_size ({batch_size}) × num_devices ({num_devices}) = {per_step_total}. "
                f"Remainder: {global_batch_size % per_step_total}"
            )
        accumulate_grad_batches = global_batch_size // per_step_total
        config.trainer.accumulate_grad_batches = accumulate_grad_batches
        log.info(
            f"Auto-computed gradient accumulation: "
            f"global_batch_size={global_batch_size} / "
            f"(micro_bs={batch_size} × gpus={num_devices}) = "
            f"accumulate_grad_batches={accumulate_grad_batches}"
        )
    else:
        accumulate_grad_batches = config.trainer.get("accumulate_grad_batches", 1)

    effective_batch_size = batch_size * num_devices * accumulate_grad_batches
    log.info(
        f"Batch size: micro={batch_size}, gpus={num_devices}, "
        f"grad_accum={accumulate_grad_batches}, effective={effective_batch_size}"
    )

    # Detect MLM mode: FASTA file extension (not just missing targets_file since fine-tuning can have null targets)
    def _is_mlm_mode(dataset_config):
        ext = os.path.splitext(dataset_config.train_data_file)[1].lower()
        return ext in ('.fa', '.fasta', '.fna')

    mlm_mode = _is_mlm_mode(config.task.dataset)

    if mlm_mode:
        log.info("Using MLM mode: ReferenceGenomeMLMDataset (FASTA/h5 for MLM pretraining)")
        train_dataset = ReferenceGenomeMLMDataset(
            data_file=config.task.dataset.train_data_file,
            seq_length=config.task.dataset.get("seq_length", config.task.dataset.get("max_length", 131072)),
            mask_prob=config.task.dataset.get("mask_prob", 0.15),
            mask_token_prob=config.task.dataset.get("mask_token_prob", 0.8),
            random_token_prob=config.task.dataset.get("random_token_prob", 0.1),
            num_samples_per_epoch=config.task.dataset.get("train_samples_per_epoch", 10000),
            min_chr_length=config.task.dataset.get("min_chr_length", 1000000),
            exclude_chr=config.task.dataset.get("exclude_chr", "chrM,chrUn,chrEBV,_random,_alt,_fix"),
            rc_augment=config.task.dataset.get("rc_augment", True),
            max_n_ratio=config.task.dataset.get("max_n_ratio", 0.1),
            h5_seq_key=config.task.dataset.get("h5_seq_key", "sequences"),
        )
        val_dataset = None
        if config.task.dataset.val_data_file:
            val_dataset = ReferenceGenomeMLMDataset(
                data_file=config.task.dataset.val_data_file,
                seq_length=config.task.dataset.get("seq_length", config.task.dataset.get("max_length", 131072)),
                mask_prob=config.task.dataset.get("mask_prob", 0.15),
                mask_token_prob=config.task.dataset.get("mask_token_prob", 0.8),
                random_token_prob=config.task.dataset.get("random_token_prob", 0.1),
                num_samples_per_epoch=config.task.dataset.get("val_samples_per_epoch", 1000),
                min_chr_length=config.task.dataset.get("min_chr_length", 1000000),
                exclude_chr=config.task.dataset.get("exclude_chr", "chrM,chrUn,chrEBV,_random,_alt,_fix"),
                rc_augment=False,
                max_n_ratio=config.task.dataset.get("max_n_ratio", 0.1),
                h5_seq_key=config.task.dataset.get("h5_seq_key", "sequences"),
            )
        worker_fn = mlm_worker_init_fn
    else:
        log.info("Using supervised fine-tuning mode: GenomicTracksDataset (H5 format)")
        train_dataset = GenomicTracksDataset(
            data_file=config.task.dataset.train_data_file,
            targets_file=config.task.dataset.get("train_targets_file"),
            seq_key=config.task.dataset.get("seq_key", "sequences"),
            tgt_key=config.task.dataset.get("tgt_key", "targets"),
            max_length=config.task.dataset.get("max_length", 131072),
            rc_augment=True,
        )
        val_dataset = GenomicTracksDataset(
            data_file=config.task.dataset.val_data_file,
            targets_file=config.task.dataset.get("val_targets_file"),
            seq_key=config.task.dataset.get("seq_key", "sequences"),
            tgt_key=config.task.dataset.get("tgt_key", "targets"),
            max_length=config.task.dataset.get("max_length", 131072),
            rc_augment=False,
        )
        worker_fn = worker_init_fn

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_fn,
    ) if val_dataset else None

    log.info(f"Train dataset size: {len(train_dataset)}")
    if val_dataset is not None:
        log.info(f"Val dataset size: {len(val_dataset)}")
    else:
        log.info("Validation dataset not configured - skipping validation")

    # Create task/model
    log.info("Instantiating task/model...")
    task = hydra.utils.instantiate(config.task)

    # =====================================================================
    # Phase 1: Optional MLM Pre-training for Encoder
    # =====================================================================
    pretrain_config = config.get("pretrain", {})
    pretrain_enabled = pretrain_config.get("enabled", False) if pretrain_config else False

    log.info("=" * 70)
    if pretrain_enabled:
        log.info("[CONFIG] Pre-training phase: ENABLED")
        log.info(f"  pretrain.steps = {pretrain_config.get('steps', 10000)}")
        log.info(f"  pretrain.lr    = {pretrain_config.get('lr', 1e-3)}")
    else:
        log.info("[CONFIG] Pre-training phase: DISABLED")
        log.info("  To enable, set pretrain.enabled=true")
    log.info("=" * 70)

    # Print model info (before phase switch, shows baseline)
    print_model_info(task)

    if pretrain_enabled:
        _run_pretrain_phase(task, config, pretrain_config, train_loader, val_loader)

    # =====================================================================
    # Phase 2: Main Fine-tuning
    # =====================================================================
    log.info("=" * 70)
    if pretrain_enabled:
        log.info("PHASE 2: Main Fine-tuning (post pre-training)")
        log.info("  Encoder weights: from Phase 1 MLM pre-training")
        log.info("  Bridge/Decoder/Head weights: random initialization")
        log.info("  All parameters unfrozen for supervised training")
    else:
        log.info("Main Training")
    log.info("=" * 70)

    # Create main trainer
    trainer = create_trainer(config)

    # Run initial validation
    if config.get("train", {}).get("validate_at_start", False):
        log.info("Running validation before training...")
        trainer.validate(task, val_loader)

    # Resume from checkpoint
    ckpt_path = None
    if pretrain_enabled:
        # After pre-training, the model already carries pretrained encoder
        # weights in-memory. Do NOT load any finetune checkpoint, as that
        # would overwrite the freshly pretrained encoder parameters.
        log.info(
            "Skipping checkpoint resume: encoder weights come from "
            "the pre-training phase that just completed."
        )
    else:
        ckpt_file = config.get("train", {}).get("ckpt")
        if ckpt_file:
            log.info(f"Checking for checkpoint at: {ckpt_file}")
            if fsspec_exists(ckpt_file):
                log.info(f"Found checkpoint at: {ckpt_file}. Resuming training...")
                ckpt_path = ckpt_file
            else:
                log.info(f"No checkpoint found at {ckpt_file}. Training from scratch.")
        else:
            log.info("No checkpoint path specified. Training from scratch.")

    # Train
    log.info("Starting training...")
    trainer.fit(task, train_loader, val_loader, ckpt_path=ckpt_path)

    # Run test if configured
    if config.get("train", {}).get("test", False):
        log.info("Running final validation...")
        trainer.validate(task, val_loader)

    # Log results
    _log_training_results(trainer, config)


def _run_pretrain_phase(task, config, pretrain_config, train_loader, val_loader):
    """Run the MLM pre-training phase for encoder.

    Args:
        task: The unified task model.
        config: Full Hydra configuration.
        pretrain_config: Pre-training specific configuration.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
    """
    pretrain_steps = pretrain_config.get("steps", 10000)
    pretrain_lr = pretrain_config.get("lr", 1e-3)
    pretrain_warmup = pretrain_config.get("warmup_steps", 1000)
    pretrain_min_lr = pretrain_config.get("min_lr", 1e-5)

    log.info("=" * 70)
    log.info("PHASE 1: MLM Pre-training for Encoder")
    log.info(f"  Steps: {pretrain_steps}")
    log.info(f"  Learning rate: {pretrain_lr}")
    log.info(f"  Warmup steps: {pretrain_warmup}")
    log.info(f"  Min LR: {pretrain_min_lr}")
    log.info("  Mode: MLM only (bridge/decoder/head frozen)")
    log.info("  Dataset: same train/val loaders as fine-tuning")
    log.info("=" * 70)

    # Store pretrain optimizer/scheduler config in task hparams
    task.hparams['pretrain_optimizer_config'] = {
        'name': 'AdamW',
        'args': {
            'lr': pretrain_lr,
            'weight_decay': pretrain_config.get("weight_decay", 0.01),
        }
    }
    task.hparams['pretrain_scheduler_config'] = {
        'name': pretrain_config.get("scheduler", "cosine_warmup"),
        'warmup_steps': pretrain_warmup,
        'min_lr': pretrain_min_lr,
    }

    # Switch task to pre-training phase
    task.set_pretrain_phase()

    # Create pre-training trainer
    pretrain_trainer = create_pretrain_trainer(config, pretrain_config)

    # Run pre-training
    pretrain_trainer.fit(task, train_loader, val_loader)

    log.info("=" * 70)
    log.info("PHASE 1 COMPLETE: MLM Pre-training finished.")
    log.info("Switching to fine-tuning phase...")
    log.info("=" * 70)

    # Switch task to fine-tuning phase (unfreezes all parameters)
    task.set_finetune_phase()

    # Clean up wandb run so finetune phase can create a fresh one
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


def fsspec_exists(filename: str) -> bool:
    """Check if a file exists using fsspec."""
    try:
        fs, _ = fsspec.core.url_to_fs(filename)
        return fs.exists(filename)
    except Exception:
        return False


def _log_training_results(trainer: pl.Trainer, config: DictConfig):
    """Log training results to file."""
    best_ckpt_callback = None
    for callback in trainer.callbacks:
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            if callback.monitor is not None:
                best_ckpt_callback = callback
                break

    if best_ckpt_callback:
        best_model_path = best_ckpt_callback.best_model_path
        best_model_score = best_ckpt_callback.best_model_score

        log.info(f"Best model saved at: {best_model_path}")
        log.info(f"Best score ({best_ckpt_callback.monitor}): {best_model_score}")

        try:
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            log_file = os.path.join(output_dir, "training_summary.txt")

            with open(log_file, "w") as f:
                f.write("Training Summary\n")
                f.write("=" * 50 + "\n")
                f.write(f"Best Model Path: {best_model_path}\n")
                f.write(f"Monitor Metric: {best_ckpt_callback.monitor}\n")
                f.write(f"Best Score: {best_model_score}\n")
                f.write(f"Config Name: {config.get('name', 'N/A')}\n")
                f.write("\nFinal Metrics:\n")
                f.write("-" * 30 + "\n")
                for k, v in trainer.callback_metrics.items():
                    f.write(f"{k}: {v}\n")

            log.info(f"Training summary saved to: {log_file}")
        except Exception as e:
            log.warning(f"Failed to save training summary: {e}")
    else:
        log.warning("No ModelCheckpoint callback with monitor found.")


# ----------- Entry Point -----------

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    """Main entry point."""
    train(config)


if __name__ == "__main__":
    main()
