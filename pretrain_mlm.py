"""MLM Pre-training Entry Point for SF-Fuse.

This script handles:
- Configuration loading via Hydra
- Reference genome dataset creation (FASTA or H5)
- MLM model/task instantiation (CaduceusForMaskedLM)
- WandB logging integration
- Pre-training with MLM objective
- Saving pretrained model in HuggingFace format for downstream fine-tuning

After pre-training, the saved model can be used in fine-tuning:
    python train.py task.pretrained_ckpt_path=./outputs/mlm_pretrain/hf_model

Usage:
    python pretrain_mlm.py                          # Default config
    python pretrain_mlm.py task.dataset.data_file=/path/to/genome.fa
    python pretrain_mlm.py task.optimizer_config.args.lr=1e-3
"""

# ---------------------------------------------------------------------------
# Compatibility shim (same as train.py)
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
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from torch.utils.data import DataLoader

from src.dataloaders.genome_mlm_dataset import (
    ReferenceGenomeMLMDataset,
    mlm_worker_init_fn,
)
from src.caduceus.tokenization_caduceus import CaduceusTokenizer
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


# ----------- Custom WandB Logger (reused from train.py) -----------

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
    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)
        return get_experiment() or DummyExperiment()
    return experiment


class CustomWandbLogger(WandbLogger):
    """WandB logger with retry logic."""
    
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

def create_pretrain_trainer(config: DictConfig, has_val_data: bool = True) -> pl.Trainer:
    """Create PyTorch Lightning Trainer for MLM pre-training.
    
    Args:
        config: Hydra configuration.
        has_val_data: Whether validation data is available. If False,
            callbacks that monitor validation metrics are disabled.
    """
    callbacks: List[pl.Callback] = []
    logger = None

    # WandB Logging
    if config.get("wandb") is not None and config.wandb.get("mode") != "disabled":
        log.info("Initializing WandB logger for MLM pre-training...")
        try:
            logger = CustomWandbLogger(
                config=OmegaConf.to_container(config, resolve=True),
                settings=wandb.Settings(start_method="fork"),
                **config.wandb,
            )
        except Exception as e:
            log.warning(f"Failed to initialize WandB logger: {e}. Continuing without WandB.")
            logger = None

    # Callbacks that require validation data
    _val_dependent_callbacks = {
        "early_stopping", "model_checkpoint_best", "val_every_n_global_steps"
    }

    # Instantiate callbacks
    if "callbacks" in config:
        for cb_name, cb_conf in config.callbacks.items():
            if cb_conf is None:
                continue
            if logger is None and cb_name in ["learning_rate_monitor"]:
                log.info(f"Skipping callback <{cb_name}> because WandB is disabled")
                continue
            # Skip val-dependent callbacks if no validation data
            if not has_val_data and cb_name in _val_dependent_callbacks:
                log.info(f"Skipping callback <{cb_name}> because no validation data provided")
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

    # Handle strategy
    strategy_conf = trainer_conf.get("strategy")
    if strategy_conf == "ddp":
        log.info("Using DDPStrategy with find_unused_parameters=True")
        trainer_conf["strategy"] = DDPStrategy(find_unused_parameters=True)
    elif isinstance(strategy_conf, dict) and "_target_" in strategy_conf:
        log.info(f"Instantiating strategy: {strategy_conf['_target_']}")
        trainer_conf["strategy"] = hydra.utils.instantiate(strategy_conf)

    # Create trainer
    log.info("Creating PyTorch Lightning Trainer for MLM pre-training...")
    trainer = pl.Trainer(
        **trainer_conf,
        callbacks=callbacks,
        logger=logger,
    )

    return trainer


# ----------- Main Pre-training Function -----------

def pretrain(config: DictConfig):
    """Main MLM pre-training function."""
    # Process config
    config = process_config(config)
    print_config(config, resolve=True)

    # Set seed
    seed = config.get("seed", 42)
    log.info(f"Setting random seed: {seed}")
    pl.seed_everything(seed, workers=True)

    # --- Create datasets ---
    log.info("Creating MLM datasets...")
    dataset_cfg = config.task.dataset
    batch_size = dataset_cfg.get("batch_size", 1)
    num_workers = dataset_cfg.get("num_workers", 4)
    seq_length = dataset_cfg.get("seq_length", 131072)

    # --- Token-based Batch Size (Caduceus-style) ---
    # Priority:
    #   1. Explicit global_batch_size from config (backward compatible)
    #   2. tokens_per_batch from config (Caduceus-style)
    #   3. Default: 1,048,576 tokens per update
    tokens_per_batch = config.train.get("tokens_per_batch", None)
    global_batch_size = config.get("train", {}).get("global_batch_size", None)
    num_devices = config.trainer.get("devices", 1)
    if isinstance(num_devices, (list, tuple)):
        num_devices = len(num_devices)

    # Compute global_batch_size from tokens_per_batch if not explicitly set
    if global_batch_size is None or global_batch_size <= 0:
        if tokens_per_batch is None:
            tokens_per_batch = 1048576  # Default: 2^20 tokens per update
        global_batch_size = tokens_per_batch // seq_length
        log.info(
            f"Token-based batch size: "
            f"tokens_per_batch={tokens_per_batch} / seq_length={seq_length} = "
            f"global_batch_size={global_batch_size}"
        )
    elif tokens_per_batch is not None:
        # global_batch_size was explicitly set, but also tokens_per_batch - verify consistency
        expected_tokens = global_batch_size * seq_length
        if expected_tokens != tokens_per_batch:
            log.info(
                f"global_batch_size explicitly set to {global_batch_size}, "
                f"overriding tokens_per_batch={tokens_per_batch} "
                f"(would have given {tokens_per_batch // seq_length} global batch size)"
            )

    # Log gradient accumulation (from config, may be overridden by Python calculation)
    accumulate_grad_batches = config.trainer.get("accumulate_grad_batches", 1)
    effective_batch_size = batch_size * num_devices * accumulate_grad_batches
    total_tokens_per_update = effective_batch_size * seq_length
    log.info(
        f"Batch configuration: micro={batch_size}, gpus={num_devices}, "
        f"grad_accum={accumulate_grad_batches}, effective={effective_batch_size}, "
        f"total_tokens_per_update={total_tokens_per_update:,}"
    )

    # Initialize tokenizer for Caduceus
    log.info("Initializing CaduceusTokenizer...")
    tokenizer = CaduceusTokenizer(model_max_length=seq_length)

    # Build dataset kwargs (Caduceus-style BED file approach)
    common_kwargs = dict(
        max_length=seq_length,
        mlm=dataset_cfg.get("mlm", True),
        mlm_probability=dataset_cfg.get("mlm_probability", 0.15),
        tokenizer=tokenizer,
        rc_augment=dataset_cfg.get("rc_augment", False),
    )

    # Training dataset
    train_dataset = ReferenceGenomeMLMDataset(
        bed_file=dataset_cfg.train_bed_file,
        fasta_file=dataset_cfg.train_fasta_file,
        split=dataset_cfg.get("train_split", "train"),
        **common_kwargs,
    )

    # Validation dataset (optional)
    val_dataset = None
    val_bed_file = dataset_cfg.get("val_bed_file", None)
    val_fasta_file = dataset_cfg.get("val_fasta_file", None)
    val_split = dataset_cfg.get("val_split", "valid")
    if val_bed_file and val_bed_file != "null" and val_fasta_file:
        val_dataset = ReferenceGenomeMLMDataset(
            bed_file=val_bed_file,
            fasta_file=val_fasta_file,
            split=val_split,
            **common_kwargs,
        )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=mlm_worker_init_fn,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=mlm_worker_init_fn,
        )

    log.info(f"Train dataset: {len(train_dataset)} samples/epoch")
    if val_dataset:
        log.info(f"Val dataset: {len(val_dataset)} samples/epoch")

    # --- Create task/model ---
    log.info("Instantiating MLM pre-training task...")
    task = hydra.utils.instantiate(config.task)
    print_model_info(task)

    # --- Create trainer ---
    has_val_data = val_dataset is not None
    trainer = create_pretrain_trainer(config, has_val_data=has_val_data)

    # --- Resume from checkpoint ---
    ckpt_path = None
    ckpt_file = config.get("train", {}).get("ckpt")
    if ckpt_file and ckpt_file != "null":
        if fsspec_exists(ckpt_file):
            log.info(f"Resuming MLM pre-training from: {ckpt_file}")
            ckpt_path = ckpt_file
        else:
            log.info(f"No checkpoint found at {ckpt_file}. Pre-training from scratch.")

    # --- Train ---
    log.info("Starting MLM pre-training...")
    trainer.fit(task, train_loader, val_loader, ckpt_path=ckpt_path)

    # Note: HuggingFace model is auto-saved via on_train_end hook (rank 0 only).
    # No need for explicit save here.

    log.info("MLM pre-training complete!")


def fsspec_exists(filename: str) -> bool:
    """Check if a file exists (local or remote)."""
    try:
        fs, _ = fsspec.core.url_to_fs(filename)
        return fs.exists(filename)
    except Exception:
        return False


# ----------- Entry Point -----------

@hydra.main(config_path="configs", config_name="pretrain_mlm.yaml", version_base=None)
def main(config: DictConfig):
    """MLM pre-training entry point."""
    pretrain(config)


if __name__ == "__main__":
    main()
