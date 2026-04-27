"""
MLM Pre-training Task for Caduceus DNA Language Model.

This module implements a PyTorch Lightning task for self-supervised
pre-training of the Caduceus encoder using Masked Language Modeling (MLM)
on reference genome data.

Architecture:
    Input tokens (B, L=131072) with MLM masking
    → Caduceus Embedding
    → Caduceus Encoder Layers (Mamba SSM + Attention)
    → LM Head (Linear projection to vocab_size)
    → Cross-Entropy Loss on masked positions

After pre-training, the model is saved in HuggingFace format and can be
loaded by the sandwich fine-tuning pipeline via `pretrained_ckpt_path`.

Usage:
    1. Pre-train:  python pretrain_mlm.py
    2. Fine-tune:  python train.py task.pretrained_ckpt_path=./outputs/mlm_pretrain/hf_model
"""

import copy
import logging
import os

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics import Metric

log = logging.getLogger(__name__)
for _level in ("debug", "info", "warning", "error", "critical"):
    setattr(log, _level, rank_zero_only(getattr(log, _level)))


# ─────────────────────────────────────────────────────────────────────────────
# TorchMetrics for MLM Pre-training (aligned with caduceus project)
# ─────────────────────────────────────────────────────────────────────────────

class Perplexity(Metric):
    """Perplexity metric: exp(average NLL), not average(exp(NLL)).

    This is the correct way to compute perplexity - accumulate log probs
    across all tokens, then exponentiate the mean.
    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_log_probs", default=torch.tensor(0.0, dtype=torch.float64),
                       dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64),
                       dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor,
               loss: torch.Tensor = None) -> None:
        """Update metric state.

        Args:
            preds: Model predictions (not used, kept for API compatibility)
            target: Ground truth targets
            loss: Pre-computed loss value (scalar tensor)
        """
        count = target.numel()
        if loss is None:
            # Should not happen, but compute loss if not provided
            loss = nn.functional.cross_entropy(
                preds.view(-1, preds.size(-1)),
                target.view(-1),
                ignore_index=4
            )
        self.total_log_probs += loss.double() * count
        self.count += count

    def compute(self) -> torch.Tensor:
        """Compute perplexity as exp(average loss)."""
        return torch.exp(self.total_log_probs / self.count)


class NumTokens(Metric):
    """Keep track of how many tokens we've processed."""
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64),
                       dist_reduce_fx="sum", persistent=True)

    def update(self, preds: torch.Tensor, target: torch.Tensor,
               loss: torch.Tensor = None) -> None:
        """Update token count."""
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Return total token count."""
        return self.count

    def reset(self):
        """Reset metric but preserve count (for epoch accumulation)."""
        count = self.count
        super().reset()
        self.count = count

    def _forward_reduce_state_update(self, *args, **kwargs) -> None:
        """Forward computation using single call to update."""
        self.update(*args, **kwargs)
        return self.compute()


class MLMPretrainTask(pl.LightningModule):
    """MLM pre-training task for DNA language model encoder.

    Trains a CaduceusForMaskedLM model using BERT-style masked language
    modeling on reference genome sequences.

    After training, call `save_pretrained(path)` to export the model in
    HuggingFace format for downstream fine-tuning.

    Args:
        encoder_config: Configuration for the Caduceus model.
        optimizer_config: Optimizer configuration dict.
        scheduler_config: LR scheduler configuration dict.
        n_pretrain_layers: If set, only train the first N encoder layers
            (freeze the rest). None = train all layers.
        mask_token_id: Token ID for [MASK]. Default 5.
        save_hf_path: Path to save HuggingFace format model after training.
    """

    def __init__(
        self,
        encoder_config,
        optimizer_config,
        scheduler_config,
        n_pretrain_layers=None,
        mask_token_id=3,
        pad_token_id=4,
        save_hf_path=None,
        **kwargs,
    ):
        super().__init__()

        # --- Sanitize configs for serialization ---
        if OmegaConf.is_config(encoder_config):
            hparams_encoder_config = OmegaConf.to_container(encoder_config, resolve=True)
        elif isinstance(encoder_config, dict):
            hparams_encoder_config = copy.deepcopy(encoder_config)
        else:
            hparams_encoder_config = encoder_config

        if isinstance(hparams_encoder_config, dict):
            cfg_inner = hparams_encoder_config.get('config')
            if cfg_inner is not None and not isinstance(cfg_inner, dict) and hasattr(cfg_inner, 'to_dict'):
                hparams_encoder_config['config'] = cfg_inner.to_dict()

        if OmegaConf.is_config(optimizer_config):
            optimizer_config = OmegaConf.to_container(optimizer_config, resolve=True)
        if OmegaConf.is_config(scheduler_config):
            scheduler_config = OmegaConf.to_container(scheduler_config, resolve=True)

        self.save_hyperparameters({
            "encoder_config": hparams_encoder_config,
            "optimizer_config": optimizer_config,
            "scheduler_config": scheduler_config,
            "n_pretrain_layers": n_pretrain_layers,
            "mask_token_id": mask_token_id,
            "pad_token_id": pad_token_id,
            "save_hf_path": save_hf_path,
            **kwargs,
        })

        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.n_pretrain_layers = n_pretrain_layers
        self.save_hf_path = save_hf_path

        # --- Initialize CaduceusForMaskedLM ---
        self._init_model(encoder_config)

        # --- Selectively freeze layers if n_pretrain_layers is set ---
        if n_pretrain_layers is not None:
            self._freeze_post_layers(n_pretrain_layers)

        # --- Loss ---
        # Use pad_token_id as ignore_index to match caduceus project
        # Non-masked positions in target are set to pad_token_id (4)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

        # --- Metrics tracking (aligned with caduceus project) ---
        # Perplexity: exp(average loss), computed correctly via torchmetric
        self.perplexity_metric = Perplexity()
        self.num_tokens_metric = NumTokens()
        self._val_losses = []
        self._epoch_tokens = 0

    # ======================================================================
    # Model Initialization
    # ======================================================================

    def _init_model(self, encoder_config):
        """Initialize CaduceusForMaskedLM from config."""
        from src.caduceus.configuration_caduceus import CaduceusConfig
        from src.caduceus.modeling_caduceus import CaduceusForMaskedLM

        config_obj = (
            encoder_config.get('config') if isinstance(encoder_config, dict)
            else getattr(encoder_config, 'config', None)
        )

        if isinstance(config_obj, CaduceusConfig):
            caduceus_config = config_obj
        elif config_obj is not None:
            # Convert to native dict first
            if isinstance(config_obj, DictConfig):
                config_dict = OmegaConf.to_container(config_obj, resolve=True)
            elif isinstance(config_obj, dict):
                config_dict = copy.deepcopy(config_obj)
            else:
                raise ValueError(f"Cannot process config of type {type(config_obj)}")

            # Remove Hydra / HF internal keys
            excluded_keys = {
                '_target_', '_recursive_', '_convert_', '_args_',
                'return_dict', 'output_hidden_states', 'output_attentions',
                'torchscript', 'torch_dtype', 'use_bfloat16', 'tf_legacy_loss',
                'pruned_heads', 'tie_word_embeddings', 'chunk_size_feed_forward',
                'is_encoder_decoder', 'is_decoder', 'cross_attention_hidden_size',
                'add_cross_attention', 'tie_encoder_decoder',
                'architectures', 'finetuning_task', 'id2label', 'label2id',
                'tokenizer_class', 'prefix', 'bos_token_id', 'pad_token_id',
                'eos_token_id', 'sep_token_id', 'decoder_start_token_id',
                'task_specific_params', 'problem_type', '_name_or_path',
                'transformers_version', 'model_type', '_commit_hash',
                'attn_implementation',
            }
            import ast
            filtered = {}
            for k, v in config_dict.items():
                if k not in excluded_keys:
                    if isinstance(v, str) and v.startswith('{'):
                        try:
                            filtered[k] = ast.literal_eval(v)
                        except (ValueError, SyntaxError):
                            filtered[k] = v
                    else:
                        filtered[k] = v
            caduceus_config = CaduceusConfig(**filtered)
        else:
            raise ValueError("Could not resolve Caduceus configuration.")

        self.model = CaduceusForMaskedLM(caduceus_config)
        self._d_model = caduceus_config.d_model
        self._vocab_size = caduceus_config.vocab_size
        self._n_layer = caduceus_config.n_layer

        log.info(
            f"Initialized CaduceusForMaskedLM: d_model={self._d_model}, "
            f"vocab_size={self._vocab_size}, n_layer={self._n_layer}"
        )

    def _freeze_post_layers(self, n_pretrain_layers):
        """Freeze encoder layers after the first n_pretrain_layers.

        This trains only the embedding + first N layers + LM head,
        matching the layers that will serve as pre-downsample layers
        in the sandwich architecture.
        """
        all_layers = list(self.model.caduceus.backbone.layers)
        total_layers = len(all_layers)

        if n_pretrain_layers >= total_layers:
            log.info(
                f"n_pretrain_layers ({n_pretrain_layers}) >= total layers ({total_layers}). "
                f"Training ALL layers."
            )
            return

        # Freeze layers [n_pretrain_layers:]
        for i, layer in enumerate(all_layers[n_pretrain_layers:], start=n_pretrain_layers):
            for param in layer.parameters():
                param.requires_grad = False

        # Count params
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        log.info(
            f"Partial pre-training: first {n_pretrain_layers}/{total_layers} encoder layers trainable. "
            f"Trainable: {trainable:,}, Frozen: {frozen:,}"
        )

    # ======================================================================
    # Forward & Loss
    # ======================================================================

    def forward(self, input_ids, labels=None):
        """Forward pass through CaduceusForMaskedLM.

        Args:
            input_ids: (B, L) masked token IDs.
            labels: (B, L) target labels (pad_token_id for non-masked positions).

        Returns:
            MaskedLMOutput with loss and logits.
        """
        return self.model(input_ids=input_ids, labels=labels)

    # ======================================================================
    # Training & Validation Steps
    # ======================================================================

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch  # (B, L), (B, L)

        # Forward WITHOUT labels — compute loss ourselves to avoid
        # missing cross_entropy helper in CaduceusForMaskedLM
        outputs = self(input_ids, labels=None)
        logits = outputs.logits  # (B, L, vocab_size)

        # Compute loss with our own CrossEntropyLoss (ignore_index=pad_token_id)
        loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        # Compute accuracy on masked positions
        with torch.no_grad():
            preds = logits.argmax(dim=-1)  # (B, L)
            mask = labels != self.pad_token_id
            if mask.any():
                correct = (preds[mask] == labels[mask]).float().mean()
            else:
                correct = torch.tensor(0.0, device=self.device)

        # Update torchmetrics (aligned with caduceus project)
        self.perplexity_metric(logits, labels, loss=loss)
        self.num_tokens_metric(logits, labels, loss=loss)
        self._epoch_tokens += labels.numel()

        # Log metrics
        self.log("pretrain/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("pretrain/accuracy", correct, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # Perplexity and num_tokens are logged in training_epoch_end via torchmetrics

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch

        outputs = self(input_ids, labels=None)
        logits = outputs.logits

        loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = labels != self.pad_token_id
            if mask.any():
                correct = (preds[mask] == labels[mask]).float().mean()
            else:
                correct = torch.tensor(0.0, device=self.device)

        # Update torchmetrics
        self.perplexity_metric(logits, labels, loss=loss)
        self.num_tokens_metric(logits, labels, loss=loss)

        # Log metrics
        self.log("pretrain_val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("pretrain_val/accuracy", correct, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # Perplexity and num_tokens logged in validation_epoch_end

        return {"loss": loss, "accuracy": correct}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    # ======================================================================
    # Epoch End Hooks (log torchmetrics)
    # ======================================================================

    def on_train_epoch_end(self):
        """Log training torchmetrics at epoch end."""
        # Compute and log perplexity (correct: exp(average loss))
        ppl = self.perplexity_metric.compute()
        self.log("pretrain/perplexity", ppl, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("pretrain/num_tokens", self.num_tokens_metric.compute().long(),
                 on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        # Reset metrics for next epoch
        self.perplexity_metric.reset()
        self.num_tokens_metric.reset()
        self._epoch_tokens = 0

    def on_validation_epoch_end(self):
        """Log validation torchmetrics at epoch end."""
        ppl = self.perplexity_metric.compute()
        self.log("pretrain_val/perplexity", ppl, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("pretrain_val/num_tokens", self.num_tokens_metric.compute().long(),
                 on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        # Reset metrics for next validation
        self.perplexity_metric.reset()
        self.num_tokens_metric.reset()

    # ======================================================================
    # Optimizer & Scheduler
    # ======================================================================

    def configure_optimizers(self):
        """Configure optimizer and LR scheduler."""
        from src.utils.scheduler import get_scheduler

        params = filter(lambda p: p.requires_grad, self.parameters())

        opt_cfg = self.hparams.optimizer_config
        opt_name = opt_cfg['name'] if isinstance(opt_cfg, dict) else opt_cfg.name
        opt_args = opt_cfg['args'] if isinstance(opt_cfg, dict) else opt_cfg.args
        opt_args = dict(opt_args)

        optimizer_cls = getattr(torch.optim, opt_name)
        optimizer = optimizer_cls(params, **opt_args)

        sched_cfg = self.hparams.scheduler_config
        if sched_cfg is None:
            return optimizer

        sched_name = sched_cfg['name'] if isinstance(sched_cfg, dict) else getattr(sched_cfg, 'name', 'constant')
        if sched_name in ['constant', 'none', None]:
            return optimizer

        if isinstance(sched_cfg, dict):
            warmup_steps = sched_cfg.get('warmup_steps', 0)
            min_lr = sched_cfg.get('min_lr', 0.0)
        else:
            warmup_steps = getattr(sched_cfg, 'warmup_steps', 0)
            min_lr = getattr(sched_cfg, 'min_lr', 0.0)

        total_steps = self.trainer.estimated_stepping_batches
        log.info(
            f"Scheduler: name={sched_name}, warmup={warmup_steps}, "
            f"total_steps={total_steps}, min_lr={min_lr}"
        )

        lr_scheduler_config = get_scheduler(
            name=sched_name,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
        )

        if lr_scheduler_config is None:
            return optimizer

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    # ======================================================================
    # Save Pretrained Model
    # ======================================================================

    def save_pretrained(self, save_path):
        """Save the trained model in HuggingFace format.

        The saved model can be loaded by the sandwich fine-tuning pipeline:
            task.pretrained_ckpt_path=<save_path>

        Args:
            save_path: Directory to save the model files.
        """
        os.makedirs(save_path, exist_ok=True)

        # Save as HuggingFace format
        self.model.save_pretrained(save_path)

        # Also save the underlying Caduceus config
        self.model.config.save_pretrained(save_path)

        log.info(f"Pretrained model saved to: {save_path}")
        log.info(f"  → Use in fine-tuning: task.pretrained_ckpt_path={save_path}")

    def on_train_end(self):
        """Auto-save the HuggingFace model at the end of training (rank 0 only)."""
        if not self.trainer.is_global_zero:
            return

        # Determine save path
        if self.save_hf_path:
            save_path = self.save_hf_path
        else:
            try:
                import hydra
                output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
                save_path = os.path.join(output_dir, "hf_model")
            except Exception:
                save_path = os.path.join(".", "hf_model")

        # Attempt to save in HuggingFace format
        # If this fails due to shared weights, fall back to PyTorch checkpoint
        try:
            self.save_pretrained(save_path)
        except RuntimeError as e:
            if "shared tensors" in str(e).lower() or "shared weights" in str(e).lower():
                # Shared weights detected - save as PyTorch checkpoint instead
                pt_path = os.path.join(save_path, "pytorch_checkpoint.pt")
                os.makedirs(os.path.dirname(pt_path) if os.path.dirname(pt_path) else ".", exist_ok=True)
                torch.save(
                    {
                        "state_dict": self.state_dict(),
                        "hparams": dict(self.hparams),
                    },
                    pt_path,
                )
                log.warning(
                    f"Skipping HuggingFace format save due to shared weights: {e}. "
                    f"Saved PyTorch checkpoint to: {pt_path}"
                )
            else:
                # Re-raise if not a shared weights error
                raise
