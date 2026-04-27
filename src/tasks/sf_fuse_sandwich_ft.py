"""
SF-Fuse fine-tuning task with Sandwich (mid-sequence downsample) architecture.

Architecture:
    Input tokens (B, L=131072)
    → Caduceus Embedding
    → First n_pre_layers at full 131k length (long-range context capture)
    → CNN Bridge [128x downsampling: 131072 → 1024]
    → Remaining n_post_layers at 1024 length (efficient processing)
    → Simple Projection Head [center crop to 896 + MLP]
    → (B, 896, num_tracks)

Key design choices:
    - Uses a SINGLE Caduceus encoder model, splitting its layers into
      pre-downsample and post-downsample stages
    - Pre-downsample layers (typically Mamba SSM) capture long-range
      dependencies at full 131k resolution
    - CNN bridge compresses sequence via residual Conv + MaxPool blocks
    - Post-downsample layers process the shorter sequence more efficiently
    - Supports loading pretrained models directly (no weight splitting needed)
    - No MLM joint training (not applicable after mid-sequence downsampling)

Comparison with other architectures:
    Original (sf_fuse_ft):
        All layers at 131k → AvgPool(128) in head → 896 bins
        Pro: Full-length processing. Con: Very expensive

    CNN Stem (sf_fuse_cnn_stem_ft):
        CNN ↓128x at front → All layers at 1024 → 896 bins
        Pro: Fast. Con: No full-length processing

    Sandwich (this file):
        N1 layers at 131k → CNN ↓128x → N2 layers at 1024 → 896 bins
        Pro: Balance of long-range capture and efficiency
"""

import copy
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.utilities import rank_zero_only

from src.tasks.track_utils import load_track_names, log_top_tracks, save_pcc_ranking

log = logging.getLogger(__name__)
for _level in ("debug", "info", "warning", "error", "critical"):
    setattr(log, _level, rank_zero_only(getattr(log, _level)))


class SFFuseSandwichTask(pl.LightningModule):
    """SF-Fuse fine-tuning task with sandwich (mid-downsample) architecture.

    Splits the Caduceus encoder layers into two stages:
    1. Pre-downsample stage: first n_pre_layers at full sequence length
    2. CNN bridge: downsample hidden states (e.g., 131072 → 1024)
    3. Post-downsample stage: remaining layers at compressed length
    4. Simple projection head for track prediction

    Args:
        encoder_config: Configuration for the DNA-LM backbone (Caduceus).
        bridge_config: Configuration for CNNDownsampleBridge.
        head_config: Configuration for SimpleProjectionHead.
        n_pre_layers: Number of encoder layers before the bridge (at full length).
        optimizer_config: Optimizer configuration.
        scheduler_config: LR scheduler configuration.
        freeze_encoder: Whether to freeze encoder weights.
        pretrained_ckpt_path: Path to pretrained encoder checkpoint.
    """

    def __init__(
        self,
        encoder_config,
        bridge_config,
        head_config,
        n_pre_layers,
        optimizer_config,
        scheduler_config,
        freeze_encoder=False,
        pretrained_ckpt_path=None,
        track_names_file=None,
        **kwargs,
    ):
        super().__init__()
        
        # Load track names
        num_tracks = head_config.get('num_tracks', 5313) if isinstance(head_config, dict) else getattr(head_config, 'num_tracks', 5313)
        self.track_names = load_track_names(track_names_file, num_tracks)

        # --- Extract key parameters ---
        if hasattr(encoder_config, 'config') and hasattr(encoder_config.config, 'vocab_size'):
            self._vocab_size = encoder_config.config.vocab_size
            self._d_model = encoder_config.config.d_model
        elif isinstance(encoder_config, dict) and 'config' in encoder_config:
            cfg = encoder_config['config']
            if isinstance(cfg, dict):
                self._vocab_size = cfg.get('vocab_size', 12)
                self._d_model = cfg.get('d_model', 256)
            elif hasattr(cfg, 'vocab_size'):
                self._vocab_size = cfg.vocab_size
                self._d_model = cfg.d_model
            else:
                self._vocab_size = 12
                self._d_model = 256
        else:
            self._vocab_size = getattr(encoder_config, 'vocab_size', 12)
            self._d_model = getattr(encoder_config, 'd_model', 256)

        self.n_pre_layers = n_pre_layers

        # --- Sanitize configs for serialization ---
        runtime_encoder_config = encoder_config

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

        if OmegaConf.is_config(bridge_config):
            bridge_config = OmegaConf.to_container(bridge_config, resolve=True)
        elif not isinstance(bridge_config, dict):
            bridge_config = dict(bridge_config)
        else:
            bridge_config = copy.deepcopy(bridge_config)

        if OmegaConf.is_config(head_config):
            head_config = OmegaConf.to_container(head_config, resolve=True)
        elif not isinstance(head_config, dict):
            head_config = dict(head_config)
        else:
            head_config = copy.deepcopy(head_config)

        if OmegaConf.is_config(optimizer_config):
            optimizer_config = OmegaConf.to_container(optimizer_config, resolve=True)
        if OmegaConf.is_config(scheduler_config):
            scheduler_config = OmegaConf.to_container(scheduler_config, resolve=True)

        self.save_hyperparameters({
            "encoder_config": hparams_encoder_config,
            "bridge_config": bridge_config,
            "head_config": head_config,
            "n_pre_layers": n_pre_layers,
            "optimizer_config": optimizer_config,
            "scheduler_config": scheduler_config,
            "freeze_encoder": freeze_encoder,
            "pretrained_ckpt_path": pretrained_ckpt_path,
            "vocab_size": self._vocab_size,
            "d_model": self._d_model,
            **kwargs,
        })

        # --- 1. Initialize Encoder ---
        # NOTE: This may update self._d_model and self._vocab_size from pretrained config
        self._init_encoder(runtime_encoder_config, pretrained_ckpt_path)

        # Validate n_pre_layers
        total_layers = len(list(self.encoder.backbone.layers))
        if self.n_pre_layers < 0 or self.n_pre_layers > total_layers:
            raise ValueError(
                f"n_pre_layers ({self.n_pre_layers}) must be in [0, {total_layers}]. "
                f"Encoder has {total_layers} layers total."
            )
        n_post_layers = total_layers - self.n_pre_layers
        log.info(
            f"Sandwich architecture: {self.n_pre_layers} pre-downsample layers "
            f"(at full length) + {n_post_layers} post-downsample layers "
            f"(at compressed length)"
        )

        # --- 2. Initialize CNN Bridge ---
        bridge_config['d_model'] = self._d_model
        from src.models.cnn_bridge import CNNDownsampleBridge
        self.bridge = CNNDownsampleBridge(**bridge_config)
        log.info(
            f"CNN Bridge initialized: d_model={self._d_model}, "
            f"downsample_factor={self.bridge.total_downsample_factor}x, "
            f"stages={self.bridge.num_downsample_stages}"
        )

        # --- 3. Initialize Simple Projection Head ---
        head_config['d_model'] = self._d_model
        if head_config.get('hidden_dim') is None:
            head_config['hidden_dim'] = self._d_model * 2

        from src.models.cnn_stem import SimpleProjectionHead
        self.head = SimpleProjectionHead(**head_config)

        # Update hparams with final configs
        self.hparams.bridge_config = bridge_config
        self.hparams.head_config = head_config

        # --- 4. Loss Function ---
        self.loss_fn = nn.PoissonNLLLoss(log_input=False, full=True)

        # --- 5. Freeze encoder if requested ---
        if freeze_encoder:
            self._freeze_encoder()
        else:
            log.info("Full fine-tuning enabled: all parameters will be updated.")

    # ======================================================================
    # Encoder Initialization (reused from sf_fuse_ft / sf_fuse_cnn_stem_ft)
    # ======================================================================

    def _init_encoder(self, encoder_config, pretrained_ckpt_path):
        """Initialize encoder from pretrained checkpoint or from scratch."""
        if pretrained_ckpt_path:
            log.info(f"Loading pretrained DNA-LM from: {pretrained_ckpt_path}")
            self._load_pretrained_encoder(pretrained_ckpt_path)
        else:
            log.info("No pretrained checkpoint. Initializing encoder from scratch...")
            self._init_from_scratch(encoder_config)

    def _load_pretrained_encoder(self, pretrained_ckpt_path):
        """Load pretrained encoder from HF format checkpoint."""
        import os
        model_type = self._detect_model_type(pretrained_ckpt_path)
        log.info(f"Detected model type: {model_type}")

        if model_type == "caduceus":
            self._load_caduceus_pretrained(pretrained_ckpt_path)
        else:
            self._load_hf_automodel(pretrained_ckpt_path)

    def _detect_model_type(self, path):
        """Detect model type from config.json."""
        import os, json
        config_path = os.path.join(path, "config.json") if os.path.isdir(path) else None
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            return config_dict.get("model_type", "unknown")
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(path, trust_remote_code=True)
            return getattr(config, "model_type", "unknown")
        except Exception:
            return "unknown"

    def _load_caduceus_pretrained(self, pretrained_ckpt_path):
        """Load pretrained Caduceus model.

        For the sandwich architecture, we keep fourier_max_seq_len at the
        full sequence length since the pre-downsample layers process full-length
        sequences. Post-downsample attention layers will still work correctly
        with shorter sequences (the FourierEmbedding adapts to actual seq_len).
        """
        from src.caduceus.configuration_caduceus import CaduceusConfig
        from src.caduceus.modeling_caduceus import CaduceusForMaskedLM, Caduceus

        pretrained_config = CaduceusConfig.from_pretrained(pretrained_ckpt_path)

        # Update internal parameters from pretrained config
        self._d_model = pretrained_config.d_model
        self._vocab_size = pretrained_config.vocab_size
        log.info(f"Updated from pretrained: d_model={self._d_model}, vocab_size={self._vocab_size}")

        # --- Keep fourier_max_seq_len at full length for pre-downsample attention ---
        target_max_seq_len = self._get_target_max_seq_len()
        original_fourier_len = getattr(pretrained_config, 'fourier_max_seq_len', 16384)
        if target_max_seq_len > original_fourier_len:
            log.info(
                f"Updating fourier_max_seq_len: {original_fourier_len} -> {target_max_seq_len} "
                f"(pre-downsample layers need full-length support)"
            )
            pretrained_config.fourier_max_seq_len = target_max_seq_len
        else:
            log.info(f"fourier_max_seq_len: {original_fourier_len} (unchanged)")

        self.hparams.d_model = self._d_model
        self.hparams.vocab_size = self._vocab_size

        # Try loading as CaduceusForMaskedLM first
        try:
            model = CaduceusForMaskedLM.from_pretrained(
                pretrained_ckpt_path, config=pretrained_config
            )
            self.encoder = model.caduceus
            log.info("Loaded pretrained CaduceusForMaskedLM, extracted backbone.")
        except Exception as e:
            log.warning(f"Failed to load as CaduceusForMaskedLM: {e}")
            try:
                self.encoder = Caduceus.from_pretrained(
                    pretrained_ckpt_path, config=pretrained_config
                )
                log.info("Loaded pretrained Caduceus model.")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load Caduceus from {pretrained_ckpt_path}. "
                    f"CaduceusForMaskedLM error: {e}, Caduceus error: {e2}"
                )

    def _get_target_max_seq_len(self):
        """Get the target maximum sequence length for fine-tuning.

        For the sandwich architecture, this should be the full input sequence
        length since the pre-downsample layers process full-length sequences.
        """
        enc_config = self.hparams.get('encoder_config', {})
        if isinstance(enc_config, dict):
            inner_config = enc_config.get('config', {})
            if isinstance(inner_config, dict):
                fourier_len = inner_config.get('fourier_max_seq_len')
                if fourier_len is not None:
                    return fourier_len
        # Default to SF-Fuse standard
        return 131072

    def _load_hf_automodel(self, pretrained_ckpt_path):
        """Load pretrained model using HuggingFace AutoModel."""
        from transformers import AutoModel, AutoConfig

        try:
            config = AutoConfig.from_pretrained(pretrained_ckpt_path, trust_remote_code=True)
            if hasattr(config, 'd_model'):
                self._d_model = config.d_model
            if hasattr(config, 'vocab_size'):
                self._vocab_size = config.vocab_size

            self.hparams.d_model = self._d_model
            self.hparams.vocab_size = self._vocab_size

            self.encoder = AutoModel.from_pretrained(
                pretrained_ckpt_path, trust_remote_code=True
            )

            # Strip wrapper to get Caduceus backbone
            # IMPORTANT: We need the Caduceus object (which has .backbone),
            # NOT the CaduceusMixerModel directly, because _sandwich_forward
            # calls self.encoder.backbone.
            from src.caduceus.modeling_caduceus import Caduceus as _CaduceusModel
            if hasattr(self.encoder, 'caduceus'):
                self.encoder = self.encoder.caduceus  # CaduceusForMaskedLM → Caduceus
            elif isinstance(self.encoder, _CaduceusModel):
                pass  # Already the correct type
            elif hasattr(self.encoder, 'model'):
                self.encoder = self.encoder.model

            log.info("Loaded pretrained encoder via AutoModel.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {pretrained_ckpt_path}: {e}")

    def _init_from_scratch(self, encoder_config):
        """Initialize encoder from scratch using config."""
        _name = (
            encoder_config.get('_name_') if isinstance(encoder_config, dict)
            else getattr(encoder_config, '_name_', '')
        )

        config_obj = (
            encoder_config.get('config') if isinstance(encoder_config, dict)
            else getattr(encoder_config, 'config', None)
        )

        if 'caduceus' in str(_name):
            self._init_caduceus_from_scratch(config_obj)
        else:
            raise ValueError(
                f"Unknown encoder type: {_name}. "
                f"Sandwich architecture requires Caduceus encoder."
            )

    def _init_caduceus_from_scratch(self, config_obj):
        """Initialize Caduceus model from config object or dict.

        NOTE: We avoid hydra.utils.instantiate for CaduceusConfig because
        Hydra may pass OmegaConf DictConfig objects for nested dicts
        (ssm_cfg, attn_cfg, initializer_cfg), which can trigger AssertionError
        in some versions of transformers' PretrainedConfig.__init__.
        Instead, we always convert to native Python dicts first.
        """
        from src.caduceus.configuration_caduceus import CaduceusConfig
        from src.caduceus.modeling_caduceus import Caduceus

        feature_config = None

        if isinstance(config_obj, CaduceusConfig):
            feature_config = config_obj
        elif config_obj is not None:
            # Always convert to native Python dict to avoid DictConfig issues
            if isinstance(config_obj, DictConfig):
                config_dict = OmegaConf.to_container(config_obj, resolve=True)
            elif isinstance(config_obj, dict):
                config_dict = copy.deepcopy(config_obj)
            else:
                config_dict = None

            if isinstance(config_dict, dict):
                # Remove Hydra internal keys and HF PretrainedConfig keys
                excluded_keys = {
                    # Hydra internal keys
                    '_target_', '_recursive_', '_convert_', '_args_',
                    # HF PretrainedConfig keys
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
                feature_config = CaduceusConfig(**filtered)
            else:
                raise ValueError(
                    f"Cannot convert config_obj (type={type(config_obj)}) "
                    f"to a dict for CaduceusConfig creation."
                )

        if feature_config is None:
            raise ValueError("Could not resolve Caduceus configuration.")

        self.encoder = Caduceus(feature_config)
        self._d_model = feature_config.d_model
        self._vocab_size = feature_config.vocab_size
        log.info(f"Initialized Caduceus from scratch: d_model={self._d_model}")

    # ======================================================================
    # Forward Pass — Sandwich Architecture
    # ======================================================================

    def _sandwich_forward(self, input_ids):
        """Sandwich forward: pre-layers → bridge → post-layers → head.

        This method manually splits the encoder's forward pass into two stages,
        with a CNN downsampling bridge in between.

        Args:
            input_ids: (B, L) token IDs.

        Returns:
            hidden_states: (B, L_down, D) final encoder hidden states.
        """
        backbone = self.encoder.backbone

        # --- 1. Embedding at full length ---
        hidden_states = backbone.embeddings(input_ids)  # (B, L, D)
        residual = None

        all_layers = list(backbone.layers)

        # --- 2. Pre-downsample layers (full 131k length) ---
        for layer in all_layers[:self.n_pre_layers]:
            hidden_states, residual = layer(hidden_states, residual)

        # --- 3. Merge residual before downsampling ---
        # The fused_add_norm pattern stores (hidden_states, residual) separately.
        # Before downsampling, we merge them into a single representation.
        if residual is not None:
            hidden_states = hidden_states + residual
            residual = None

        # --- 4. CNN Bridge: downsample 131k → 1024 ---
        hidden_states = self.bridge(hidden_states)  # (B, L/factor, D)

        # --- 5. Post-downsample layers (compressed length) ---
        for layer in all_layers[self.n_pre_layers:]:
            hidden_states, residual = layer(hidden_states, residual)

        # --- 6. Final normalization ---
        if not backbone.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = backbone.norm_f(
                residual.to(dtype=backbone.norm_f.weight.dtype)
            )
        else:
            try:
                from mamba_ssm.ops.triton.layernorm import layer_norm_fn
            except ImportError:
                from mamba_ssm.ops.triton.layer_norm import layer_norm_fn
            _ln_out = layer_norm_fn(
                hidden_states,
                backbone.norm_f.weight,
                backbone.norm_f.bias,
                eps=backbone.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=backbone.residual_in_fp32,
            )
            # Some mamba_ssm versions return tuple even with prenorm=False
            hidden_states = _ln_out[0] if isinstance(_ln_out, tuple) else _ln_out

        return hidden_states

    def forward(self, x):
        """
        Forward pass: Sandwich architecture.

        Args:
            x: Input token IDs (B, L), e.g., (B, 131072).

        Returns:
            track_preds: (B, target_len, num_tracks), e.g., (B, 896, 5313).
        """
        # 1. Sandwich forward through encoder with bridge
        hidden_states = self._sandwich_forward(x)    # (B, L_down, D)

        # 2. Simple head: crop + project to tracks
        track_preds = self.head(hidden_states)        # (B, 896, num_tracks)

        return track_preds

    # ======================================================================
    # Freezing
    # ======================================================================

    def _freeze_encoder(self):
        """Freeze encoder parameters (only train bridge + head)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        log.info("Encoder backbone frozen. Training bridge + head only.")

    # ======================================================================
    # Training & Validation
    # ======================================================================

    @staticmethod
    def _compute_per_track_pcc(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation coefficient per track.

        Args:
            preds: (num_tracks, N) predictions.
            targets: (num_tracks, N) targets.
        Returns:
            (num_tracks,) PCC per track.
        """
        preds_mean = preds.mean(dim=1, keepdim=True)
        targets_mean = targets.mean(dim=1, keepdim=True)
        preds_centered = preds - preds_mean
        targets_centered = targets - targets_mean

        cov = (preds_centered * targets_centered).sum(dim=1)
        preds_std = preds_centered.pow(2).sum(dim=1).sqrt()
        targets_std = targets_centered.pow(2).sum(dim=1).sqrt()

        denom = preds_std * targets_std
        pcc = cov / denom.clamp(min=1e-8)
        pcc = torch.where(torch.isfinite(pcc), pcc, torch.zeros_like(pcc))
        return pcc

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (B, L), y: (B, T_len, num_tracks)

        track_preds = self(x)
        loss = self.loss_fn(track_preds, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Per-track PCC monitoring (top 10 tracks)
        with torch.no_grad():
            n_tracks = track_preds.shape[-1]
            preds_per_track = track_preds.detach().reshape(-1, n_tracks).T
            targets_per_track = y.detach().reshape(-1, n_tracks).T
            pcc_per_track = self._compute_per_track_pcc(preds_per_track, targets_per_track)
            # Sort PCC values in descending order and take top 10
            top_k = min(10, len(pcc_per_track))
            top_pcc_values, _ = torch.topk(pcc_per_track, k=top_k, largest=True)
            mean_pcc_top10 = top_pcc_values.mean()
            self.log("train/pcc", mean_pcc_top10, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)            
            # Print top 3 track names periodically (every 100 steps)
            if batch_idx % 100 == 0:
                log_top_tracks(pcc_per_track, self.track_names, top_k=3, step=self.global_step, prefix="Train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        loss = self.loss_fn(y_pred, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        n_tracks = y_pred.shape[-1]
        preds_per_track = y_pred.reshape(-1, n_tracks).T
        targets_per_track = y.reshape(-1, n_tracks).T
        pcc_per_track = self._compute_per_track_pcc(preds_per_track, targets_per_track)
        # Sort PCC values in descending order and take top 10
        top_k = min(10, len(pcc_per_track))
        top_pcc_values, _ = torch.topk(pcc_per_track, k=top_k, largest=True)
        mean_pcc_top10 = top_pcc_values.mean()
        self.log("val/pcc_top10", mean_pcc_top10, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "preds": y_pred, "targets": y, "pcc_per_track": pcc_per_track}
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """Collect PCC scores from each validation batch."""
        if outputs is None:
            return
        
        if 'pcc_per_track' in outputs:
            pcc = outputs['pcc_per_track']
            if not hasattr(self, '_validation_pcc_buffer'):
                self._validation_pcc_buffer = pcc.unsqueeze(0)
            else:
                self._validation_pcc_buffer = torch.cat([self._validation_pcc_buffer, pcc.unsqueeze(0)], dim=0)
    
    def on_validation_epoch_end(self):
        """Generate PCC ranking file at end of validation epoch."""
        if self.trainer.is_global_zero and hasattr(self, '_validation_pcc_buffer'):
            # Average PCC across all validation batches
            pcc_avg = self._validation_pcc_buffer.mean(dim=0)
            
            # Save ranking to CSV
            output_dir = Path(self.trainer.log_dir) if self.trainer.log_dir else Path('./outputs')
            save_pcc_ranking(pcc_avg, self.track_names, output_dir, self.current_epoch, prefix="val")
            
            # Also log top tracks
            log_top_tracks(pcc_avg, self.track_names, top_k=10, prefix="Validation")
            
            # Clear buffer for next epoch
            del self._validation_pcc_buffer

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        x, y = batch if isinstance(batch, (list, tuple)) else (batch, None)
        return self(x)

    # ======================================================================
    # Optimizer & Scheduler
    # ======================================================================

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        from src.utils.scheduler import get_scheduler

        params = filter(lambda p: p.requires_grad, self.parameters())

        opt_cfg = self.hparams.optimizer_config
        opt_name = opt_cfg['name'] if isinstance(opt_cfg, dict) else opt_cfg.name
        opt_args = opt_cfg['args'] if isinstance(opt_cfg, dict) else opt_cfg.args
        opt_args = dict(opt_args)  # Ensure native dict (not DictConfig)

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
