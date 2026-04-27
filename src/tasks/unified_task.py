"""
SF-Fuse fine-tuning task with configurable encoder/decoder split.

Architecture:
    Input tokens (B, L=131072)
    → Caduceus Embedding
    → First n_pre_layers (encoder) at full sequence length
    → CNN Bridge [128x downsampling: 131072 → 1024]
    → Remaining layers (decoder) at compressed length
    → Simple Projection Head [center crop to 896 + MLP]
    → (B, 896, num_tracks)

Supports:
    - Configurable encoder/decoder split via n_pre_layers (0 to n_layer)
    - Optional MLM joint training on encoder output
    - Optional MLM pre-training phase for encoder before main training
    - Loading pretrained checkpoints
    - Independent gated attention decoder (DualRep architecture)

Special cases:
    n_pre_layers=0:  All layers after bridge (equivalent to CNN Stem architecture)
    n_pre_layers=N:  First N layers at full length, rest after bridge (Sandwich)
    n_pre_layers=all: All layers before bridge (full-length processing)

DualRep Architecture (with independent decoder):
    Encoder: 12-layer Wisteria (with optional LoRA)
    CNN Bridge: 128× downsampling
    Decoder: 11-layer gated attention decoder (trained from scratch)
    Head: Output projection
"""

import copy
import importlib
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.utilities import rank_zero_only

from src.tasks.track_utils import (
    load_track_names, log_top_tracks, save_pcc_ranking,
    compute_category_statistics, print_category_statistics,
)
from src.tasks.category_adaptive_mlm import create_category_adaptive_mlm_loss

log = logging.getLogger(__name__)
for _level in ("debug", "info", "warning", "error", "critical"):
    setattr(log, _level, rank_zero_only(getattr(log, _level)))


class SFFuseTask(pl.LightningModule):
    """Unified architecture for DNA genomic track prediction.

    Splits the Caduceus encoder layers into two stages with a CNN bridge:
    1. Encoder stage: first n_pre_layers at full sequence length (131k)
    2. CNN bridge: downsample hidden states (131072 → 1024)
    3. Decoder stage: remaining layers at compressed length (1024)
    4. Simple projection head for track prediction (crop to 896 + MLP)

    Args:
        encoder_config: Configuration for the DNA-LM backbone (Caduceus).
        bridge_config: Configuration for CNNDownsampleBridge.
        head_config: Configuration for SimpleProjectionHead.
        n_pre_layers: Number of encoder layers before the bridge.
        optimizer_config: Optimizer configuration.
        scheduler_config: LR scheduler configuration.
        use_mlm_loss: Enable MLM joint training on encoder output.
        mlm_lambda: Weight for MLM loss in joint training.
        mlm_probability: Masking probability for MLM.
        mask_token_id: Token ID used for [MASK].
        freeze_encoder: Whether to freeze encoder weights.
        pretrained_ckpt_path: Path to pretrained encoder checkpoint.
        decoder_config: Configuration for independent gated attention decoder (DualRep).
            If None, uses standard Caduceus layers as decoder.
    """

    def __init__(
        self,
        encoder_config,
        bridge_config,
        head_config,
        n_pre_layers,
        optimizer_config,
        scheduler_config,
        use_mlm_loss=False,
        mlm_lambda=0.1,
        mlm_probability=0.15,
        mask_token_id=5,
        freeze_encoder=False,
        pretrained_ckpt_path=None,
        track_names_file=None,
        mlm_strategy='default',
        use_lora=False,
        lora_config=None,
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()

        # --- Load track names if provided ---
        self.track_names = load_track_names(track_names_file, head_config.get('num_tracks', 5313))

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
        self.use_mlm_loss = use_mlm_loss
        self.mlm_lambda = mlm_lambda
        self.mlm_probability = mlm_probability
        self.mask_token_id = mask_token_id
        self.mlm_strategy = mlm_strategy

        # Phase: 'pretrain' or 'finetune'
        self._phase = 'finetune'

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

        # Store LoRA configuration
        self.use_lora = use_lora
        self.lora_config = lora_config if lora_config is not None else {}

        # Store decoder configuration (for DualRep independent decoder)
        self.decoder_config = decoder_config

        self.save_hyperparameters({
            "encoder_config": hparams_encoder_config,
            "bridge_config": bridge_config,
            "head_config": head_config,
            "n_pre_layers": n_pre_layers,
            "optimizer_config": optimizer_config,
            "scheduler_config": scheduler_config,
            "use_mlm_loss": use_mlm_loss,
            "mlm_lambda": mlm_lambda,
            "mlm_probability": mlm_probability,
            "mask_token_id": mask_token_id,
            "mlm_strategy": mlm_strategy,
            "freeze_encoder": freeze_encoder,
            "pretrained_ckpt_path": pretrained_ckpt_path,
            "use_lora": use_lora,
            "lora_config": lora_config,
            "decoder_config": decoder_config,
            "vocab_size": self._vocab_size,
            "d_model": self._d_model,
            **kwargs,
        })

        # --- 1. Initialize Encoder (Caduceus) ---
        self._init_encoder(runtime_encoder_config, pretrained_ckpt_path)

        # Validate n_pre_layers
        total_layers = len(list(self._get_backbone().layers))
        if self.n_pre_layers < 0 or self.n_pre_layers > total_layers:
            raise ValueError(
                f"n_pre_layers ({self.n_pre_layers}) must be in [0, {total_layers}]. "
                f"Encoder has {total_layers} layers total."
            )
        n_post_layers = total_layers - self.n_pre_layers
        log.info(
            f"Unified architecture: {self.n_pre_layers} encoder layers "
            f"(full length) + bridge + {n_post_layers} decoder layers "
            f"(compressed length)"
        )

        # --- 1b. Freeze encoder's internal embedding layer if n_pre_layers == 0 ---
        # When n_pre_layers=0, all processing happens after bridge. The encoder's
        # built-in embedding is still used (for the initial embedding step).
        # But we should NOT freeze it here since we always go through it.

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

        # --- 4. Initialize Decoder (DualRep: independent gated attention decoder) ---
        if decoder_config is not None:
            self._init_decoder(decoder_config)
        else:
            self.decoder = None
            log.info("No independent decoder: using standard Caduceus layers as decoder")

        # Update hparams with final configs
        self.hparams.bridge_config = bridge_config
        self.hparams.head_config = head_config
        self.hparams.track_names_file = track_names_file

        # --- 5. Loss Functions ---
        self.task_loss_fn = nn.PoissonNLLLoss(log_input=False, full=True)

        # --- 6. Category-Adaptive MLM Loss ---
        if use_mlm_loss:
            self.adaptive_mlm_loss = create_category_adaptive_mlm_loss(
                track_names=self.track_names,
                base_mlm_lambda=mlm_lambda,
                strategy=mlm_strategy,
            )
        else:
            self.adaptive_mlm_loss = None

        # --- 7. MLM Components ---
        # Created if joint training is enabled OR if pre-training may be used
        # (pre-training flag is controlled externally via set_pretrain_phase)
        self._need_mlm = use_mlm_loss or kwargs.get('pretrain_encoder', False)
        # Always create MLM head — supports dynamic pre-training phase switches
        self.mlm_projection = nn.Linear(self._d_model, self._vocab_size)
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.encoder_norm = nn.LayerNorm(self._d_model)
        log.info(
            f"MLM components initialized: vocab_size={self._vocab_size}, "
            f"d_model={self._d_model}"
        )

        # --- 8. Set training phase ---
        if freeze_encoder:
            self._freeze_encoder()
        else:
            self.set_finetune_phase()  # Unfreezes all params + applies LoRA if configured

    # ======================================================================
    # Phase Management (for pre-training support)
    # ======================================================================

    def set_pretrain_phase(self):
        """Switch to pre-training mode.

        Freezes bridge, decoder layers, and head.
        Only trains encoder layers + embedding + MLM projection.

        For DualRep architecture:
        - Independent decoder is also frozen (not used in pre-training)
        - Only encoder (Wisteria) is trained with MLM loss
        """
        self._phase = 'pretrain'
        self._phase_step_logged = False  # for first-step verification

        # Freeze bridge
        for param in self.bridge.parameters():
            param.requires_grad = False

        # Freeze head
        for param in self.head.parameters():
            param.requires_grad = False

        # Freeze standard decoder layers (post-bridge Caduceus layers)
        backbone = self._get_backbone()
        all_layers = list(backbone.layers)
        for layer in all_layers[self.n_pre_layers:]:
            for param in layer.parameters():
                param.requires_grad = False

        # Freeze independent decoder if it exists (DualRep architecture)
        if hasattr(self, 'decoder') and self.decoder is not None:
            for param in self.decoder.parameters():
                param.requires_grad = False
            log.info("Independent decoder frozen (DualRep decoder not used in pre-training)")

        # Ensure encoder layers (pre-bridge) are trainable
        for layer in all_layers[:self.n_pre_layers]:
            for param in layer.parameters():
                param.requires_grad = True

        # Ensure embedding is trainable
        for param in backbone.embeddings.parameters():
            param.requires_grad = True

        # Ensure MLM projection and norm are trainable
        for param in self.mlm_projection.parameters():
            param.requires_grad = True
        for param in self.encoder_norm.parameters():
            param.requires_grad = True

        # Count params per component
        def _count(module):
            t = sum(p.numel() for p in module.parameters() if p.requires_grad)
            f = sum(p.numel() for p in module.parameters() if not p.requires_grad)
            return t, f

        enc_t, enc_f = _count(self.encoder)
        bri_t, bri_f = _count(self.bridge)
        hd_t, hd_f = _count(self.head)
        mlm_t, _ = _count(self.mlm_projection)
        norm_t, _ = _count(self.encoder_norm)
        dec_t, dec_f = (0, 0)
        if hasattr(self, 'decoder') and self.decoder is not None:
            dec_t, dec_f = _count(self.decoder)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)

        log.info("=" * 70)
        log.info("[PRE-TRAIN PHASE] Parameter status:")
        log.info(f"  Encoder  : trainable={enc_t:>12,}  frozen={enc_f:>12,}")
        log.info(f"  Bridge   : trainable={bri_t:>12,}  frozen={bri_f:>12,}")
        log.info(f"  Head     : trainable={hd_t:>12,}  frozen={hd_f:>12,}")
        if dec_t > 0 or dec_f > 0:
            log.info(f"  Decoder  : trainable={dec_t:>12,}  frozen={dec_f:>12,}")
        log.info(f"  MLM proj : trainable={mlm_t:>12,}")
        log.info(f"  Enc norm : trainable={norm_t:>12,}")
        log.info(f"  ─────────────────────────────────")
        log.info(f"  TOTAL    : trainable={trainable:>12,}  frozen={frozen:>12,}")
        log.info("=" * 70)

    def set_finetune_phase(self):
        """Switch to fine-tuning mode.

        Unfreezes all parameters (bridge, decoder, head).
        Optionally applies LoRA to encoder if use_lora is True.

        For DualRep architecture:
        - Independent decoder is trained from scratch
        - LoRA is applied only to encoder (not decoder)
        - Bridge and head are also trainable
        """
        self._phase = 'finetune'

        # Unfreeze everything
        for param in self.parameters():
            param.requires_grad = True

        # Apply LoRA to encoder if requested (after unfreezing)
        # Note: LoRA is ONLY applied to encoder, never to decoder
        if self.use_lora:
            self._apply_lora_to_encoder()

        # Count parameters with breakdown
        def _count(module):
            t = sum(p.numel() for p in module.parameters() if p.requires_grad)
            f = sum(p.numel() for p in module.parameters() if not p.requires_grad)
            return t, f

        enc_t, enc_f = _count(self.encoder)
        bri_t, bri_f = _count(self.bridge)
        hd_t, hd_f = _count(self.head)
        dec_t, dec_f = (0, 0)
        if hasattr(self, 'decoder') and self.decoder is not None:
            dec_t, dec_f = _count(self.decoder)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())

        log.info("=" * 70)
        log.info("[FINETUNE PHASE] Parameter status:")
        log.info(f"  Encoder  : trainable={enc_t:>12,}  frozen={enc_f:>12,}")
        log.info(f"  Bridge   : trainable={bri_t:>12,}  frozen={bri_f:>12,}")
        log.info(f"  Head     : trainable={hd_t:>12,}  frozen={hd_f:>12,}")
        if dec_t > 0 or dec_f > 0:
            log.info(f"  Decoder  : trainable={dec_t:>12,}  frozen={dec_f:>12,}")
        log.info(f"  ─────────────────────────────────")
        log.info(f"  TOTAL    : trainable={trainable:>12,} / {total_params:>12,} "
                 f"({100*trainable/total_params:.2f}%)")
        log.info("=" * 70)

    def _apply_lora_to_encoder(self):
        """Apply LoRA (Low-Rank Adaptation) to encoder for parameter-efficient fine-tuning.

        LoRA injects trainable low-rank matrices into encoder layers while freezing
        the original weights. This significantly reduces trainable parameters while
        maintaining performance.
        """
        # Guard: skip if LoRA already applied
        if getattr(self, '_lora_applied', False):
            log.info("[LoRA] Already applied, skipping.")
            return

        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            log.error(
                "LoRA requested but 'peft' library not installed. "
                "Install with: pip install peft>=0.7.0"
            )
            raise

        # Get LoRA configuration with sensible defaults
        rank = self.lora_config.get('rank', 8)
        alpha = self.lora_config.get('alpha', 16)
        dropout = self.lora_config.get('dropout', 0.05)
        target_modules = self.lora_config.get(
            'target_modules',
            ["query", "value"]  # Common names, but may need adjustment
        )

        # Create LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # For encoder
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            inference_mode=False,
        )

        log.info("=" * 70)
        log.info("[LoRA] Applying Low-Rank Adaptation to encoder...")
        log.info(f"  Rank (r): {rank}")
        log.info(f"  Alpha: {alpha}")
        log.info(f"  Dropout: {dropout}")
        log.info(f"  Target modules: {target_modules}")

        # Apply LoRA to encoder
        try:
            self.encoder = get_peft_model(self.encoder, peft_config)
            self.encoder.print_trainable_parameters()
            self._lora_applied = True
        except Exception as e:
            log.warning(f"  Failed to apply LoRA with target_modules={target_modules}")
            log.warning(f"  Error: {e}")
            log.info("  Attempting with alternative module names...")
            
            # Try alternative common names for attention layers
            for alt_targets in [
                ["q_proj", "v_proj"],
                ["mixer.in_proj", "mixer.out_proj"],
                ["attn.Wqkv", "attn.out_proj"],
            ]:
                try:
                    peft_config.target_modules = alt_targets
                    log.info(f"  Trying target_modules: {alt_targets}")
                    self.encoder = get_peft_model(self.encoder, peft_config)
                    self.encoder.print_trainable_parameters()
                    self._lora_applied = True
                    log.info(f"  [OK] LoRA applied successfully with {alt_targets}")
                    break
                except Exception as e2:
                    log.warning(f"  Failed with {alt_targets}: {e2}")
                    continue
            else:
                log.error("  [FAIL] Could not apply LoRA with any target module names.")
                log.error("  Falling back to full fine-tuning (no LoRA).")
                self.use_lora = False
                return

        self._lora_applied = True

        # Count trainable vs frozen parameters
        def _count_params(module):
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total = sum(p.numel() for p in module.parameters())
            return trainable, total

        enc_train, enc_total = _count_params(self.encoder)
        bridge_train, bridge_total = _count_params(self.bridge)
        head_train, head_total = _count_params(self.head)
        dec_train, dec_total = (0, 0)
        if hasattr(self, 'decoder') and self.decoder is not None:
            dec_train, dec_total = _count_params(self.decoder)

        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())

        log.info("  ─────────────────────────────────")
        log.info(f"  Encoder  : {enc_train:>12,} / {enc_total:>12,} trainable")
        log.info(f"  Bridge   : {bridge_train:>12,} / {bridge_total:>12,} trainable")
        log.info(f"  Head     : {head_train:>12,} / {head_total:>12,} trainable")
        if dec_train > 0 or dec_total > 0:
            log.info(f"  Decoder  : {dec_train:>12,} / {dec_total:>12,} trainable")
        log.info(f"  ─────────────────────────────────")
        log.info(f"  TOTAL    : {total_trainable:>12,} / {total_params:>12,} "
                 f"({100*total_trainable/total_params:.2f}%)")
        log.info("=" * 70)
        log.info("  [NOTE] LoRA is applied ONLY to encoder, not decoder")

    # ======================================================================
    # Encoder Initialization
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
        """Load pretrained encoder from HF format or PL checkpoint."""
        import os

        # Check if it's a PL checkpoint (.ckpt file)
        if os.path.isfile(pretrained_ckpt_path) and pretrained_ckpt_path.endswith('.ckpt'):
            log.info(f"Detected PL checkpoint: {pretrained_ckpt_path}")
            self._load_pl_checkpoint(pretrained_ckpt_path)
            return

        model_type = self._detect_model_type(pretrained_ckpt_path)
        log.info(f"Detected model type: {model_type}")

        if model_type == "caduceus":
            self._load_caduceus_pretrained(pretrained_ckpt_path)
        elif model_type == "hyenadna":
            self._load_hyenadna_pretrained(pretrained_ckpt_path)
        else:
            self._load_hf_automodel(pretrained_ckpt_path)

    def _detect_model_type(self, path):
        """Detect model type from config.json."""
        import os, json
        config_path = os.path.join(path, "config.json") if os.path.isdir(path) else None
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            model_type = config_dict.get("model_type", "")
            if model_type == "hyenadna":
                return "hyenadna"
            if model_type == "caduceus":
                return "caduceus"
            # Legacy HyenaDNA: no model_type but has hyena-specific keys
            if "layer" in config_dict and isinstance(config_dict.get("layer"), dict):
                layer_cfg = config_dict["layer"]
                if layer_cfg.get("_name_") == "hyena" or "filter_order" in layer_cfg:
                    return "hyenadna"
            return model_type or "unknown"
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(path, trust_remote_code=True)
            model_type = getattr(config, "model_type", "unknown")
            if model_type == "hyenadna":
                return "hyenadna"
            return model_type
        except Exception:
            return "unknown"

    def _load_pl_checkpoint(self, pretrained_ckpt_path):
        """Load pretrained Caduceus encoder from PyTorch Lightning checkpoint."""
        try:
            from src.caduceus.configuration_caduceus import CaduceusConfig
            from src.caduceus.modeling_caduceus import Caduceus
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import Caduceus modules: {e}. "
                f"Please install mamba_ssm: pip install mamba-ssm causal-conv1d"
            )

        log.info(f"Loading encoder weights from PL checkpoint: {pretrained_ckpt_path}")
        import torch

        # Load PL checkpoint
        ckpt = torch.load(pretrained_ckpt_path, weights_only=False)
        state_dict = ckpt['state_dict']

        # Extract encoder state dict (remove 'model.' prefix)
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.caduceus.'):
                new_key = k.replace('model.caduceus.', '')
                encoder_state_dict[new_key] = v

        log.info(f"Extracted {len(encoder_state_dict)} encoder keys from PL checkpoint")

        # Get config from checkpoint hparams
        hparams = ckpt.get('hyper_parameters', {})
        enc_config = hparams.get('encoder_config', {})
        pretrained_config = None
        if isinstance(enc_config, dict) and 'config' in enc_config:
            config_dict = enc_config['config']
            if isinstance(config_dict, dict):
                # Extract only the fields needed for CaduceusConfig
                config_kwargs = {}
                for key in ['d_model', 'n_layer', 'vocab_size', 'd_intermediate', 'n_modules',
                           'layers_per_module', 'conv_layers_per_module', 'attn_layer_in_module',
                           'dilation_base', 'dropout', 'ssm_cfg', 'attn_cfg', 'bidirectional',
                           'bidirectional_strategy', 'bidirectional_weight_tie', 'rcps',
                           'use_fourier_pos_emb', 'fourier_max_seq_len', 'fourier_dim',
                           'fourier_init', 'headwise_attn_output_gate', 'elementwise_attn_output_gate',
                           'initializer_cfg', 'rescale_prenorm_residual', 'n_residuals_per_layer']:
                    if key in config_dict:
                        config_kwargs[key] = config_dict[key]
                pretrained_config = CaduceusConfig(**config_kwargs)
                self._d_model = pretrained_config.d_model
                self._vocab_size = pretrained_config.vocab_size
                log.info(f"Updated from PL checkpoint: d_model={self._d_model}, vocab_size={self._vocab_size}")

        if pretrained_config is None:
            # Fallback to hparams config
            pretrained_config = self._get_caduceus_config_from_hparams()

        # Create encoder and load weights
        self.encoder = Caduceus(config=pretrained_config)
        missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_state_dict, strict=False)
        if missing_keys:
            log.warning(f"Missing keys when loading encoder from PL checkpoint: {missing_keys}")
        if unexpected_keys:
            log.warning(f"Unexpected keys when loading encoder from PL checkpoint: {unexpected_keys}")

        log.info("Loaded encoder weights from PL checkpoint successfully.")

    def _get_caduceus_config_from_hparams(self):
        """Get CaduceusConfig from hparams encoder_config."""
        enc_config = self.hparams.get('encoder_config', {})
        if isinstance(enc_config, dict) and 'config' in enc_config:
            config_dict = enc_config['config']
            if isinstance(config_dict, dict):
                config_kwargs = {}
                for key in ['d_model', 'n_layer', 'vocab_size', 'd_intermediate', 'n_modules',
                           'layers_per_module', 'conv_layers_per_module', 'attn_layer_in_module',
                           'dilation_base', 'dropout', 'ssm_cfg', 'attn_cfg', 'bidirectional',
                           'bidirectional_strategy', 'bidirectional_weight_tie', 'rcps',
                           'use_fourier_pos_emb', 'fourier_max_seq_len', 'fourier_dim',
                           'fourier_init', 'headwise_attn_output_gate', 'elementwise_attn_output_gate',
                           'initializer_cfg', 'rescale_prenorm_residual', 'n_residuals_per_layer']:
                    if key in config_dict:
                        config_kwargs[key] = config_dict[key]
                from src.caduceus.configuration_caduceus import CaduceusConfig
                return CaduceusConfig(**config_kwargs)
        raise ValueError("Could not extract CaduceusConfig from hparams")

    def _get_encoder_config_for_pl_load(self):
        """Get encoder config for PL checkpoint loading (with fallbacks)."""
        enc_config = self.hparams.get('encoder_config', {})
        if isinstance(enc_config, dict) and 'config' in enc_config:
            config_dict = enc_config['config']
            if isinstance(config_dict, dict):
                return config_dict
        # Fallback to default config
        from src.caduceus.configuration_caduceus import CaduceusConfig
        return {
            'd_model': self._d_model if hasattr(self, '_d_model') else 256,
            'vocab_size': self._vocab_size if hasattr(self, '_vocab_size') else 12,
            'n_layer': 8,
        }

    def _load_caduceus_pretrained(self, pretrained_ckpt_path):
        """Load pretrained Caduceus model."""
        try:
            from src.caduceus.configuration_caduceus import CaduceusConfig
            from src.caduceus.modeling_caduceus import CaduceusForMaskedLM, Caduceus
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import Caduceus modules: {e}. "
                f"Please install mamba_ssm: pip install mamba-ssm causal-conv1d"
            )

        pretrained_config = CaduceusConfig.from_pretrained(pretrained_ckpt_path)

        # Update internal parameters from pretrained config
        self._d_model = pretrained_config.d_model
        self._vocab_size = pretrained_config.vocab_size
        log.info(f"Updated from pretrained: d_model={self._d_model}, vocab_size={self._vocab_size}")

        # Keep fourier_max_seq_len at full length for encoder attention layers
        target_max_seq_len = self._get_target_max_seq_len()
        original_fourier_len = getattr(pretrained_config, 'fourier_max_seq_len', 16384)
        if target_max_seq_len > original_fourier_len:
            log.info(
                f"Updating fourier_max_seq_len: {original_fourier_len} -> {target_max_seq_len}"
            )
            pretrained_config.fourier_max_seq_len = target_max_seq_len

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

    def _load_hyenadna_pretrained(self, pretrained_ckpt_path):
        """Load pretrained HyenaDNA model from legacy format (config.json + weights.ckpt).

        Handles both legacy format (from original hyena-dna repo) and HF -hf format.
        Legacy: uses HyenaDNAPreTrainedModel.from_pretrained() which loads weights.ckpt
        HF -hf: uses standard AutoModel.from_pretrained() with trust_remote_code=True
        """
        import os
        import json

        # Check if this is legacy format (has weights.ckpt)
        is_legacy = os.path.isfile(os.path.join(pretrained_ckpt_path, "weights.ckpt"))

        if is_legacy:
            log.info(f"Loading legacy HyenaDNA from: {pretrained_ckpt_path}")
            # Load config
            with open(os.path.join(pretrained_ckpt_path, "config.json"), "r") as f:
                config_dict = json.load(f)

            # Add hyena-dna to path for loading the model
            hyena_dna_path = os.path.join(os.path.dirname(__file__), "..", "..", "hyena-dna")
            if hyena_dna_path not in sys.path:
                sys.path.insert(0, hyena_dna_path)

            try:
                import importlib
                # Import standalone_hyenadna and huggingface modules
                spec = importlib.util.spec_from_file_location(
                    "standalone_hyenadna",
                    os.path.join(hyena_dna_path, "standalone_hyenadna.py")
                )
                standalone_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(standalone_mod)

                spec2 = importlib.util.spec_from_file_location(
                    "huggingface_wrapper",
                    os.path.join(hyena_dna_path, "huggingface.py")
                )
                hf_mod = importlib.util.module_from_spec(spec2)
                spec2.loader.exec_module(hf_mod)

                self.encoder = hf_mod.HyenaDNAPreTrainedModel.from_pretrained(
                    os.path.dirname(pretrained_ckpt_path) or ".",
                    os.path.basename(pretrained_ckpt_path),
                    download=False,
                    config=config_dict,
                    use_head=False,
                    n_classes=2,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load legacy HyenaDNA from {pretrained_ckpt_path}: {e}"
                )

            self._d_model = config_dict.get("d_model", 256)
            self._vocab_size = config_dict.get("vocab_size", 12)
        else:
            # HF -hf format: use AutoModel
            log.info(f"Loading HF-format HyenaDNA from: {pretrained_ckpt_path}")
            from transformers import AutoModel

            try:
                self.encoder = AutoModel.from_pretrained(
                    pretrained_ckpt_path,
                    trust_remote_code=True,
                    output_loading_info=False,
                )
                # Extract backbone (HyenaDNAModel has .backbone = HyenaLMBackbone)
                if hasattr(self.encoder, 'backbone'):
                    self.encoder = self.encoder.backbone
                    log.info("Extracted HyenaLMBackbone from HyenaDNAModel.")
                elif hasattr(self.encoder, 'model'):
                    self.encoder = self.encoder.model
                elif hasattr(self.encoder, 'caduceus'):
                    self.encoder = self.encoder.caduceus
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load HF HyenaDNA from {pretrained_ckpt_path}: {e}"
                )

            # Load config for d_model and vocab_size
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(pretrained_ckpt_path, trust_remote_code=True)
            self._d_model = getattr(config, "d_model", 256)
            self._vocab_size = getattr(config, "vocab_size", 12)

        log.info(f"Loaded HyenaDNA: d_model={self._d_model}, vocab_size={self._vocab_size}")
        self.hparams.d_model = self._d_model
        self.hparams.vocab_size = self._vocab_size

    def _get_target_max_seq_len(self):
        """Get the target maximum sequence length for fine-tuning."""
        enc_config = self.hparams.get('encoder_config', {})
        if isinstance(enc_config, dict):
            inner_config = enc_config.get('config', {})
            if isinstance(inner_config, dict):
                fourier_len = inner_config.get('fourier_max_seq_len')
                if fourier_len is not None:
                    return fourier_len
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

            try:
                from src.caduceus.modeling_caduceus import Caduceus as _CaduceusModel
            except ImportError:
                _CaduceusModel = type(None)

            if hasattr(self.encoder, 'caduceus'):
                self.encoder = self.encoder.caduceus
            elif isinstance(self.encoder, _CaduceusModel):
                pass
            elif hasattr(self.encoder, 'backbone'):
                self.encoder = self.encoder.backbone
                log.info("Extracted backbone from HF model (HyenaDNA).")
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

        if 'caduceus' in str(_name).lower():
            self._init_caduceus_from_scratch(config_obj)
        elif 'hyena' in str(_name).lower():
            raise ValueError(
                f"HyenaDNA from-scratch initialization is not supported. "
                f"Use pretrained_ckpt_path to load a pretrained HyenaDNA model."
            )
        else:
            raise ValueError(
                f"Unknown encoder type: {_name}. "
                f"Unified architecture requires Caduceus or HyenaDNA encoder."
            )

    def _init_caduceus_from_scratch(self, config_obj):
        """Initialize Caduceus model from config object or dict."""
        try:
            from src.caduceus.configuration_caduceus import CaduceusConfig
            from src.caduceus.modeling_caduceus import Caduceus
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import Caduceus modules: {e}. "
                f"Please install mamba_ssm: pip install mamba-ssm causal-conv1d"
            )

        feature_config = None

        if isinstance(config_obj, CaduceusConfig):
            feature_config = config_obj
        elif config_obj is not None:
            if isinstance(config_obj, DictConfig):
                config_dict = OmegaConf.to_container(config_obj, resolve=True)
            elif isinstance(config_obj, dict):
                config_dict = copy.deepcopy(config_obj)
            else:
                config_dict = None

            if isinstance(config_dict, dict):
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
        # Sync hyperparameters (vocab_size may be padded by CaduceusConfig)
        self.hparams.d_model = self._d_model
        self.hparams.vocab_size = self._vocab_size
        log.info(f"Initialized Caduceus from scratch: d_model={self._d_model}, vocab_size={self._vocab_size}")

    # ======================================================================
    # Decoder Initialization (DualRep: Independent Gated Attention Decoder)
    # ======================================================================

    def _init_decoder(self, decoder_config):
        """Initialize independent gated attention decoder for DualRep architecture.

        The decoder is a separate module with its own gated attention layers,
        trained from scratch on top of the frozen/pretrained encoder + bridge.

        Args:
            decoder_config: Configuration dictionary for the decoder.
                Expected keys:
                - _target_: Module target (default: GatedAttentionDecoder)
                - n_layers: Number of decoder layers (default: 11)
                - d_model: Model dimension (default: same as encoder)
                - num_heads: Number of attention heads
                - headwise_gate: Enable headwise gating (default: True)
                - elementwise_gate: Enable elementwise gating (default: False)
                - dropout: Dropout rate
                - mlp_dim: Feedforward dimension
                - causal_mask: Use causal masking (default: False)
        """
        from src.models.gated_attention_decoder import GatedAttentionDecoder

        log.info("Initializing independent gated attention decoder (DualRep)...")

        # Parse decoder configuration
        if OmegaConf.is_config(decoder_config):
            decoder_config = OmegaConf.to_container(decoder_config, resolve=True)

        n_layers = decoder_config.get('n_layers', 11)
        d_model = decoder_config.get('d_model', self._d_model)
        num_heads = decoder_config.get('num_heads', 12)
        headwise_gate = decoder_config.get('headwise_gate', True)
        elementwise_gate = decoder_config.get('elementwise_gate', False)
        dropout = decoder_config.get('dropout', 0.1)
        mlp_dim = decoder_config.get('mlp_dim', d_model * 2)
        causal_mask = decoder_config.get('causal_mask', False)

        # Create decoder using GatedAttentionDecoder module
        self.decoder = GatedAttentionDecoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=n_layers,
            dim_ff=mlp_dim,
            dropout=dropout,
        )

        log.info(
            f"Decoder initialized: {n_layers} layers, d_model={d_model}, "
            f"num_heads={num_heads}, headwise_gate={headwise_gate}"
        )

        # Count decoder parameters
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        log.info(f"Decoder trainable parameters: {decoder_params:,}")

    # ======================================================================
    # Forward Pass
    # ======================================================================

    def _run_backbone_layers(self, layers, hidden_states, residual):
        """Run backbone layers, auto-detecting Caduceus vs HyenaDNA signature.

        Caduceus layers: (h, residual) -> (h, residual) via fused_add_norm
        HyenaDNA layers: (h) -> (h), residual handled internally
        """
        for layer in layers:
            if hasattr(layer, 'fused_add_norm') and layer.fused_add_norm:
                # Caduceus: prenorm with external residual
                hidden_states, residual = layer(hidden_states, residual)
            else:
                # HyenaDNA: single-arg forward, residual handled internally
                hidden_states = layer(hidden_states)
                residual = None
        return hidden_states, residual

    def _final_norm(self, backbone, hidden_states, residual):
        """Apply final LayerNorm, handling both Caduceus and HyenaDNA backbones.

        Caduceus: norm_f (RMSNorm), supports fused_add_norm with residual
        HyenaDNA: ln_f (LayerNorm), no fused_add_norm
        """
        if hasattr(backbone, 'fused_add_norm') and backbone.fused_add_norm:
            # Caduceus path: fused LayerNorm with residual
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
            return _ln_out[0] if isinstance(_ln_out, tuple) else _ln_out
        else:
            # HyenaDNA / generic path: standard LayerNorm
            residual = (hidden_states + residual) if residual is not None else hidden_states
            norm = getattr(backbone, 'norm_f', None) or getattr(backbone, 'ln_f', None)
            if norm is None:
                return hidden_states
            weight_dtype = norm.weight.dtype if hasattr(norm, 'weight') else torch.float32
            return norm(residual.to(dtype=weight_dtype))

    def _get_backbone(self):
        """Get the backbone module, handling both wrapped and direct encoder formats."""
        if hasattr(self.encoder, 'backbone'):
            return self.encoder.backbone
        return self.encoder

    def _encoder_forward(self, input_ids):
        """Forward pass through encoder only (pre-bridge layers).

        Used for MLM: embedding → pre_layers → LayerNorm → hidden states.

        Args:
            input_ids: (B, L) token IDs.

        Returns:
            hidden_states: (B, L, D) encoder hidden states at full length.
        """
        backbone = self._get_backbone()
        hidden_states = backbone.embeddings(input_ids)  # (B, L, D)
        residual = None

        all_layers = list(backbone.layers)

        # Pre-downsample layers (encoder)
        hidden_states, residual = self._run_backbone_layers(
            all_layers[:self.n_pre_layers], hidden_states, residual
        )

        # Merge residual (HyenaDNA handles this internally, residual will be None)
        if residual is not None:
            hidden_states = hidden_states + residual

        # Normalize for stable MLM projection
        hidden_states = self.encoder_norm(hidden_states)

        return hidden_states

    def _full_forward(self, input_ids):
        """Full forward: embedding → pre_layers → bridge → post_layers → head.

        Args:
            input_ids: (B, L) token IDs.

        Returns:
            track_preds: (B, target_len, num_tracks).
        """
        hidden_states = self._backbone_forward(input_ids)
        track_preds = self.head(hidden_states)
        return track_preds

    def _backbone_forward(self, input_ids):
        """Backbone forward: embedding → pre_layers → bridge → post_layers/decoder.

        Args:
            input_ids: (B, L) token IDs.

        Returns:
            hidden_states: (B, L_down, D) final hidden states.
        """
        backbone = self._get_backbone()
        hidden_states = backbone.embeddings(input_ids)  # (B, L, D)
        residual = None

        all_layers = list(backbone.layers)

        # --- 1. Pre-downsample layers (encoder, full length) ---
        hidden_states, residual = self._run_backbone_layers(
            all_layers[:self.n_pre_layers], hidden_states, residual
        )

        # --- 2. Merge residual before downsampling ---
        if residual is not None:
            hidden_states = hidden_states + residual
            residual = None

        # --- 3. CNN Bridge: downsample (e.g., 131072 → 1024) ---
        hidden_states = self.bridge(hidden_states)

        # --- 4. Post-downsample processing ---
        if hasattr(self, 'decoder') and self.decoder is not None:
            # DualRep: Use independent gated attention decoder
            hidden_states = self._decoder_forward(hidden_states)
        else:
            # Standard: Use remaining encoder layers as decoder
            hidden_states, residual = self._run_backbone_layers(
                all_layers[self.n_pre_layers:], hidden_states, residual
            )

            # --- 5. Final normalization ---
            hidden_states = self._final_norm(backbone, hidden_states, residual)

        return hidden_states

    def _decoder_forward(self, hidden_states):
        """Forward pass through independent decoder (DualRep architecture).

        Args:
            hidden_states: (B, L_down, D) bridge output hidden states.

        Returns:
            hidden_states: (B, L_down, D) decoder output hidden states.
        """
        output, _ = self.decoder(hidden_states)
        return output

    def forward(self, x):
        """Forward pass through the full model.

        Args:
            x: Input token IDs (B, L).

        Returns:
            track_preds: (B, target_len, num_tracks).
        """
        return self._full_forward(x)

    # ======================================================================
    # MLM Masking
    # ======================================================================

    def _mask_input(self, input_ids):
        """Apply BERT-style MLM masking.

        Args:
            input_ids: (B, L) original token IDs.

        Returns:
            masked_input_ids: (B, L) with masking applied.
            labels: (B, L) targets, -100 for non-masked positions.
        """
        input_ids = input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(
            labels.shape,
            self.mlm_probability,
            device=labels.device,
            dtype=torch.float,
        )

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% → [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool()
            & masked_indices
        )
        input_ids[indices_replaced] = self.mask_token_id

        # 10% → random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            self._vocab_size, labels.shape, device=labels.device, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        # 10% → keep original (no change)
        return input_ids, labels

    # ======================================================================
    # Freezing
    # ======================================================================

    def _freeze_encoder(self):
        """Freeze encoder pre-layers (only train bridge + decoder + head).

        For DualRep architecture with independent decoder:
        - Encoder (Wisteria) is frozen
        - Independent decoder is still trainable (trained from scratch)
        - Bridge and head are also trainable
        """
        backbone = self._get_backbone()
        all_layers = list(backbone.layers)
        for layer in all_layers[:self.n_pre_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        for param in backbone.embeddings.parameters():
            param.requires_grad = False

        # Note: Independent decoder (if exists) is NOT frozen here
        # It is trained from scratch in fine-tuning phase

        log.info(
            f"Encoder frozen: embedding + first {self.n_pre_layers} layers. "
            f"Training bridge + decoder + head only."
        )

    # ======================================================================
    # Training & Validation
    # ======================================================================

    @staticmethod
    def _compute_per_track_pcc(preds, targets):
        """Compute Pearson correlation coefficient per track."""
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
        # Log phase verification on first step
        if not getattr(self, '_phase_step_logged', True):
            self._phase_step_logged = True
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            log.info(
                f"[PHASE VERIFY] training_step phase='{self._phase}', "
                f"trainable={trainable:,}/{total:,}"
            )
        if self._phase == 'pretrain':
            return self._pretrain_training_step(batch, batch_idx)
        else:
            return self._finetune_training_step(batch, batch_idx)

    def _pretrain_training_step(self, batch, batch_idx):
        """Pre-training step: MLM on encoder only."""
        x, _y = batch  # Ignore targets during pre-training

        # Apply MLM masking
        x_masked, mlm_labels = self._mask_input(x)

        # Encoder forward only
        encoder_hidden = self._encoder_forward(x_masked)  # (B, L, D)

        # MLM prediction
        mlm_logits = self.mlm_projection(encoder_hidden)  # (B, L, vocab_size)
        loss = self.mlm_loss_fn(
            mlm_logits.view(-1, mlm_logits.size(-1)),
            mlm_labels.view(-1),
        )

        # Accuracy on masked positions
        with torch.no_grad():
            preds = mlm_logits.argmax(dim=-1)
            mask = mlm_labels != -100
            if mask.any():
                accuracy = (preds[mask] == mlm_labels[mask]).float().mean()
            else:
                accuracy = torch.tensor(0.0, device=self.device)

        self.log("pretrain/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("pretrain/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("pretrain/perplexity", torch.exp(loss).clamp(max=1000), on_step=True, sync_dist=False)

        return loss

    def _finetune_training_step(self, batch, batch_idx):
        """Fine-tuning step: task loss + optional category-adaptive MLM joint loss."""
        x, y = batch  # x: (B, L), y: (B, T_len, num_tracks)

        if self.use_mlm_loss:
            # --- Joint training: SF-Fuse + MLM ---

            # Task branch: clean input → full forward
            track_preds = self.forward(x)
            task_loss = self.task_loss_fn(track_preds, y)

            # MLM branch: masked input → encoder only
            x_masked, mlm_labels = self._mask_input(x)
            encoder_hidden = self._encoder_forward(x_masked)
            mlm_logits = self.mlm_projection(encoder_hidden)
            mlm_loss = self.mlm_loss_fn(
                mlm_logits.view(-1, self._vocab_size),
                mlm_labels.view(-1),
            )

            # Category-adaptive combined loss
            total_loss = self.adaptive_mlm_loss(task_loss, mlm_loss)

            self.log("train/task_loss", task_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("train/mlm_loss", mlm_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("train/mlm_lambda", self.adaptive_mlm_loss.mlm_lambda, on_step=False, on_epoch=True, sync_dist=False)
        else:
            # Task only
            track_preds = self.forward(x)
            task_loss = self.task_loss_fn(track_preds, y)
            total_loss = task_loss

            self.log("train/task_loss", task_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=False)

        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)

        # PCC monitoring (mean across all tracks)
        # Note: sync_dist=False to avoid DDP deadlocks during PCC computation
        with torch.no_grad():
            n_tracks = track_preds.shape[-1]
            preds_per_track = track_preds.detach().reshape(-1, n_tracks).T
            targets_per_track = y.detach().reshape(-1, n_tracks).T
            pcc_per_track = self._compute_per_track_pcc(preds_per_track, targets_per_track)
            mean_pcc = pcc_per_track.mean()
            self.log("train/pcc", mean_pcc, on_step=True, on_epoch=True, prog_bar=False, sync_dist=False)

            # Print mean PCC periodically (every 100 steps)
            if batch_idx % 100 == 0:
                log.info(f"[Step {self.global_step}] Train mean PCC: {mean_pcc.item():.4f}")

        return total_loss

    def validation_step(self, batch, batch_idx):
        if self._phase == 'pretrain':
            return self._pretrain_validation_step(batch, batch_idx)
        else:
            return self._finetune_validation_step(batch, batch_idx)

    def _pretrain_validation_step(self, batch, batch_idx):
        """Pre-training validation: MLM metrics."""
        x, _y = batch

        x_masked, mlm_labels = self._mask_input(x)
        encoder_hidden = self._encoder_forward(x_masked)
        mlm_logits = self.mlm_projection(encoder_hidden)

        loss = self.mlm_loss_fn(
            mlm_logits.view(-1, mlm_logits.size(-1)),
            mlm_labels.view(-1),
        )

        with torch.no_grad():
            preds = mlm_logits.argmax(dim=-1)
            mask = mlm_labels != -100
            if mask.any():
                accuracy = (preds[mask] == mlm_labels[mask]).float().mean()
            else:
                accuracy = torch.tensor(0.0, device=self.device)

        self.log("pretrain_val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("pretrain_val/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)

        return {"loss": loss, "accuracy": accuracy}

    def _finetune_validation_step(self, batch, batch_idx):
        """Fine-tuning validation: task metrics."""
        x, y = batch
        y_pred = self(x)

        loss = self.task_loss_fn(y_pred, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)

        n_tracks = y_pred.shape[-1]
        preds_per_track = y_pred.reshape(-1, n_tracks).T
        targets_per_track = y.reshape(-1, n_tracks).T
        pcc_per_track = self._compute_per_track_pcc(preds_per_track, targets_per_track)

        # Store PCC per track for end-of-validation ranking
        return {"loss": loss, "preds": y_pred, "targets": y, "pcc_per_track": pcc_per_track}

    def on_validation_epoch_end(self):
        """Generate PCC ranking and category statistics at end of validation epoch."""
        if self._phase != 'finetune':
            return

        if self.trainer.is_global_zero and hasattr(self, '_validation_pcc_buffer'):
            # Average PCC across all validation batches
            pcc_avg = self._validation_pcc_buffer.mean(dim=0)

            # Save ranking to CSV
            output_dir = Path(self.trainer.log_dir) if self.trainer.log_dir else Path('./outputs')
            save_pcc_ranking(pcc_avg, self.track_names, output_dir, self.current_epoch, prefix="val")

            # Log mean PCC
            log.info(f"[Epoch {self.current_epoch}] Validation mean PCC (all tracks): {pcc_avg.mean().item():.4f}")

            # Category-wise PCC statistics
            cat_stats = compute_category_statistics(pcc_avg, self.track_names)
            print_category_statistics(cat_stats)
            for cat in ['CAGE', 'ChIP-Histone', 'ChIP-TF', 'DNase/ATAC']:
                if cat in cat_stats and cat_stats[cat]['count'] > 0:
                    self.log(f"val/pcc_{cat}", cat_stats[cat]['mean'],
                             on_step=False, on_epoch=True, sync_dist=False)
            self.log("val/pcc_all", cat_stats['All']['mean'],
                     on_step=False, on_epoch=True, sync_dist=False)

            # Four-category mean PCC (paper metric): average of category means
            category_keys = ['DNase/ATAC', 'ChIP-Histone', 'ChIP-TF', 'CAGE']
            cat_means = []
            for cat in category_keys:
                if cat in cat_stats and cat_stats[cat]['count'] > 0:
                    cat_means.append(cat_stats[cat]['mean'])
            if cat_means:
                mean_pcc_4cat = sum(cat_means) / len(cat_means)
                self.log("val/pcc_4cat", mean_pcc_4cat, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
            else:
                # Fallback: use val/pcc_all if no category tracks found
                self.log("val/pcc_4cat", cat_stats['All']['mean'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)

            # Clear buffer for next epoch
            del self._validation_pcc_buffer
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """Collect PCC scores from each validation batch."""
        if self._phase != 'finetune' or outputs is None:
            return

        if 'pcc_per_track' in outputs:
            pcc = outputs['pcc_per_track']
            if not hasattr(self, '_validation_pcc_buffer'):
                self._validation_pcc_buffer = pcc.unsqueeze(0)
            else:
                self._validation_pcc_buffer = torch.cat([self._validation_pcc_buffer, pcc.unsqueeze(0)], dim=0)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        x, y = batch if isinstance(batch, (list, tuple)) else (batch, None)
        return self(x)

    # ======================================================================
    # Optimizer & Scheduler
    # ======================================================================

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.

        Returns different configs based on current phase (pretrain vs finetune).
        """
        from src.utils.scheduler import get_scheduler

        params = filter(lambda p: p.requires_grad, self.parameters())

        # Use pretrain-specific config if in pretrain phase
        if self._phase == 'pretrain':
            opt_cfg = self.hparams.get('pretrain_optimizer_config', self.hparams.optimizer_config)
            sched_cfg = self.hparams.get('pretrain_scheduler_config', self.hparams.scheduler_config)
        else:
            opt_cfg = self.hparams.optimizer_config
            sched_cfg = self.hparams.scheduler_config

        opt_name = opt_cfg['name'] if isinstance(opt_cfg, dict) else opt_cfg.name
        opt_args = opt_cfg['args'] if isinstance(opt_cfg, dict) else opt_cfg.args
        opt_args = dict(opt_args)

        optimizer_cls = getattr(torch.optim, opt_name)
        optimizer = optimizer_cls(params, **opt_args)

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
            f"Scheduler [{self._phase}]: name={sched_name}, warmup={warmup_steps}, "
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
