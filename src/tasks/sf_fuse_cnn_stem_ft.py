"""
SF-Fuse fine-tuning task with CNN stem downsampling.

Architecture:
    Input tokens (B, L=131072)
    → CNN Stem [7 stages of 2x downsampling = 128x total]
    → (B, 1024, d_model)
    → Caduceus Encoder (processes shorter sequences via inputs_embeds)
    → (B, 1024, d_model)
    → Simple Projection Head [center crop to 896 + MLP]
    → (B, 896, num_tracks)

Key differences from sf_fuse_ft.py:
    - All spatial downsampling happens at the beginning via CNN stem
    - Encoder processes much shorter sequences (1024 vs 131072)
    - No pooling in the head, just crop and linear projection
    - No MLM support (not applicable after token-level downsampling)
    - Significantly reduced compute and memory for the encoder
"""

import copy
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
import hydra
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.utilities import rank_zero_only

from src.tasks.track_utils import load_track_names, log_top_tracks, save_pcc_ranking

log = logging.getLogger(__name__)
for _level in ("debug", "info", "warning", "error", "critical"):
    setattr(log, _level, rank_zero_only(getattr(log, _level)))


class SFFuseCNNStemTask(pl.LightningModule):
    """SF-Fuse fine-tuning task with CNN stem for front-end downsampling.
    
    Instead of processing the full 131k sequence through the encoder and
    downsampling at the end (SFFuseHead), this architecture:
    1. Uses a CNN stem to downsample tokens from 131072 → 1024 positions
    2. Feeds the downsampled embeddings into the Caduceus encoder
    3. Uses a simple MLP head (crop + project) for track prediction
    
    Args:
        stem_config: Configuration for CNNDownsampleStem.
        encoder_config: Configuration for the DNA-LM backbone (Caduceus).
        head_config: Configuration for SimpleProjectionHead.
        optimizer_config: Optimizer configuration.
        scheduler_config: LR scheduler configuration.
        freeze_encoder: Whether to freeze encoder weights.
        pretrained_ckpt_path: Path to pretrained encoder checkpoint.
    """
    
    def __init__(
        self,
        stem_config,
        encoder_config,
        head_config,
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
        
        if OmegaConf.is_config(stem_config):
            stem_config = OmegaConf.to_container(stem_config, resolve=True)
        elif not isinstance(stem_config, dict):
            stem_config = dict(stem_config)
        else:
            stem_config = copy.deepcopy(stem_config)
            
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
            "stem_config": stem_config,
            "encoder_config": hparams_encoder_config,
            "head_config": head_config,
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
        
        # --- 1b. Freeze encoder's internal embedding layer ---
        # The CNN stem provides its own embeddings via `inputs_embeds`, bypassing the
        # encoder's built-in embedding layer. We must freeze those unused parameters
        # to avoid DDP errors ("parameters that were not used in producing loss").
        self._freeze_encoder_embeddings()
        
        # --- 2. Initialize CNN Stem ---
        # Update stem config to match encoder's d_model and vocab_size
        stem_config['d_model'] = self._d_model
        stem_config['vocab_size'] = self._vocab_size
        
        from src.models.cnn_stem import CNNDownsampleStem
        self.stem = CNNDownsampleStem(**stem_config)
        log.info(
            f"CNN Stem initialized: vocab_size={self._vocab_size}, d_model={self._d_model}, "
            f"downsample_factor={self.stem.total_downsample_factor}x, "
            f"stages={self.stem.num_downsample_stages}"
        )
        
        # --- 3. Initialize Simple Projection Head ---
        head_config['d_model'] = self._d_model
        if head_config.get('hidden_dim') is None:
            head_config['hidden_dim'] = self._d_model * 2
        
        from src.models.cnn_stem import SimpleProjectionHead
        self.head = SimpleProjectionHead(**head_config)
        
        # Update hparams with final configs
        self.hparams.stem_config = stem_config
        self.hparams.head_config = head_config
        
        # --- 4. Loss Function ---
        self.loss_fn = nn.PoissonNLLLoss(log_input=False, full=True)
        
        # --- 5. Freeze encoder if requested ---
        if freeze_encoder:
            self._freeze_encoder()
        else:
            log.info("Full fine-tuning enabled: encoder parameters will be updated.")

    # ======================================================================
    # Encoder Initialization (adapted from sf_fuse_ft.py)
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
        
        For the CNN stem variant, the encoder's fourier_max_seq_len is set to
        match the downsampled sequence length (not the full input length),
        since the encoder now processes shorter sequences.
        """
        from src.caduceus.configuration_caduceus import CaduceusConfig
        from src.caduceus.modeling_caduceus import CaduceusForMaskedLM, Caduceus
        
        pretrained_config = CaduceusConfig.from_pretrained(pretrained_ckpt_path)
        
        # Update internal parameters from pretrained config
        self._d_model = pretrained_config.d_model
        self._vocab_size = pretrained_config.vocab_size
        log.info(f"Updated from pretrained: d_model={self._d_model}, vocab_size={self._vocab_size}")
        
        # --- Override fourier_max_seq_len for shorter encoder sequences ---
        # The encoder now processes downsampled sequences (e.g., 1024 positions).
        # We don't need fourier_max_seq_len=131072 anymore.
        stem_cfg = self.hparams.get('stem_config', {})
        num_stages = stem_cfg.get('num_downsample_stages', 7)
        downsample_factor = 2 ** num_stages
        
        # Get input length from head config (target_len) or default
        # The downsampled length = input_length / downsample_factor
        # For 131072 / 128 = 1024, we set fourier_max_seq_len = 2048 (with margin)
        downsampled_seq_len = 131072 // downsample_factor
        target_fourier_len = downsampled_seq_len * 2  # 2x margin for safety
        
        original_fourier_len = getattr(pretrained_config, 'fourier_max_seq_len', 16384)
        pretrained_config.fourier_max_seq_len = target_fourier_len
        log.info(
            f"Overriding fourier_max_seq_len: {original_fourier_len} -> {target_fourier_len} "
            f"(encoder seq_len ~{downsampled_seq_len})"
        )
        
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
            
            # Strip wrapper to get backbone
            if hasattr(self.encoder, 'caduceus'):
                self.encoder = self.encoder.caduceus
            elif hasattr(self.encoder, 'backbone'):
                self.encoder = self.encoder.backbone
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
        elif str(_name) == 'simple_cnn':
            from src.models.simple_cnn import SimpleCNN
            d = encoder_config.get('d_model', 256) if isinstance(encoder_config, dict) else getattr(encoder_config, 'd_model', 256)
            self.encoder = SimpleCNN(input_channels=4, d_model=d)
        else:
            raise ValueError(f"Unknown encoder type: {_name}")

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
        
        # Override fourier_max_seq_len for shorter encoder sequences
        stem_cfg = self.hparams.get('stem_config', {})
        num_stages = stem_cfg.get('num_downsample_stages', 7)
        downsample_factor = 2 ** num_stages
        downsampled_len = 131072 // downsample_factor
        target_fourier_len = downsampled_len * 2
        
        original_fourier_len = getattr(feature_config, 'fourier_max_seq_len', 16384)
        feature_config.fourier_max_seq_len = target_fourier_len
        log.info(
            f"Setting fourier_max_seq_len: {original_fourier_len} -> {target_fourier_len} "
            f"(encoder seq_len ~{downsampled_len})"
        )
        
        self.encoder = Caduceus(feature_config)
        self._d_model = feature_config.d_model
        self._vocab_size = feature_config.vocab_size
        log.info(f"Initialized Caduceus from scratch: d_model={self._d_model}")

    # ======================================================================
    # Forward Pass
    # ======================================================================
    
    def _get_hidden_states(self, stem_output):
        """Extract hidden states from encoder given pre-computed embeddings.
        
        Args:
            stem_output: (B, L_down, d_model) from CNN stem.
        Returns:
            (B, L_down, d_model) encoder output hidden states.
        """
        outputs = self.encoder(input_ids=None, inputs_embeds=stem_output)
        
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            return outputs[0]
        else:
            return outputs

    def forward(self, x):
        """
        Forward pass: CNN stem → Encoder → Simple head.
        
        Args:
            x: Input token IDs (B, L), e.g., (B, 131072).
            
        Returns:
            track_preds: (B, target_len, num_tracks), e.g., (B, 896, 5313).
        """
        # 1. CNN Stem: downsample tokens
        stem_out = self.stem(x)                     # (B, L/128, d_model)
        
        # 2. Encoder: process short sequence
        hidden_states = self._get_hidden_states(stem_out)  # (B, L/128, d_model)
        
        # 3. Simple head: crop + project to tracks
        track_preds = self.head(hidden_states)       # (B, 896, num_tracks)
        
        return track_preds

    # ======================================================================
    # Freezing
    # ======================================================================
    
    def _freeze_encoder(self):
        """Freeze encoder parameters (only train stem + head)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        log.info("Encoder backbone frozen. Training stem + head only.")

    def _freeze_encoder_embeddings(self):
        """Freeze the encoder's internal embedding layer.
        
        In the CNN stem architecture, the encoder receives pre-computed embeddings
        via `inputs_embeds`, so its own embedding layer is never used in the forward
        pass. We freeze these parameters to prevent DDP from raising errors about
        parameters that did not receive gradients.
        """
        frozen_count = 0
        # Handle Caduceus model structure: encoder.backbone.embeddings
        embeddings_module = None
        if hasattr(self.encoder, 'backbone') and hasattr(self.encoder.backbone, 'embeddings'):
            embeddings_module = self.encoder.backbone.embeddings
        elif hasattr(self.encoder, 'embeddings'):
            embeddings_module = self.encoder.embeddings
        
        if embeddings_module is not None:
            for param in embeddings_module.parameters():
                param.requires_grad = False
                frozen_count += 1
            log.info(
                f"Froze {frozen_count} encoder embedding parameters "
                f"(unused in CNN stem architecture, bypassed by inputs_embeds)."
            )
        else:
            log.warning(
                "Could not locate encoder embedding layer to freeze. "
                "If using DDP, set find_unused_parameters=True as a fallback."
            )

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
