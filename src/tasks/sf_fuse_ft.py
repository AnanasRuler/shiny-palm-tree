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
# Use rank_zero_only for all log methods to avoid duplicate logs in multi-GPU
for _level in ("debug", "info", "warning", "error", "critical"):
    setattr(log, _level, rank_zero_only(getattr(log, _level)))

class SFFuseFineTuningTask(pl.LightningModule):
    def __init__(
        self,
        encoder_config,   # Config for the DNA-LM backbone
        head_config,      # Config for the SFFuseHead
        optimizer_config,
        scheduler_config,
        freeze_encoder=False,
        pretrained_ckpt_path=None, 
        use_mlm_loss=False,        # Flag: Joint Training with MLM
        mlm_lambda=0.1,            # MLM Loss Weight
        mlm_probability=0.15,      # MLM masking probability
        track_names_file=None,     # Path to track names file
        **kwargs,                  # Accept extra args
    ):
        super().__init__()
        
        # Load track names
        num_tracks = head_config.get('num_tracks', 5313) if isinstance(head_config, dict) else getattr(head_config, 'num_tracks', 5313)
        self.track_names = self._load_track_names(track_names_file, num_tracks)
        
        # 1. Capture the Runtime Config (passed from Hydra)
        runtime_encoder_config = encoder_config 
        
        # 2. Extract key parameters for MLM before sanitization
        # Handle nested config structure for accessing vocab_size/d_model
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
        
        # 3. Prepare Sanitized Config for Serialization
        import copy
        if OmegaConf.is_config(encoder_config):
            hparams_encoder_config = OmegaConf.to_container(encoder_config, resolve=True)
        elif isinstance(encoder_config, dict):
            hparams_encoder_config = copy.deepcopy(encoder_config)
        else:
             hparams_encoder_config = encoder_config

        # Sanitize inner config object if needed for JSON serialization
        if isinstance(hparams_encoder_config, dict):
             cfg_inner = hparams_encoder_config.get('config')
             if cfg_inner is not None and not isinstance(cfg_inner, dict) and hasattr(cfg_inner, 'to_dict'):
                  log.info(f"Sanitizing pre-instantiated config object {type(cfg_inner)} to dict for PL logging.")
                  hparams_encoder_config['config'] = cfg_inner.to_dict()

        if OmegaConf.is_config(head_config):
            head_config = OmegaConf.to_container(head_config, resolve=True)
        if OmegaConf.is_config(optimizer_config):
             optimizer_config = OmegaConf.to_container(optimizer_config, resolve=True)
        if OmegaConf.is_config(scheduler_config):
             scheduler_config = OmegaConf.to_container(scheduler_config, resolve=True)
            
        # Save SANITIZED config
        self.save_hyperparameters({
            "encoder_config": hparams_encoder_config,
            "head_config": head_config,
            "optimizer_config": optimizer_config,
            "scheduler_config": scheduler_config,
            "freeze_encoder": freeze_encoder,
            "pretrained_ckpt_path": pretrained_ckpt_path,
            "use_mlm_loss": use_mlm_loss,
            "mlm_lambda": mlm_lambda,
            "mlm_probability": mlm_probability,
            "vocab_size": self._vocab_size,
            "d_model": self._d_model,
            **kwargs
        })
        
        # Store MLM parameters
        self.mlm_probability = mlm_probability
        self.mask_token_id = 4  # 'N' token as mask
        
        # --- 1. Encoder Instantiation ---
        # Use the RUNTIME config (with objects)
        # NOTE: _init_encoder may update self._d_model and self._vocab_size if loading pretrained
        self._init_encoder(runtime_encoder_config, pretrained_ckpt_path)

        # --- 2. Head Instantiation ---
        # Update head_config with correct d_model from encoder (may have changed after loading pretrained)
        from src.models.sf_fuse_head import SFFuseHead
        head_config_updated = head_config.copy() if isinstance(head_config, dict) else dict(head_config)
        head_config_updated['d_model'] = self._d_model
        # Also update head_hidden_dim if it was based on d_model (typically 2 * d_model)
        if 'head_hidden_dim' in head_config_updated:
            # Keep explicit head_hidden_dim if set, or use 2 * d_model as default
            if head_config_updated.get('head_hidden_dim') is None:
                head_config_updated['head_hidden_dim'] = self._d_model * 2
        self.head = SFFuseHead(**head_config_updated)
        # Update hparams with final head_config
        self.hparams.head_config = head_config_updated

        # --- 3. Loss Functions ---
        # NOTE: PoissonNLLLoss with log_input=False requires non-negative predictions.
        # This is guaranteed by SFFuseHead's final Softplus activation.
        # If changing the head activation, switch to log_input=True and remove Softplus.
        self.loss_fn = nn.PoissonNLLLoss(log_input=False, full=True)
        self.track_loss_fn = self.loss_fn
        
        # --- 4. MLM Components ---
        self.use_mlm_loss = use_mlm_loss
        if self.use_mlm_loss:
            self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            # Projection for MLM (Hidden -> Vocab)
            self.mlm_projection = nn.Linear(self._d_model, self._vocab_size)
            log.info(f"MLM Joint Training Enabled: vocab_size={self._vocab_size}, d_model={self._d_model}, lambda={mlm_lambda}")

        # --- 5. Freezing Logic ---
        if freeze_encoder:
            self.freeze_encoder()
        else:
            log.info("Full Fine-tuning Enabled: Encoder parameters will be updated.")

    def _init_encoder(self, encoder_config, pretrained_ckpt_path):
        """
        Flexible Initialization:
        1. If pretrained_ckpt_path is provided: Load weights from HF format checkpoint.
        2. If None: Initialize from scratch using encoder_config.
        
        For pretrained models, the d_model and vocab_size will be read from the 
        pretrained model's config.json, overriding any values in encoder_config.
        """
        if pretrained_ckpt_path:
            log.info(f"Loading pretrained DNA-LM from: {pretrained_ckpt_path}")
            self._load_pretrained_encoder(pretrained_ckpt_path)
        else:
            log.info("No pretrained checkpoint provided. Initializing encoder from scratch...")
            self._init_from_scratch(encoder_config)

    def _load_pretrained_encoder(self, pretrained_ckpt_path):
        """
        Load pretrained encoder from HF format checkpoint.
        
        Supports:
        1. Custom Caduceus models (CaduceusForMaskedLM, Caduceus)
        2. Standard HuggingFace models via AutoModel
        
        Args:
            pretrained_ckpt_path: Path to the pretrained model directory or HF Hub model ID
        """
        import os
        
        # First, try to detect model type from config.json
        model_type = self._detect_model_type(pretrained_ckpt_path)
        log.info(f"Detected model type: {model_type}")
        
        if model_type == "caduceus":
            # Load custom Caduceus model
            self._load_caduceus_pretrained(pretrained_ckpt_path)
        else:
            # Try standard HuggingFace AutoModel
            self._load_hf_automodel(pretrained_ckpt_path)
    
    def _detect_model_type(self, path):
        """Detect model type from config.json."""
        import os
        import json
        
        config_path = os.path.join(path, "config.json") if os.path.isdir(path) else None
        
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            return config_dict.get("model_type", "unknown")
        
        # For HF Hub models, try AutoConfig
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(path, trust_remote_code=True)
            return getattr(config, "model_type", "unknown")
        except Exception:
            return "unknown"
    
    def _load_caduceus_pretrained(self, pretrained_ckpt_path):
        """Load pretrained Caduceus model.
        
        Important: This method handles the case where the pretrained model was trained
        with a shorter sequence length (e.g., 16k) but fine-tuning requires a longer
        sequence length (e.g., 131k). The fourier_max_seq_len parameter is updated
        to the fine-tuning target length to ensure proper positional encoding.
        """
        from src.caduceus.configuration_caduceus import CaduceusConfig
        from src.caduceus.modeling_caduceus import CaduceusForMaskedLM, Caduceus
        
        # Load config from pretrained path
        pretrained_config = CaduceusConfig.from_pretrained(pretrained_ckpt_path)
        
        # Update internal d_model and vocab_size from pretrained config
        self._d_model = pretrained_config.d_model
        self._vocab_size = pretrained_config.vocab_size
        log.info(f"Updated from pretrained config: d_model={self._d_model}, vocab_size={self._vocab_size}")
        
        # --- CRITICAL: Update fourier_max_seq_len for longer fine-tuning sequences ---
        # The pretrained model may have been trained with shorter sequences (e.g., 16k)
        # but fine-tuning requires longer sequences (e.g., 131k for SF-Fuse-style training).
        # We need to update fourier_max_seq_len to support the longer sequence length.
        original_max_seq_len = getattr(pretrained_config, 'fourier_max_seq_len', 16384)
        target_max_seq_len = self._get_target_max_seq_len()
        
        if target_max_seq_len > original_max_seq_len:
            log.info(f"Updating fourier_max_seq_len: {original_max_seq_len} -> {target_max_seq_len} (for fine-tuning)")
            pretrained_config.fourier_max_seq_len = target_max_seq_len
        else:
            log.info(f"fourier_max_seq_len: {original_max_seq_len} (unchanged)")
        
        # Update hparams for proper serialization
        self.hparams.d_model = self._d_model
        self.hparams.vocab_size = self._vocab_size
        
        # Try loading as CaduceusForMaskedLM first (most common for MLM pretrained)
        try:
            model = CaduceusForMaskedLM.from_pretrained(
                pretrained_ckpt_path, 
                config=pretrained_config
            )
            # Extract backbone (Caduceus) from the MLM wrapper
            self.encoder = model.caduceus
            log.info("Successfully loaded pretrained CaduceusForMaskedLM, extracted backbone.")
        except Exception as e:
            log.warning(f"Failed to load as CaduceusForMaskedLM: {e}")
            # Try loading as base Caduceus model
            try:
                self.encoder = Caduceus.from_pretrained(
                    pretrained_ckpt_path,
                    config=pretrained_config
                )
                log.info("Successfully loaded pretrained Caduceus model.")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load Caduceus model from {pretrained_ckpt_path}. "
                    f"CaduceusForMaskedLM error: {e}, Caduceus error: {e2}"
                )
    
    def _get_target_max_seq_len(self):
        """Get the target maximum sequence length for fine-tuning.
        
        This determines the fourier_max_seq_len that should be used during fine-tuning.
        Priority:
        1. encoder_config.config.fourier_max_seq_len (if specified in config)
        2. dataset.max_length (if available in hparams)
        3. Default to 131072 (SF-Fuse standard)
        """
        # Try to get from encoder_config
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
            # Load config to get model parameters
            config = AutoConfig.from_pretrained(pretrained_ckpt_path, trust_remote_code=True)
            
            # Update internal d_model and vocab_size if available
            if hasattr(config, 'd_model'):
                self._d_model = config.d_model
            if hasattr(config, 'vocab_size'):
                self._vocab_size = config.vocab_size
            
            # Update hparams
            self.hparams.d_model = self._d_model
            self.hparams.vocab_size = self._vocab_size
            log.info(f"Updated from pretrained config: d_model={self._d_model}, vocab_size={self._vocab_size}")
            
            # Load model
            self.encoder = AutoModel.from_pretrained(
                pretrained_ckpt_path, 
                trust_remote_code=True
            )
            
            # Handle Wrapper models (e.g. ForMaskedLM) - strip to backbone
            if hasattr(self.encoder, 'caduceus'):  # CaduceusForMaskedLM
                self.encoder = self.encoder.caduceus
            elif hasattr(self.encoder, 'backbone'):
                self.encoder = self.encoder.backbone
            elif hasattr(self.encoder, 'model'):  # Some HF mappings
                self.encoder = self.encoder.model
            elif hasattr(self.encoder, 'layers'):  # Direct Mamba/GPT
                pass
                
            log.info("Successfully loaded pretrained encoder via AutoModel.")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load pretrained model from {pretrained_ckpt_path}: {e}"
            )

    def _init_from_registry(self, encoder_config, ckpt_path=None):
        """Initialize using the internal registry mapping (e.g. for SimpleCNN or specific Mamba)"""
        # Legacy registry support - not used in current implementation
        # Falls back to _init_from_scratch
        log.info("Registry-based initialization not available. Using _init_from_scratch instead.")
        self._init_from_scratch(encoder_config)

    def _init_from_scratch(self, encoder_config):
        """
        Initialize encoder from scratch.
        Supports:
        1. Specialized CaduceusConfig (already instantiated or via Hydra _target_)
        2. Simple parameter dictionary (Legacy/SimpleCNN)
        """
        
        # 1. Identify if we are dealing with Caduceus
        is_caduceus = False
        config_obj = None
        
        # Handle dict vs object access
        _name = encoder_config.get('_name_') if isinstance(encoder_config, dict) else getattr(encoder_config, '_name_', '')
        if 'caduceus' in str(_name):
            is_caduceus = True
        
        # Extract internal config object if present
        if isinstance(encoder_config, dict):
            config_obj = encoder_config.get('config')
        else:
            config_obj = getattr(encoder_config, 'config', None)

        if is_caduceus:
            log.info("Initializing Caduceus/Mamba model...")
            feature_config = None
            
            # Import CaduceusConfig for type checking and manual instantiation
            from src.caduceus.configuration_caduceus import CaduceusConfig
            
            # A. If config_obj is already a CaduceusConfig object, use it directly
            # This is the preferred path when Hydra instantiates recursively
            if config_obj is not None and isinstance(config_obj, CaduceusConfig):
                 log.info(f"Using pre-instantiated config object: {type(config_obj)}")
                 feature_config = config_obj
            
            # B. If config_obj is a dict/DictConfig, we need to instantiate it
            elif config_obj is not None:
                 log.info("Instantiating config from dict/DictConfig...")
                 
                 # Convert DictConfig to plain dict if needed
                 if isinstance(config_obj, DictConfig):
                     config_dict = OmegaConf.to_container(config_obj, resolve=True)
                 elif isinstance(config_obj, dict):
                     config_dict = copy.deepcopy(config_obj)
                 else:
                     config_dict = None
                 
                 if isinstance(config_dict, dict):
                     # Always use manual instantiation to avoid DictConfig issues
                     # with Hydra + transformers PretrainedConfig
                     # Filter out Hydra internal keys and HF PretrainedConfig keys
                     excluded_keys = {
                         # Hydra internal keys
                         '_target_', '_recursive_', '_convert_', '_args_',
                         # HF PretrainedConfig keys
                         'return_dict', 'output_hidden_states', 'output_attentions',
                         'torchscript', 'torch_dtype', 'use_bfloat16', 'tf_legacy_loss',
                         'pruned_heads', 'tie_word_embeddings', 'chunk_size_feed_forward',
                         'is_encoder_decoder', 'is_decoder', 'cross_attention_hidden_size',
                         'add_cross_attention', 'tie_encoder_decoder', 'max_length',
                         'min_length', 'do_sample', 'early_stopping', 'num_beams',
                         'num_beam_groups', 'diversity_penalty', 'temperature', 'top_k',
                         'top_p', 'typical_p', 'repetition_penalty', 'length_penalty',
                         'no_repeat_ngram_size', 'encoder_no_repeat_ngram_size',
                         'bad_words_ids', 'num_return_sequences', 'output_scores',
                         'return_dict_in_generate', 'forced_bos_token_id', 'forced_eos_token_id',
                         'remove_invalid_values', 'exponential_decay_length_penalty',
                         'suppress_tokens', 'begin_suppress_tokens', 'architectures',
                         'finetuning_task', 'id2label', 'label2id', 'tokenizer_class',
                         'prefix', 'bos_token_id', 'pad_token_id', 'eos_token_id',
                         'sep_token_id', 'decoder_start_token_id', 'task_specific_params',
                         'problem_type', '_name_or_path', 'transformers_version', 'model_type',
                         '_commit_hash', 'attn_implementation',
                     }
                     
                     # Parse string representations of nested dicts back to dicts
                     import ast
                     filtered_config = {}
                     for k, v in config_dict.items():
                         if k not in excluded_keys:
                             # Handle string-encoded dicts (e.g., ssm_cfg, attn_cfg)
                             if isinstance(v, str) and v.startswith('{'):
                                 try:
                                     filtered_config[k] = ast.literal_eval(v)
                                 except (ValueError, SyntaxError):
                                     filtered_config[k] = v
                             else:
                                 filtered_config[k] = v
                     
                     log.info(f"Creating CaduceusConfig from dict with keys: {list(filtered_config.keys())}")
                     feature_config = CaduceusConfig(**filtered_config)
                 else:
                     raise ValueError(
                         f"Cannot convert config_obj (type={type(config_obj)}) "
                         f"to a dict for CaduceusConfig creation."
                     )
            
            if feature_config is None:
                raise ValueError("Could not resolve specific Caduceus configuration.")

            # Instantiate Model
            try:
                from src.caduceus.modeling_caduceus import Caduceus
                self.encoder = Caduceus(feature_config)
                log.info("Initialized Caduceus model successfully.")
            except Exception as e:
                log.error(f"Error initializing Caduceus: {e}")
                raise e
                
        elif str(_name) == "simple_cnn":
             # Legacy Simple CNN
             from src.models.simple_cnn import SimpleCNN
             self.encoder = SimpleCNN(
                input_channels=4, 
                d_model=encoder_config.get('d_model', 256) if isinstance(encoder_config, dict) else getattr(encoder_config, 'd_model', 256),
                n_layers=encoder_config.get('n_layer', 4) if isinstance(encoder_config, dict) else getattr(encoder_config, 'n_layer', 4)
             )
        else:
             raise ValueError(f"Unknown encoder config structure. Name: {_name}, Type: {type(encoder_config)}")


        


    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        log.info("Encoder backbone frozen.")

    def _get_hidden_states(self, x):
        """Extract hidden states from encoder output."""
        outputs = self.encoder(x)
        
        # Handle different output types
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs  # (B, L, D)
            
        return hidden_states

    def forward(self, x, return_hidden_states=False):
        """
        Forward pass.
        
        Args:
            x: Input token ids (B, L)
            return_hidden_states: If True, also return hidden states for MLM
            
        Returns:
            track_preds: Predicted genomic tracks (B, T_len, num_tracks)
            hidden_states (optional): Encoder hidden states (B, L, D)
        """
        hidden_states = self._get_hidden_states(x)
        
        # Downstream Task Prediction
        track_preds = self.head(hidden_states)
        
        if return_hidden_states:
            return track_preds, hidden_states
        return track_preds

    def _mask_input(self, input_ids):
        """
        Applies MLM masking logic to input_ids.
        
        Following BERT masking strategy:
        - 15% of tokens are selected for prediction
        - Of those: 80% replaced with [MASK], 10% random, 10% unchanged
        
        Args:
            input_ids: Original input token ids (B, L)
            
        Returns:
            masked_input_ids: Input with masking applied (B, L)
            labels: Labels for MLM loss, -100 for non-masked positions (B, L)
        """
        # Defensive copy to avoid in-place mutation of the caller's tensor
        input_ids = input_ids.clone()
        labels = input_ids.clone()
        
        # Create probability matrix for masking
        probability_matrix = torch.full(
            labels.shape, 
            self.mlm_probability, 
            device=labels.device,
            dtype=torch.float
        )
        
        # Don't mask special tokens (if any) - for DNA we typically don't have special tokens
        # But we should avoid masking padding if present
        
        # Sample masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Only compute loss on masked tokens
        labels[~masked_indices] = -100
        
        # 80% of masked tokens -> replace with [MASK] token
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, 0.8, device=labels.device)
        ).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id

        # 10% of masked tokens -> replace with random token
        # (0.5 of remaining 20% = 10% of total masked)
        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5, device=labels.device)
        ).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            self._vocab_size, 
            labels.shape, 
            device=labels.device, 
            dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        # Remaining 10% -> keep original (no change needed)
        
        return input_ids, labels

    def training_step(self, batch, batch_idx):
        """
        Training step with joint SF-Fuse + MLM loss.
        
        The training uses a shared encoder but different tasks:
        1. SF-Fuse task: Predict genomic tracks from clean sequence
        2. MLM task: Predict masked tokens from masked sequence
        """
        x, y = batch  # x: (B, L), y: (B, T_len, num_tracks)
        
        # --- Strategy: Compute both tasks efficiently ---
        # For SF-Fuse: use clean input
        # For MLM: use masked input
        
        if self.use_mlm_loss:
            # Create masked version for MLM
            x_masked, mlm_labels = self._mask_input(x.clone())
            
            # Forward pass with masked input (used for MLM)
            # We get hidden states from masked input for MLM prediction
            hidden_states_masked = self._get_hidden_states(x_masked)
            
            # Forward pass with clean input (used for SF-Fuse task)
            # This ensures SF-Fuse sees clean sequences for accurate prediction
            hidden_states_clean = self._get_hidden_states(x)
            track_preds = self.head(hidden_states_clean)
            
            # --- 1. SF-Fuse Loss (Primary Task) ---
            task_loss = self.track_loss_fn(track_preds, y)
            
            # --- 2. MLM Loss (Auxiliary Task) ---
            mlm_logits = self.mlm_projection(hidden_states_masked)  # (B, L, vocab_size)
            mlm_loss = self.mlm_loss_fn(
                mlm_logits.view(-1, self._vocab_size), 
                mlm_labels.view(-1)
            )
            
            # --- Combined Loss ---
            total_loss = task_loss + self.hparams.mlm_lambda * mlm_loss
            
            # Logging
            self.log("train/task_loss", task_loss, prog_bar=True, sync_dist=True)
            self.log("train/mlm_loss", mlm_loss, prog_bar=True, sync_dist=True)
            
        else:
            # Only SF-Fuse task (no MLM)
            track_preds = self(x)
            task_loss = self.track_loss_fn(track_preds, y)
            total_loss = task_loss
            
            self.log("train/task_loss", task_loss, prog_bar=True, sync_dist=True)

        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Compute per-track PCC for training monitoring (detached, no grad)
        with torch.no_grad():
            # track_preds, y: (B, T_len, num_tracks) -> mean PCC across tracks
            preds_flat = track_preds.detach()
            targets_flat = y.detach()
            # Reshape to (num_tracks, B*T_len) and compute per-track correlation
            n_tracks = preds_flat.shape[-1]
            preds_per_track = preds_flat.reshape(-1, n_tracks).T   # (num_tracks, B*T_len)
            targets_per_track = targets_flat.reshape(-1, n_tracks).T
            # Compute per-track PCC
            pcc_per_track = self._compute_per_track_pcc(preds_per_track, targets_per_track)
            # Sort PCC values in descending order and take top 10
            top_k = min(10, len(pcc_per_track))
            top_pcc_values, _ = torch.topk(pcc_per_track, k=top_k, largest=True)
            mean_pcc_top10 = top_pcc_values.mean()
            self.log("train/pcc", mean_pcc_top10, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            
            # Print top 3 track names periodically (every 100 steps)
            if batch_idx % 100 == 0:
                log_top_tracks(pcc_per_track, self.track_names, top_k=3, step=self.global_step, prefix="Train")
        
        return total_loss

    @staticmethod
    def _compute_per_track_pcc(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation coefficient for each track.
        
        Args:
            preds: (num_tracks, N) predictions per track.
            targets: (num_tracks, N) targets per track.
            
        Returns:
            Tensor of shape (num_tracks,) with PCC per track.
        """
        # Center
        preds_mean = preds.mean(dim=1, keepdim=True)
        targets_mean = targets.mean(dim=1, keepdim=True)
        preds_centered = preds - preds_mean
        targets_centered = targets - targets_mean
        
        # Covariance and std
        cov = (preds_centered * targets_centered).sum(dim=1)
        preds_std = preds_centered.pow(2).sum(dim=1).sqrt()
        targets_std = targets_centered.pow(2).sum(dim=1).sqrt()
        
        # PCC with numerical stability
        denom = preds_std * targets_std
        pcc = cov / denom.clamp(min=1e-8)
        
        # Replace NaN (constant tracks) with 0
        pcc = torch.where(torch.isfinite(pcc), pcc, torch.zeros_like(pcc))
        return pcc

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        
        loss = self.loss_fn(y_pred, y)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Compute per-track PCC (top 10 tracks)
        # y_pred and y shape: (B, T_len, num_tracks)
        n_tracks = y_pred.shape[-1]
        preds_per_track = y_pred.reshape(-1, n_tracks).T     # (num_tracks, B*T_len)
        targets_per_track = y.reshape(-1, n_tracks).T
        pcc_per_track = self._compute_per_track_pcc(preds_per_track, targets_per_track)
        # Sort PCC values in descending order and take top 10
        top_k = min(10, len(pcc_per_track))
        top_pcc_values, _ = torch.topk(pcc_per_track, k=top_k, largest=True)
        mean_pcc_top10 = top_pcc_values.mean()
        
        self.log("val/pcc:", mean_pcc_top10, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {"loss": loss, "preds": y_pred, "targets": y, "pcc_per_track": pcc_per_track}

    def test_step(self, batch, batch_idx):
        """Test step - same as validation."""
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx):
        """Predict step for inference."""
        x, y = batch if isinstance(batch, (list, tuple)) else (batch, None)
        y_pred = self(x)
        return y_pred

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Uses trainer.estimated_stepping_batches to correctly compute total
        training steps, which accounts for gradient accumulation, multi-GPU,
        and dataset size automatically.
        """
        from src.utils.scheduler import get_scheduler
        
        # Filter params based on what requires grad (handles frozen encoder automatically)
        params = filter(lambda p: p.requires_grad, self.parameters())
        
        opt_cfg = self.hparams.optimizer_config
        # Handle dict or object access
        opt_name = opt_cfg['name'] if isinstance(opt_cfg, dict) else opt_cfg.name
        opt_args = opt_cfg['args'] if isinstance(opt_cfg, dict) else opt_cfg.args
        
        optimizer_cls = getattr(torch.optim, opt_name)
        optimizer = optimizer_cls(params, **opt_args)
        
        # Scheduler configuration
        sched_cfg = self.hparams.scheduler_config
        if sched_cfg is None:
            return optimizer
            
        sched_name = sched_cfg['name'] if isinstance(sched_cfg, dict) else getattr(sched_cfg, 'name', 'constant')
        
        if sched_name in ['constant', 'none', None]:
            return optimizer
        
        # Get scheduler args
        if isinstance(sched_cfg, dict):
            warmup_steps = sched_cfg.get('warmup_steps', 0)
            min_lr = sched_cfg.get('min_lr', 0.0)
        else:
            warmup_steps = getattr(sched_cfg, 'warmup_steps', 0)
            min_lr = getattr(sched_cfg, 'min_lr', 0.0)
        
        # Use trainer.estimated_stepping_batches for accurate total steps.
        # This correctly handles gradient accumulation, multi-GPU, and
        # limit_train_batches automatically.
        total_steps = self.trainer.estimated_stepping_batches
        log.info(f"Scheduler config: name={sched_name}, warmup_steps={warmup_steps}, "
                 f"total_steps={total_steps}, min_lr={min_lr}")
        
        # Create scheduler
        lr_scheduler_config = get_scheduler(
            name=sched_name,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr
        )
        
        if lr_scheduler_config is None:
            return optimizer
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }
