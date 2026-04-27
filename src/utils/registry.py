"""Class registry for models, layers, optimizers, schedulers, and callbacks."""

# Model Registry
model = {
    "mamba_lm": "src.models.mamba_lm.MambaLMHeadModel",
    "caduceus": "src.caduceus.modeling_caduceus.CaduceusForMaskedLM", 
    "caduceus_lm": "src.caduceus.modeling_caduceus.CaduceusForMaskedLM",
    "simple_cnn": "src.models.simple_cnn.SimpleConvModel",
}

# Optimizer Registry
optimizer = {
    "adam": "torch.optim.Adam",
    "adamw": "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd": "torch.optim.SGD",
}

# Scheduler Registry
scheduler = {
    "constant": "transformers.get_constant_schedule",
    "plateau": "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step": "torch.optim.lr_scheduler.StepLR",
    "multistep": "torch.optim.lr_scheduler.MultiStepLR",
    "cosine": "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup": "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup": "transformers.get_cosine_schedule_with_warmup",
}

# Callbacks Registry
callbacks = {
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint": "pytorch_lightning.callbacks.ModelCheckpoint",
    "model_checkpoint_every_n_steps": "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping": "pytorch_lightning.callbacks.EarlyStopping",
    "rich_model_summary": "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar": "pytorch_lightning.callbacks.RichProgressBar",
    "params": "src.callbacks.params.ParamsLog",
    "timer": "src.callbacks.timer.Timer",
    "val_every_n_global_steps": "src.callbacks.validation.ValEveryNGlobalSteps",
}
