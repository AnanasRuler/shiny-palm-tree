"""
Learning Rate Schedulers with Warmup Support

This module provides learning rate schedulers commonly used in deep learning,
particularly for transformer-based models.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineDecayScheduler(_LRScheduler):
    """
    Linear warmup followed by cosine decay.
    
    This is the most commonly used scheduler for transformer training.
    
    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate after decay (default: 0).
        last_epoch: The index of last epoch. Default: -1.
    """
    
    def __init__(
        self, 
        optimizer, 
        warmup_steps: int, 
        total_steps: int, 
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor 
                for base_lr in self.base_lrs
            ]


class LinearWarmupLinearDecayScheduler(_LRScheduler):
    """
    Linear warmup followed by linear decay.
    
    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate after decay (default: 0).
        last_epoch: The index of last epoch. Default: -1.
    """
    
    def __init__(
        self, 
        optimizer, 
        warmup_steps: int, 
        total_steps: int, 
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            decay_factor = 1.0 - progress
            return [
                self.min_lr + (base_lr - self.min_lr) * decay_factor 
                for base_lr in self.base_lrs
            ]


class ConstantWithWarmupScheduler(_LRScheduler):
    """
    Linear warmup followed by constant learning rate.
    
    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps.
        last_epoch: The index of last epoch. Default: -1.
    """
    
    def __init__(
        self, 
        optimizer, 
        warmup_steps: int, 
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Constant
            return list(self.base_lrs)


def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 0,
    total_steps: int = None,
    total_epochs: int = None,
    steps_per_epoch: int = None,
    min_lr: float = 0.0,
    **kwargs
):
    """
    Factory function to create a learning rate scheduler.
    
    Args:
        name: Scheduler name. Options: 
            - "cosine_warmup": Linear warmup + cosine decay
            - "linear_warmup": Linear warmup + linear decay  
            - "constant_warmup": Linear warmup + constant
            - "cosine": CosineAnnealingLR (no warmup)
            - "constant": No scheduling
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps (required for cosine/linear decay).
        total_epochs: Alternative to total_steps (requires steps_per_epoch).
        steps_per_epoch: Steps per epoch (used with total_epochs).
        min_lr: Minimum learning rate for decay schedulers.
        **kwargs: Additional arguments.
        
    Returns:
        dict: Contains scheduler and its configuration for PyTorch Lightning.
    """
    
    # Calculate total_steps if not provided
    if total_steps is None or total_steps <= 0:
        if total_epochs is not None and steps_per_epoch is not None:
            total_steps = total_epochs * steps_per_epoch
        else:
            raise ValueError(
                "Either total_steps or (total_epochs + steps_per_epoch) must be provided "
                f"for scheduler '{name}'. Got total_steps={total_steps}, "
                f"total_epochs={total_epochs}, steps_per_epoch={steps_per_epoch}"
            )
    
    name = name.lower()
    
    if name == "cosine_warmup":
        scheduler = LinearWarmupCosineDecayScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr
        )
        interval = "step"
        
    elif name == "linear_warmup":
        scheduler = LinearWarmupLinearDecayScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr
        )
        interval = "step"
        
    elif name == "constant_warmup":
        scheduler = ConstantWithWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps
        )
        interval = "step"
        
    elif name == "cosine":
        # Standard cosine annealing without warmup (epoch-based)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_epochs if total_epochs else total_steps,
            eta_min=min_lr
        )
        interval = "epoch" if total_epochs else "step"
        
    elif name == "constant" or name == "none":
        # No scheduling
        return None
        
    else:
        raise ValueError(
            f"Unknown scheduler: {name}. "
            f"Available options: cosine_warmup, linear_warmup, constant_warmup, cosine, constant"
        )
    
    return {
        "scheduler": scheduler,
        "interval": interval,
        "frequency": 1,
        "monitor": "val/loss",
        "strict": True,
        "name": f"lr_scheduler_{name}"
    }
