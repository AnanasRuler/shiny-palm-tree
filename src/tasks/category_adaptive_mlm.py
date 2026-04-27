"""
Unified MLM Training Strategy

Simple unified MLM pretraining: total_loss = task_loss + λ * mlm_loss

All tracks receive the same MLM weight. Category-specific effects (e.g., DNase/ATAC
benefits more than CAGE) are observed empirically, not designed into the loss function.
"""

import logging
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class UnifiedMLMLoss(nn.Module):
    """Simple unified MLM loss: total_loss = task_loss + lambda * mlm_loss

    Args:
        mlm_lambda: MLM loss weight (default: 0.1)
    """

    def __init__(self, mlm_lambda: float = 0.1):
        super().__init__()
        self.mlm_lambda = mlm_lambda

        log.info("=" * 70)
        log.info("Unified MLM Configuration")
        log.info(f"  MLM Lambda: {self.mlm_lambda}")
        log.info("  All tracks use the same MLM weight")
        log.info("=" * 70)

    def forward(self, task_loss: torch.Tensor, mlm_loss: torch.Tensor) -> torch.Tensor:
        """Compute combined loss: task_loss + mlm_lambda * mlm_loss."""
        return task_loss + self.mlm_lambda * mlm_loss


# Backward compatibility: keep old class name as alias
CategoryAdaptiveMLMLoss = UnifiedMLMLoss


def create_category_adaptive_mlm_loss(
    track_names=None,  # Kept for backward compatibility but unused
    base_mlm_lambda: float = 0.1,
    strategy: str = "default",  # Kept for backward compatibility but unused
) -> UnifiedMLMLoss:
    """Factory function to create UnifiedMLMLoss.

    Note: track_names and strategy parameters are kept for backward compatibility
    but are no longer used. All tracks receive the same MLM weight.
    """
    return UnifiedMLMLoss(mlm_lambda=base_mlm_lambda)
