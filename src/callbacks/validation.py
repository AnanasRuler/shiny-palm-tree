"""Check validation every n **global** steps.

Pytorch Lightning has a `val_check_interval` parameter that checks validation every n batches,
but does not support checking every n **global** steps when using gradient accumulation.
This callback provides that functionality.
"""

import logging
from typing import Any, Optional

from pytorch_lightning.callbacks import Callback

log = logging.getLogger(__name__)


class ValEveryNGlobalSteps(Callback):
    """Check validation every n **global** steps.

    Uses a lightweight monkey-patch approach on the epoch loop's _should_check_val_fx
    to trigger validation at specific global steps without corrupting Lightning's
    internal state machine (the previous approach of calling trainer.validate()
    directly caused AssertionError in LoggerConnector._first_loop_iter).

    Args:
        every_n: Run validation every n global steps. If set together with
            num_validations, num_validations takes priority.
        num_validations: Number of equally-spaced validations throughout
            training. The interval is computed as
            ``total_steps // num_validations`` at the start of training.
            Defaults to None (use ``every_n`` directly).
    """
    def __init__(self, every_n: int = 1000, num_validations: Optional[int] = None):
        super().__init__()
        self._every_n_cfg = every_n
        self.every_n = every_n
        self.num_validations = num_validations
        self.last_run: Optional[int] = None
        self._best_val_loss = float('inf')
        self._best_val_pcc = 0.0
        self._original_should_check_val = None
        self._trigger_step: Optional[int] = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Set trigger step before each batch."""
        if self._trigger_step is not None:
            return
        if trainer.global_step % self.every_n == 0 and trainer.global_step != 0:
            if trainer.global_step != self.last_run:
                self._trigger_step = trainer.global_step
                log.info(f"\n{'='*60}")
                log.info(f"Validation scheduled at global step {trainer.global_step}")
                log.info(f"{'='*60}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Patch _should_check_val_fx to return True at the trigger step."""
        if self._trigger_step is None:
            return

        if trainer.global_step == self._trigger_step:
            epoch_loop = trainer.fit_loop.epoch_loop

            # Save original method
            if self._original_should_check_val is None:
                self._original_should_check_val = epoch_loop._should_check_val_fx

            # Monkey-patch to force validation this batch
            def patched_check_val(data_fetcher):
                # Restore original first
                epoch_loop._should_check_val_fx = self._original_should_check_val
                self._original_should_check_val = None
                return True

            epoch_loop._should_check_val_fx = patched_check_val
            log.info(f"Triggering validation at step {trainer.global_step}")
            self._trigger_step = None

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation summary after validation completes."""
        self.last_run = trainer.global_step
        self._log_validation_summary(trainer)

    def _log_validation_summary(self, trainer):
        """Log a summary of validation results."""
        callback_metrics = trainer.callback_metrics

        val_loss = callback_metrics.get('val/loss', None)
        val_pcc_all = callback_metrics.get('val/pcc_all', None)
        val_pcc_4cat = callback_metrics.get('val/pcc_4cat', None)

        log.info(f"\n{'='*60}")
        log.info(f"Validation Results at Step {trainer.global_step}:")
        log.info(f"{'='*60}")

        if val_loss is not None:
            val_loss_val = val_loss.item() if hasattr(val_loss, 'item') else val_loss
            improved_loss = val_loss_val < self._best_val_loss
            if improved_loss:
                self._best_val_loss = val_loss_val
            log.info(f"  val/loss:   {val_loss_val:.6f}  (best: {self._best_val_loss:.6f}{'  ★ NEW BEST!' if improved_loss else ''})")

        # val/pcc_all is the primary PCC metric (all tracks)
        pcc_metric = val_pcc_all if val_pcc_all is not None else val_pcc_4cat
        if pcc_metric is not None:
            pcc_val = pcc_metric.item() if hasattr(pcc_metric, 'item') else pcc_metric
            improved_pcc = pcc_val > self._best_val_pcc
            if improved_pcc:
                self._best_val_pcc = pcc_val
            metric_name = 'val/pcc_all' if val_pcc_all is not None else 'val/pcc_4cat'
            log.info(f"  {metric_name}: {pcc_val:.6f}  (best: {self._best_val_pcc:.6f}{'  ★ NEW BEST!' if improved_pcc else ''})")

        log.info(f"{'='*60}\n")

    def on_fit_start(self, trainer, pl_module):
        """Compute validation interval from total steps if num_validations is set."""
        if self.num_validations is not None and self.num_validations > 0:
            total_steps = trainer.estimated_stepping_batches
            self.every_n = max(1, int(total_steps // self.num_validations))
            log.info(f"\n{'='*60}")
            log.info(f"ValEveryNGlobalSteps callback enabled (auto-computed):")
            log.info(f"  - Total estimated steps: {total_steps}")
            log.info(f"  - Number of validations: {self.num_validations}")
            log.info(f"  - Validation interval: every {self.every_n} global steps")
            log.info(f"{'='*60}\n")
        else:
            log.info(f"\n{'='*60}")
            log.info(f"ValEveryNGlobalSteps callback enabled:")
            log.info(f"  - Validation interval: every {self.every_n} global steps")
            log.info(f"{'='*60}\n")
