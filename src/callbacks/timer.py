"""Callback to monitor the speed of each step and each epoch.

Adapted from:
    https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/callbacks/gpu_stats_monitor.html
"""

import time
from typing import Any

from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities.parsing import AttributeDict


class Timer(Callback):
    """Monitor the speed of each step and each epoch.
    
    Args:
        step: Log step time.
        inter_step: Log time between steps.
        epoch: Log epoch time.
        val: Log validation time.
    """
    def __init__(
        self,
        step: bool = True,
        inter_step: bool = True,
        epoch: bool = True,
        val: bool = True,
    ):
        super().__init__()
        self._log_stats = AttributeDict({
            'step_time': step,
            'inter_step_time': inter_step,
            'epoch_time': epoch,
            'val_time': val,
        })
        self._snap_step_time = None
        self._snap_inter_step_time = None
        self._snap_epoch_time = None
        self._snap_val_time = None

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._snap_epoch_time = None

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._snap_step_time = None
        self._snap_inter_step_time = None
        self._snap_epoch_time = time.time()

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._log_stats.step_time:
            self._snap_step_time = time.time()

        if not self._should_log(trainer):
            return

        logs = {}
        if self._log_stats.inter_step_time and self._snap_inter_step_time:
            logs["timer/inter_step"] = time.time() - self._snap_inter_step_time

        if logs:
            pl_module.log_dict(logs, on_step=True, on_epoch=False, prog_bar=False)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._log_stats.inter_step_time:
            self._snap_inter_step_time = time.time()

        if not self._should_log(trainer):
            return

        logs = {}
        if self._log_stats.step_time and self._snap_step_time:
            logs["timer/step"] = time.time() - self._snap_step_time

        if logs:
            pl_module.log_dict(logs, on_step=True, on_epoch=False, prog_bar=False)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logs = {}
        if self._log_stats.epoch_time and self._snap_epoch_time:
            logs["timer/epoch"] = time.time() - self._snap_epoch_time
        if logs:
            pl_module.log_dict(logs, on_step=False, on_epoch=True, prog_bar=False)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._log_stats.val_time:
            self._snap_val_time = time.time()

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Note: Cannot use pl_module.log_dict() in on_validation_end
        # Log directly to the logger instead
        if self._log_stats.val_time and self._snap_val_time:
            val_time = time.time() - self._snap_val_time
            if trainer.logger:
                trainer.logger.log_metrics({"timer/validation": val_time}, step=trainer.global_step)

    @staticmethod
    def _should_log(trainer: Trainer) -> bool:
        return (
            trainer.global_step + 1
        ) % trainer.log_every_n_steps == 0 or trainer.should_stop
