"""NLG-specific callbacks for FBD-LoRA training."""

import logging
from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from fbd_lora.fbd.hooks import FBDHookState
from fbd_lora.fbd.metrics import aggregate_gradient_stats

logger = logging.getLogger(__name__)


class GradientStatsCallback(TrainerCallback):
    """Callback that logs FBD gradient statistics to W&B at regular intervals.

    Args:
        fbd_state: FBDHookState from apply_fbd_to_peft_model.
        log_interval: Log every N steps.
        wandb_run: W&B run object (optional; also logs via trainer).
    """

    def __init__(
        self,
        fbd_state: FBDHookState,
        log_interval: int = 10,
        wandb_run=None,
    ) -> None:
        self.fbd_state = fbd_state
        self.log_interval = log_interval
        self.wandb_run = wandb_run

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        step = state.global_step
        if step % self.log_interval != 0:
            return

        stats = self.fbd_state.get_and_clear_stats()
        if not stats:
            return

        # Log through W&B if available
        if self.wandb_run is not None:
            try:
                self.wandb_run.log(stats, step=step)
            except Exception as e:
                logger.debug("W&B log failed: %s", e)

        # Also log via standard logging
        logger.debug("FBD gradient stats at step %d: %s", step, stats)
