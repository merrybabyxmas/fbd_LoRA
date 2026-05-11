"""Custom Trainer subclass for FBD-LoRA NLG experiments.

Extends HuggingFace Trainer to:
- Log FBD gradient statistics every gradient_stats_interval steps.
- Support custom W&B run initialization.
"""

import logging
from typing import Optional

import torch
from transformers import Trainer

from fbd_lora.fbd.hooks import FBDHookState

logger = logging.getLogger(__name__)


class FBDTrainer(Trainer):
    """Trainer with FBD gradient stats logging.

    Args:
        fbd_state: FBDHookState returned by apply_fbd_to_peft_model.
        gradient_stats_interval: Log FBD stats every N steps.
        wandb_run: W&B run object for logging.
        All other args/kwargs passed to Trainer.__init__.
    """

    def __init__(
        self,
        *args,
        fbd_state: Optional[FBDHookState] = None,
        gradient_stats_interval: int = 10,
        wandb_run=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fbd_state = fbd_state
        self.gradient_stats_interval = gradient_stats_interval
        self.wandb_run = wandb_run

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to collect FBD stats after backward."""
        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        """Override log to inject FBD gradient stats."""
        step = self.state.global_step if self.state else 0

        # Inject gradient stats at interval
        if (
            self.fbd_state is not None
            and step > 0
            and step % self.gradient_stats_interval == 0
        ):
            stats = self.fbd_state.get_and_clear_stats()
            if stats:
                logs.update(stats)
                if self.wandb_run is not None:
                    try:
                        self.wandb_run.log(stats, step=step)
                    except Exception:
                        pass

        super().log(logs, start_time)
