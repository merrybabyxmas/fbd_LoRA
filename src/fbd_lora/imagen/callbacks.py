"""Checkpoint management for diffusion model training with FBD-LoRA.

FBDDiffusersCheckpointManager saves adapter checkpoints at fixed percentage
intervals of total training steps, with optional W&B logging and GDrive upload.
"""

import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _checksum_dir(dir_path: Path) -> str:
    """SHA-256 checksum of all files in a directory (sorted order)."""
    h = hashlib.sha256()
    for fpath in sorted(dir_path.rglob("*")):
        if fpath.is_file():
            h.update(fpath.name.encode())
            with open(fpath, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
    return h.hexdigest()


class FBDDiffusersCheckpointManager:
    """Checkpoint manager for diffusers-based DreamBooth LoRA + FBD training.

    Saves the PEFT adapter (lora_A, lora_B weights) at fixed percent intervals
    of total training steps. Designed to be called once per optimizer step from
    the training loop.

    Args:
        unet: The PEFT-wrapped UNet model (output of get_peft_model).
        accelerator: Hugging Face Accelerate Accelerator object.
        output_dir: Run output directory (Path or str).
        total_steps: Total number of optimizer steps.
        save_every_percent: Save interval as percentage of total_steps (default 10).
        run_id: Run identifier string (for metadata and GDrive path).
        fbd_config: FBDConfig instance (saved to fbd_config.json in each checkpoint).
        gdrive_remote: rclone remote name (None to skip GDrive upload).
        gdrive_root: GDrive root folder.
        wandb_run: W&B run object (None to skip W&B logging).
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        accelerator,
        output_dir,
        total_steps: int,
        save_every_percent: int = 10,
        run_id: str = "unnamed",
        fbd_config=None,
        gdrive_remote: Optional[str] = None,
        gdrive_root: str = "FBD_LORA_EXPERIMENTS",
        wandb_run=None,
    ) -> None:
        self.unet = unet
        self.accelerator = accelerator
        self.output_dir = Path(output_dir)
        self.total_steps = total_steps
        self.save_every_percent = save_every_percent
        self.run_id = run_id
        self.fbd_config = fbd_config
        self.gdrive_remote = gdrive_remote
        self.gdrive_root = gdrive_root
        self.wandb_run = wandb_run

        self._saved_steps: set = set()
        self._checkpoint_steps = self._compute_schedule(total_steps, save_every_percent)
        logger.info(
            "FBDDiffusersCheckpointManager: total_steps=%d, schedule=%s",
            total_steps, self._checkpoint_steps
        )

    @staticmethod
    def _compute_schedule(total_steps: int, save_every_percent: int) -> list:
        """Compute checkpoint steps at even percentage intervals."""
        if total_steps <= 0:
            return []
        interval = max(1, int(total_steps * save_every_percent / 100))
        steps = [i * interval for i in range(1, 100 // save_every_percent + 1)
                 if i * interval <= total_steps]
        if total_steps not in steps:
            steps.append(total_steps)
        return steps

    def step(self, global_step: int, metrics: Optional[dict] = None) -> None:
        """Call every training optimizer step. Saves checkpoint at schedule points.

        Args:
            global_step: Current optimizer step (1-indexed).
            metrics: Optional dict of training metrics to log with checkpoint.
        """
        if global_step not in self._checkpoint_steps:
            return
        if global_step in self._saved_steps:
            return

        # Only save from main process
        if not self.accelerator.is_main_process:
            self._saved_steps.add(global_step)
            return

        pct = min(100, int(global_step * 100 / self.total_steps)) if self.total_steps > 0 else 0
        label = f"step_{pct:06d}pct"
        self._save(global_step, pct, label, metrics)

    def save_final(self, global_step: int, metrics: Optional[dict] = None) -> None:
        """Save the final checkpoint. Call at end of training."""
        if not self.accelerator.is_main_process:
            return
        if global_step in self._saved_steps:
            return
        self._save(global_step, 100, "final", metrics)

    def _save(
        self,
        step: int,
        pct: int,
        label: str,
        metrics: Optional[dict] = None,
    ) -> None:
        """Internal save method. Saves adapter, config, metadata, and checksums."""
        ckpt_dir = self.output_dir / "checkpoints" / label
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Unwrap model (accelerate may have wrapped it with DDP)
            unwrapped_unet = self.accelerator.unwrap_model(self.unet)

            # Save PEFT adapter
            if hasattr(unwrapped_unet, "save_pretrained"):
                unwrapped_unet.save_pretrained(str(ckpt_dir))
                logger.info("Saved PEFT adapter to %s", ckpt_dir)
            else:
                # Fallback: filter and save only LoRA parameters
                lora_state = {
                    k: v for k, v in unwrapped_unet.state_dict().items()
                    if "lora_" in k
                }
                torch.save(lora_state, ckpt_dir / "lora_weights.pt")
                logger.info("Saved LoRA state dict (%d tensors) to %s", len(lora_state), ckpt_dir)

            # Save FBD config
            if self.fbd_config is not None:
                fbd_dict = dataclasses.asdict(self.fbd_config)
                (ckpt_dir / "fbd_config.json").write_text(
                    json.dumps(fbd_dict, indent=2, default=str)
                )

            # Save metadata
            meta = {
                "step": step,
                "percent": pct,
                "run_id": self.run_id,
                "total_steps": self.total_steps,
                "label": label,
            }
            if metrics:
                meta["metrics"] = {k: float(v) if hasattr(v, "item") else v
                                   for k, v in metrics.items()}
            (ckpt_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

            # Compute and save checksum
            checksum = _checksum_dir(ckpt_dir)
            (ckpt_dir / "checksum.sha256").write_text(checksum)

            self._saved_steps.add(step)
            logger.info("Checkpoint saved: %s (step=%d, %d%%)", label, step, pct)

            # W&B logging
            if self.wandb_run is not None:
                try:
                    log_dict = {"checkpoint/step": step, "checkpoint/pct": pct}
                    if metrics:
                        log_dict.update({f"checkpoint/{k}": v for k, v in metrics.items()})
                    self.wandb_run.log(log_dict, step=step)
                except Exception as e:
                    logger.warning("W&B checkpoint log failed: %s", e)

            # GDrive upload
            if self.gdrive_remote:
                try:
                    from fbd_lora.gdrive import sync_to_gdrive
                    result = sync_to_gdrive(
                        local_dir=str(ckpt_dir),
                        remote=self.gdrive_remote,
                        gdrive_root=self.gdrive_root,
                        run_id=self.run_id,
                        subpath=f"checkpoints/{label}",
                        dry_run=False,
                    )
                    if not result.get("success", False):
                        logger.warning("GDrive upload failed at step %d: %s", step, result.get("error"))
                    else:
                        logger.info("GDrive upload OK for step %d.", step)
                except Exception as e:
                    logger.warning("GDrive sync error at step %d: %s", step, e)

        except Exception as e:
            logger.error("Checkpoint save failed at step %d: %s", step, e, exc_info=True)
