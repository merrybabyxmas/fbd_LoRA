"""FBD checkpoint callback for Transformers Trainer and Accelerate loops."""

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from fbd_lora.gdrive import sync_to_gdrive

logger = logging.getLogger(__name__)


def _checksum_dir(dir_path: Path) -> str:
    """Compute SHA-256 checksum of all files in a directory (sorted order)."""
    h = hashlib.sha256()
    for fpath in sorted(dir_path.rglob("*")):
        if fpath.is_file():
            h.update(fpath.name.encode())
            with open(fpath, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
    return h.hexdigest()


def save_checkpoint_metadata(
    ckpt_dir: Path,
    step: int,
    pct: int,
    run_id: str,
    extra: Optional[dict] = None,
) -> None:
    """Write metadata.json into checkpoint directory."""
    meta = {
        "step": step,
        "percent": pct,
        "run_id": run_id,
    }
    if extra:
        meta.update(extra)
    (ckpt_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


class FBDCheckpointCallback(TrainerCallback):
    """Transformers Trainer callback for FBD-LoRA checkpointing.

    Saves adapter checkpoints every `save_every_percent` percent of total
    training steps and optionally syncs to Google Drive.

    Args:
        run_id: Unique run identifier (used for GDrive path).
        output_dir: Local output directory root.
        fbd_config: FBDConfig instance (saved as fbd_config.yaml).
        save_every_percent: Save interval as percentage of total steps (default 10).
        gdrive_remote: rclone remote name (or None to skip GDrive upload).
        gdrive_root: GDrive root folder name.
        dry_run_gdrive: If True, use rclone --dry-run.
        wandb_run: W&B run object for logging checkpoint events.
    """

    def __init__(
        self,
        run_id: str,
        output_dir: str,
        fbd_config=None,
        save_every_percent: int = 10,
        gdrive_remote: Optional[str] = None,
        gdrive_root: str = "FBD_LORA_EXPERIMENTS",
        dry_run_gdrive: bool = False,
        wandb_run=None,
    ):
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.fbd_config = fbd_config
        self.save_every_percent = save_every_percent
        self.gdrive_remote = gdrive_remote
        self.gdrive_root = gdrive_root
        self.dry_run_gdrive = dry_run_gdrive
        self.wandb_run = wandb_run
        self._saved_steps: set[int] = set()
        self._checkpoint_steps: list[int] = []

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Compute checkpoint schedule from total training steps."""
        total = state.max_steps
        if total <= 0:
            logger.warning("max_steps not known at on_train_begin; checkpointing by pct may be inaccurate.")
            return

        interval = max(1, int(total * self.save_every_percent / 100))
        self._checkpoint_steps = [
            i * interval for i in range(1, 100 // self.save_every_percent + 1)
            if i * interval <= total
        ]
        if total not in self._checkpoint_steps:
            self._checkpoint_steps.append(total)

        logger.info("FBD checkpoint schedule: %s", self._checkpoint_steps)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        """Save checkpoint if current step is in schedule."""
        step = state.global_step
        if step not in self._checkpoint_steps or step in self._saved_steps:
            return

        total = state.max_steps if state.max_steps > 0 else step
        pct = min(100, int(step * 100 / total)) if total > 0 else 0
        self._save(model, step, pct, state)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        """Save final checkpoint."""
        step = state.global_step
        if step not in self._saved_steps:
            self._save(model, step, 100, state, label="final")

    def _save(self, model, step: int, pct: int, state: TrainerState, label: Optional[str] = None) -> None:
        """Perform the actual checkpoint save and optional GDrive upload."""
        if model is None:
            return

        if label is None:
            label = f"step_{pct:06d}pct"

        ckpt_dir = self.output_dir / "checkpoints" / label
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save PEFT adapter
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(str(ckpt_dir))
            else:
                # Fallback: save state dict
                torch.save(model.state_dict(), ckpt_dir / "model_state_dict.pt")

            # Save FBD config
            if self.fbd_config is not None:
                import dataclasses
                fbd_dict = dataclasses.asdict(self.fbd_config)
                (ckpt_dir / "fbd_config.json").write_text(json.dumps(fbd_dict, indent=2))

            # Save trainer state
            state_dict = {
                "global_step": state.global_step,
                "epoch": state.epoch,
                "best_metric": state.best_metric,
            }
            (ckpt_dir / "trainer_state.json").write_text(json.dumps(state_dict, indent=2))

            # Save metadata
            save_checkpoint_metadata(ckpt_dir, step, pct, self.run_id)

            # Compute checksum
            checksum = _checksum_dir(ckpt_dir)
            (ckpt_dir / "checksum.sha256").write_text(checksum)

            self._saved_steps.add(step)
            logger.info("Checkpoint saved: %s (step=%d, %d%%)", ckpt_dir, step, pct)

            # W&B logging
            if self.wandb_run is not None:
                try:
                    self.wandb_run.log({"checkpoint/step": step, "checkpoint/pct": pct})
                except Exception:
                    pass

            # GDrive upload
            if self.gdrive_remote:
                upload_result = sync_to_gdrive(
                    local_dir=str(ckpt_dir),
                    remote=self.gdrive_remote,
                    gdrive_root=self.gdrive_root,
                    run_id=self.run_id,
                    subpath=f"checkpoints/{label}",
                    dry_run=self.dry_run_gdrive,
                )
                upload_success = upload_result.get("success", False)
                # Update metadata with upload status
                meta_path = ckpt_dir / "metadata.json"
                meta = json.loads(meta_path.read_text())
                meta["gdrive_upload"] = upload_result
                meta_path.write_text(json.dumps(meta, indent=2))

                if not upload_success:
                    logger.warning("GDrive upload failed for step %d: %s", step, upload_result.get("error"))

        except Exception as e:
            logger.error("Checkpoint save failed at step %d: %s", step, e)
