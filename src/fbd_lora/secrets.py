"""Secret key handling: load from .env, validate, never log values."""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REQUIRED_WANDB_KEYS = ["WANDB_API_KEY", "WANDB_PROJECT"]
REQUIRED_GDRIVE_KEYS = ["GDRIVE_REMOTE", "GDRIVE_ROOT"]


def load_env_file(env_file: str = ".env") -> None:
    """Load environment variables from a .env file without printing values.

    Args:
        env_file: Path to the .env file (relative or absolute).
    """
    env_path = Path(env_file)
    if not env_path.exists():
        logger.warning("No .env file found at %s; relying on shell env vars.", env_path)
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Only set if not already defined in environment
            if key not in os.environ:
                os.environ[key] = value

    logger.debug("Loaded env file: %s", env_path)


def key_present(key: str) -> bool:
    """Return True if key is set and non-empty; never print value."""
    val = os.environ.get(key, "")
    return bool(val)


def validate_secrets(
    require_wandb: bool = True,
    require_gdrive: bool = False,
) -> None:
    """Validate that required secret keys are present.

    Args:
        require_wandb: If True, raise if W&B keys missing.
        require_gdrive: If True, raise if GDrive keys missing.

    Raises:
        EnvironmentError: If a required key is missing.
    """
    missing = []

    if require_wandb:
        wandb_mode = os.environ.get("WANDB_MODE", "online")
        if wandb_mode != "disabled":
            for k in REQUIRED_WANDB_KEYS:
                if not key_present(k):
                    missing.append(k)

    if require_gdrive:
        for k in REQUIRED_GDRIVE_KEYS:
            if not key_present(k):
                missing.append(k)

    if missing:
        raise EnvironmentError(
            f"Required secret keys not set: {missing}. "
            "Check your .env file or environment variables."
        )

    # Log presence (never values)
    present = []
    for k in REQUIRED_WANDB_KEYS + REQUIRED_GDRIVE_KEYS + ["HF_TOKEN", "OPENAI_API_KEY"]:
        if key_present(k):
            present.append(k)
    logger.info("Secret keys present: %s", present)


def get_hf_token() -> Optional[str]:
    """Return HuggingFace token or None."""
    return os.environ.get("HF_TOKEN") or None


def get_wandb_key() -> Optional[str]:
    """Return W&B API key or None."""
    return os.environ.get("WANDB_API_KEY") or None
