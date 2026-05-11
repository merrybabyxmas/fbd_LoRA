"""Google Drive sync via rclone with graceful fallback."""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _rclone_available() -> bool:
    """Check if rclone is installed and on PATH."""
    return shutil.which("rclone") is not None


def sync_to_gdrive(
    local_dir: str,
    remote: str,
    gdrive_root: str,
    run_id: str,
    subpath: str = "",
    dry_run: bool = False,
    transfers: int = 8,
    checkers: int = 16,
) -> dict:
    """Sync a local directory to Google Drive using rclone.

    Remote path format:
        {remote}:{gdrive_root}/{run_id}/{subpath}

    Args:
        local_dir: Local directory to upload.
        remote: rclone remote name (e.g., 'fbd_gdrive').
        gdrive_root: Root folder name on GDrive (e.g., 'FBD_LORA_EXPERIMENTS').
        run_id: Run identifier string.
        subpath: Optional subdirectory within run folder.
        dry_run: If True, pass --dry-run to rclone (no actual upload).
        transfers: Number of parallel file transfers.
        checkers: Number of parallel checkers.

    Returns:
        Status dict with keys: success, remote_path, error.
    """
    if not _rclone_available():
        logger.warning("rclone not found; skipping GDrive upload.")
        return {"success": False, "remote_path": None, "error": "rclone not installed"}

    remote_base = f"{remote}:{gdrive_root}/{run_id}"
    remote_path = f"{remote_base}/{subpath}".rstrip("/")

    cmd = [
        "rclone", "copy",
        str(local_dir),
        remote_path,
        "--transfers", str(transfers),
        "--checkers", str(checkers),
        "--create-empty-src-dirs",
    ]
    if dry_run:
        cmd.append("--dry-run")

    logger.info("rclone upload: %s -> %s", local_dir, remote_path)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            logger.error("rclone failed: %s", result.stderr)
            return {"success": False, "remote_path": remote_path, "error": result.stderr}

        logger.info("rclone upload succeeded: %s", remote_path)
        return {"success": True, "remote_path": remote_path, "error": None}

    except subprocess.TimeoutExpired:
        logger.error("rclone timed out uploading %s", local_dir)
        return {"success": False, "remote_path": remote_path, "error": "timeout"}
    except Exception as e:
        logger.error("rclone error: %s", e)
        return {"success": False, "remote_path": remote_path, "error": str(e)}


def validate_remote(
    remote: str,
    gdrive_root: str,
    run_id: str,
    subpath: str,
    required_files: Optional[list] = None,
) -> dict:
    """List remote directory and verify required files exist.

    Args:
        remote: rclone remote name.
        gdrive_root: Root folder name.
        run_id: Run identifier.
        subpath: Subdirectory within run.
        required_files: Files that must exist in the remote dir.

    Returns:
        Validation result dict.
    """
    if not _rclone_available():
        return {"valid": False, "error": "rclone not installed", "files": []}

    remote_path = f"{remote}:{gdrive_root}/{run_id}/{subpath}".rstrip("/")

    try:
        result = subprocess.run(
            ["rclone", "ls", remote_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return {"valid": False, "error": result.stderr, "files": []}

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        remote_files = [line.split()[-1] for line in lines if line]

        missing = []
        if required_files:
            missing = [f for f in required_files if f not in remote_files]

        return {
            "valid": len(missing) == 0,
            "files": remote_files,
            "missing": missing,
            "remote_path": remote_path,
            "error": None,
        }
    except Exception as e:
        return {"valid": False, "error": str(e), "files": []}
