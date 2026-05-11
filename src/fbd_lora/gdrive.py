"""Google Drive sync via rclone: upload → verify → delete local."""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# rclone may be installed in ~/bin rather than system PATH
_RCLONE_CANDIDATES = [
    shutil.which("rclone"),
    os.path.expanduser("~/bin/rclone"),
    "/usr/local/bin/rclone",
    "/usr/bin/rclone",
]


def _rclone_bin() -> Optional[str]:
    for candidate in _RCLONE_CANDIDATES:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def sync_to_gdrive(
    local_dir: str,
    remote: str,
    gdrive_root: str,
    run_id: str,
    subpath: str = "",
    dry_run: bool = False,
    transfers: int = 8,
    checkers: int = 16,
    delete_local_after_upload: bool = True,
) -> dict:
    """Upload local_dir to GDrive, verify integrity, then delete local copy.

    Remote path: {remote}:{gdrive_root}/{run_id}/{subpath}

    Returns dict with keys: success, verified, deleted_local, remote_path, error.
    """
    rclone = _rclone_bin()
    if rclone is None:
        logger.warning("rclone not found; skipping GDrive upload.")
        return {"success": False, "verified": False, "deleted_local": False,
                "remote_path": None, "error": "rclone not installed"}

    local_path = Path(local_dir)
    remote_path = f"{remote}:{gdrive_root}/{run_id}/{subpath}".rstrip("/")

    result = {
        "success": False,
        "verified": False,
        "deleted_local": False,
        "remote_path": remote_path,
        "error": None,
    }

    # ── 1. Upload ─────────────────────────────────────────────────────────
    copy_cmd = [
        rclone, "copy",
        str(local_path),
        remote_path,
        "--transfers", str(transfers),
        "--checkers", str(checkers),
        "--create-empty-src-dirs",
        "-v",
    ]
    if dry_run:
        copy_cmd.append("--dry-run")

    logger.info("[GDrive] Uploading %s → %s", local_path, remote_path)
    try:
        r = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            result["error"] = f"rclone copy failed: {r.stderr[:500]}"
            logger.error("[GDrive] Upload failed: %s", result["error"])
            return result
    except subprocess.TimeoutExpired:
        result["error"] = "rclone copy timed out (600s)"
        logger.error("[GDrive] %s", result["error"])
        return result
    except Exception as e:
        result["error"] = str(e)
        return result

    result["success"] = True
    logger.info("[GDrive] Upload complete: %s", remote_path)

    if dry_run:
        result["verified"] = True
        return result

    # ── 2. Verify (rclone check) ──────────────────────────────────────────
    check_cmd = [
        rclone, "check",
        str(local_path),
        remote_path,
        "--checkers", str(checkers),
    ]
    logger.info("[GDrive] Verifying integrity...")
    try:
        r = subprocess.run(check_cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            result["error"] = f"rclone check failed: {r.stderr[:500]}"
            logger.error("[GDrive] Verification failed — local copy retained: %s", result["error"])
            return result
    except subprocess.TimeoutExpired:
        result["error"] = "rclone check timed out"
        logger.error("[GDrive] %s — local copy retained", result["error"])
        return result
    except Exception as e:
        result["error"] = str(e)
        return result

    result["verified"] = True
    logger.info("[GDrive] Verification passed ✓")

    # ── 3. Delete local after confirmed upload ────────────────────────────
    if delete_local_after_upload:
        try:
            shutil.rmtree(str(local_path))
            result["deleted_local"] = True
            logger.info("[GDrive] Local checkpoint deleted: %s", local_path)
        except Exception as e:
            # Non-fatal: upload succeeded, just couldn't clean up
            logger.warning("[GDrive] Could not delete local checkpoint %s: %s", local_path, e)

    return result


def validate_remote(
    remote: str,
    gdrive_root: str,
    run_id: str,
    subpath: str,
    required_files: Optional[list] = None,
) -> dict:
    """List remote directory and verify required files exist."""
    rclone = _rclone_bin()
    if rclone is None:
        return {"valid": False, "error": "rclone not installed", "files": []}

    remote_path = f"{remote}:{gdrive_root}/{run_id}/{subpath}".rstrip("/")

    try:
        r = subprocess.run(
            [rclone, "ls", remote_path],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return {"valid": False, "error": r.stderr, "files": []}

        lines = [line.strip() for line in r.stdout.splitlines() if line.strip()]
        remote_files = [line.split()[-1] for line in lines if line]
        missing = [f for f in (required_files or []) if f not in remote_files]

        return {
            "valid": len(missing) == 0,
            "files": remote_files,
            "missing": missing,
            "remote_path": remote_path,
            "error": None,
        }
    except Exception as e:
        return {"valid": False, "error": str(e), "files": []}
