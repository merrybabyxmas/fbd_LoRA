"""Test GDrive sync in dry-run mode (no actual upload)."""

import os
import tempfile
import json
from pathlib import Path

import pytest
from fbd_lora.gdrive import sync_to_gdrive, validate_remote


class TestGDriveSyncDryrun:
    """GDrive sync must gracefully handle missing rclone and dry-run mode."""

    def test_sync_without_rclone_graceful(self):
        """If rclone is not installed, sync must return failure without crashing."""
        import shutil
        rclone_avail = shutil.which("rclone") is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            (Path(tmpdir) / "test.txt").write_text("hello")

            result = sync_to_gdrive(
                local_dir=tmpdir,
                remote="fbd_gdrive",
                gdrive_root="FBD_LORA_EXPERIMENTS",
                run_id="test_run_id",
                subpath="checkpoints/test",
                dry_run=True,  # always dry-run in tests
            )

        if not rclone_avail:
            assert result["success"] is False
            assert "rclone" in result.get("error", "").lower()
        else:
            # With rclone available, dry-run should succeed
            # (rclone dry-run doesn't actually upload)
            assert "remote_path" in result

    def test_sync_returns_required_keys(self):
        """Sync result must always have success, remote_path, error keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = sync_to_gdrive(
                local_dir=tmpdir,
                remote="fbd_gdrive",
                gdrive_root="FBD_LORA_EXPERIMENTS",
                run_id="test_run",
                subpath="",
                dry_run=True,
            )

        assert "success" in result
        assert "remote_path" in result
        assert "error" in result

    def test_validate_remote_without_rclone(self):
        """validate_remote must handle missing rclone gracefully."""
        import shutil
        result = validate_remote(
            remote="fbd_gdrive",
            gdrive_root="FBD_LORA_EXPERIMENTS",
            run_id="nonexistent_run",
            subpath="checkpoints/test",
            required_files=["adapter_model.safetensors"],
        )
        rclone_avail = shutil.which("rclone") is not None
        if not rclone_avail:
            assert result["valid"] is False
            assert "rclone" in result.get("error", "").lower()
        # Either way, must return dict with 'valid' key
        assert "valid" in result

    def test_gdrive_sync_metadata_written(self):
        """Dry-run should not modify local files and must return a dict."""
        import shutil
        rclone_avail = shutil.which("rclone") is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "adapter_config.json"
            test_file.write_text(json.dumps({"r": 4}))

            result = sync_to_gdrive(
                local_dir=tmpdir,
                remote="test_remote_that_does_not_exist",
                gdrive_root="test_root",
                run_id="test_run_dryrun",
                subpath="ckpt",
                dry_run=True,
            )

            # Local file must still exist (sync should not delete local files)
            assert test_file.exists()

        # Result must always be a dict with required keys
        assert isinstance(result, dict)
        assert "success" in result
        assert "remote_path" in result
        assert "error" in result

        # If rclone not available, must report graceful failure
        if not rclone_avail:
            assert result["success"] is False
