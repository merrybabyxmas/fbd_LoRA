"""Logging setup for FBD-LoRA experiments."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Optional[str] = None,
    log_file: str = "train.log",
    level: int = logging.INFO,
    rank: int = 0,
) -> logging.Logger:
    """Configure root logger with console and optional file handler.

    Args:
        log_dir: Directory to write log file. If None, only console logging.
        log_file: Log file name within log_dir.
        level: Logging level.
        rank: Distributed rank (only rank-0 logs to file).

    Returns:
        Root logger instance.
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = []

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    handlers.append(ch)

    # File handler (rank-0 only)
    if log_dir is not None and rank == 0:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / log_file)
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        handlers.append(fh)

    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Suppress noisy third-party loggers
    for noisy in ["urllib3", "filelock", "fsspec", "git"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger()
