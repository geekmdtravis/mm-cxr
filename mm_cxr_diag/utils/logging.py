"""Logging setup — called explicitly from CLI entrypoints or service startup.

Does not run at import time.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

LOG_FORMAT = "[ %(levelname)-8s ] %(asctime)s | %(name)s | %(message)s"


def setup_logging(
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info",
    log_file: Path | str | None = None,
    log_format: str = LOG_FORMAT,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> None:
    """Configure root logger with console (and optional rotating file) handlers."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(log_format)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
