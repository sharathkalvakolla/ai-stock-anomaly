"""
Shared utilities: logging, directory setup, and path helpers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure root logger with a consistent format for CLI and scripts.

    Args:
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("stock_anomaly_ai")


def ensure_project_dirs() -> None:
    """Create data, models, and outputs directories if they do not exist."""
    for directory in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_path(path: Path | str, base: Optional[Path] = None) -> Path:
    """
    Resolve a path relative to project root or a custom base.

    Args:
        path: File or directory path (absolute or relative).
        base: Base directory for relative paths (default: PROJECT_ROOT).

    Returns:
        Absolute resolved Path.
    """
    base = base or PROJECT_ROOT
    p = Path(path)
    return p if p.is_absolute() else base / p
