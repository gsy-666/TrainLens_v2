"""
Resource path resolver for training_center.

Unified resource resolution for both source and frozen (PyInstaller) environments.
"""

import os
import sys
from pathlib import Path


def _get_project_root() -> Path:
    """Returns the project root directory (source mode)."""
    # __file__ = anylabeling/services/training_center/resource_utils.py
    # Go up 3 levels: training_center → services → anylabeling → project_root
    return Path(__file__).resolve().parent.parent.parent.parent


def resource_path(relative_path: str) -> Path:
    """Resolve a resource path in either source or frozen mode.

    Source mode: resolved relative to the project root.
    Frozen mode: resolved relative to sys._MEIPASS (PyInstaller bundle).

    Args:
        relative_path: Path relative to project root, e.g.
            "anylabeling/services/auto_training/ultralytics/training_worker.py"

    Returns:
        Absolute Path to the resource.
    """
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", "."))
    else:
        base = _get_project_root()
    return base / relative_path


def is_frozen() -> bool:
    """Return True if running inside a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))
