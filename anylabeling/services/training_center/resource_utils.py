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
    """Resolve a resource path in source, PyInstaller, or Nuitka mode.

    Source mode: resolved relative to the project root.
    PyInstaller: resolved relative to sys._MEIPASS.
    Nuitka standalone: resolved relative to the directory containing the EXE
      (data files are placed alongside the EXE by --include-data-file).

    Args:
        relative_path: Path relative to project root, e.g.
            "anylabeling/services/auto_training/ultralytics/training_worker.py"

    Returns:
        Absolute Path to the resource.
    """
    if getattr(sys, "frozen", False):
        # PyInstaller
        base = Path(getattr(sys, "_MEIPASS", "."))
    elif "__compiled__" in globals():
        # Nuitka standalone: data files are copied alongside the EXE
        base = Path(sys.executable).resolve().parent
    else:
        base = _get_project_root()
    return base / relative_path


def is_frozen() -> bool:
    """Return True if running inside a PyInstaller or Nuitka bundle."""
    # Nuitka: __compiled__ is set in globals
    # PyInstaller: sys.frozen is True
    return bool(getattr(sys, "frozen", False)) or ("__compiled__" in globals())


def _get_bundle_dir() -> Path:
    """Return the directory containing the executable (frozen mode)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return _get_project_root()
