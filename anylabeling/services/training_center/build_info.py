"""
TrainLens build and version information.

Provides consistent version, build date, and environment metadata
for both source and frozen (PyInstaller) environments.
"""

import datetime
import os
import platform
import sys
from pathlib import Path

# -- Constants (set at build time by scripts/build_trainlens_*.ps1) --

__trainlens_version__ = "0.1.0-beta"
__trainlens_build_date__ = ""  # Populated by build script
__trainlens_git_commit__ = ""  # Populated by build script

# App identity used for branding in the packaged build
__trainlens_appname__ = "TrainLens"


def get_user_data_dir() -> Path:
    """Return the persistent user data directory (outside install tree)."""
    localappdata = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    return Path(localappdata) / "TrainLens"


def get_log_dir() -> Path:
    """Return the application log directory."""
    return get_user_data_dir() / "logs"


def ensure_user_dirs() -> None:
    """Ensure user data directories exist."""
    get_user_data_dir().mkdir(parents=True, exist_ok=True)
    get_log_dir().mkdir(parents=True, exist_ok=True)


def get_build_info() -> dict:
    """Return a structured dict of build/environment information."""
    frozen = bool(getattr(sys, "frozen", False))
    try:
        import anylabeling.app_info as ai
        app_version = ai.__version__
    except Exception:
        app_version = "N/A"

    info = {
        "trainlens_version": __trainlens_version__,
        "app_version": app_version,
        "build_date": __trainlens_build_date__,
        "git_commit": __trainlens_git_commit__,
        "frozen": frozen,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "user_data_dir": str(get_user_data_dir()),
    }
    if frozen:
        info["_MEIPASS"] = getattr(sys, "_MEIPASS", "N/A")
    return info


def print_build_info():
    """Print build info to stdout (used for --version in frozen mode)."""
    info = get_build_info()
    print(f"TrainLens {info['trainlens_version']}")
    print(f"  Build date:  {info['build_date'] or 'N/A'}")
    print(f"  Git commit:  {info['git_commit'] or 'N/A'}")
    print(f"  Python:      {info['python_version']}")
    print(f"  Frozen:      {info['frozen']}")
    print(f"  Executable:  {info['python_executable']}")
    print(f"  User data:   {info['user_data_dir']}")
