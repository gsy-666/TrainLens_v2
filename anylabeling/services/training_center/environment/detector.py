"""TrainLens Environment Wizard — detection functions.

All subprocess calls use argument lists, shell=False, with timeouts.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from .models import EnvironmentInfo, EnvironmentStatus

_VENV_CANDIDATES_WIN = [
    ".venv/Scripts/python.exe",
    "venv/Scripts/python.exe",
    "env/Scripts/python.exe",
]
_VENV_CANDIDATES_UNIX = [
    ".venv/bin/python",
    "venv/bin/python",
    "env/bin/python",
]


def find_project_python(project_dir: Path) -> Optional[Path]:
    """Find a valid Python executable in the project's virtual environment.

    Returns a valid Path or None if no working environment is found.
    """
    if not project_dir.is_dir():
        return None

    if platform.system() == "Windows":
        candidates = _VENV_CANDIDATES_WIN
    else:
        candidates = _VENV_CANDIDATES_UNIX

    for rel in candidates:
        candidate = project_dir / rel
        if _validate_python(candidate):
            return candidate

    return None


def find_system_python_for_venv() -> Optional[Path]:
    """Find a system Python installation capable of creating venvs.

    Excludes packaged executables like TrainLens.exe and pythonw.exe.
    Returns a Path or None.
    """
    candidates = []

    # 1. Current sys.executable (if not packaged)
    current = Path(sys.executable)
    if current.exists() and not _is_packaged_python(current):
        candidates.append(current)

    # 2. py launcher (Windows)
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["py", "-3", "-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                py_path = Path(result.stdout.strip())
                if py_path.exists():
                    candidates.append(py_path)
        except Exception:
            pass

    # 3. PATH: python
    for name in ("python", "python3"):
        try:
            result = subprocess.run(
                ["where", name] if platform.system() == "Windows" else ["which", name],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    p = Path(line.strip())
                    if p.exists() and not _is_packaged_python(p):
                        candidates.append(p)
        except Exception:
            pass

    # Deduplicate and validate
    seen = set()
    for c in candidates:
        key = c.resolve() if c.exists() else c
        if key in seen:
            continue
        seen.add(key)
        if _validate_python(c) and _supports_venv(c):
            return c

    return None


def inspect_python_environment(python_path: Path) -> EnvironmentInfo:
    """Inspect a Python environment by running a detection script.

    Returns an EnvironmentInfo with status set appropriately.
    """
    info = EnvironmentInfo(python_path=str(python_path))

    if not python_path.exists():
        info.status = EnvironmentStatus.ERROR
        info.error_message = f"Python not found: {python_path}"
        return info

    if not _validate_python(python_path):
        info.status = EnvironmentStatus.ERROR
        info.error_message = f"Python does not execute: {python_path}"
        return info

    # Run the inspection script
    script = _INSPECTION_SCRIPT
    try:
        result = subprocess.run(
            [str(python_path), "-c", script],
            capture_output=True, text=True, timeout=30,
            cwd=str(python_path.parent),
        )
        if result.returncode != 0:
            info.status = EnvironmentStatus.ERROR
            info.error_message = f"Detection script failed (exit {result.returncode}): {result.stderr[:500]}"
            return info

        data = json.loads(result.stdout.strip())
        _populate_from_json(info, data)

        # Determine environment type
        info.environment_type = _detect_env_type(data)

        # Set status
        if not info.executable_valid:
            info.status = EnvironmentStatus.ERROR
        elif not info.pip_available:
            info.status = EnvironmentStatus.WARNING
            info.warning_message = "pip is not available"
        else:
            # Python + pip are valid — READY for general use
            # torch/ultralytics status is informational only (displayed in summary)
            info.status = EnvironmentStatus.READY

        # requirements.txt detection
        req = _find_requirements(python_path)
        if req:
            info.requirements_path = str(req)

    except subprocess.TimeoutExpired:
        info.status = EnvironmentStatus.ERROR
        info.error_message = "Environment inspection timed out (30s)"
    except json.JSONDecodeError:
        info.status = EnvironmentStatus.ERROR
        info.error_message = "Failed to parse detection output as JSON"
    except Exception as e:
        info.status = EnvironmentStatus.ERROR
        info.error_message = f"Inspection failed: {e}"

    return info


def _find_requirements(python_path: Path) -> Optional[Path]:
    """Find requirements.txt by walking up from python_path to find project root.

    Heuristic: look for the parent of .venv/Scripts or venv/Scripts.
    """
    p = python_path.resolve()
    # Walk up to find project root (parent of .venv, venv, or env)
    for _ in range(5):
        p = p.parent
        if p.name in (".venv", "venv", "env"):
            project = p.parent
            req = project / "requirements.txt"
            if req.exists():
                return req
        if (p / "requirements.txt").exists():
            return p / "requirements.txt"
    return None


# ── helpers ────────────────────────────────────────────────────────────


def _validate_python(python_path: Path) -> bool:
    """Check that a python executable can run --version."""
    try:
        result = subprocess.run(
            [str(python_path), "--version"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _supports_venv(python_path: Path) -> bool:
    """Check that python -m venv --help works."""
    try:
        result = subprocess.run(
            [str(python_path), "-m", "venv", "--help"],
            capture_output=True, timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


def _is_packaged_python(python_path: Path) -> bool:
    """Return True if the executable looks like a packaged app (not a real Python)."""
    name_lower = python_path.name.lower()
    if name_lower in ("trainlens.exe", "pythonw.exe"):
        return True
    # Heuristic: PyInstaller one-file exe
    if name_lower.endswith(".exe") and "python" not in name_lower:
        # Could be a real python.exe on Windows — check by running -c
        try:
            result = subprocess.run(
                [str(python_path), "-c", "import sys; sys.stdout.write('ok')"],
                capture_output=True, timeout=10,
            )
            return result.stdout.strip() != "ok"
        except Exception:
            return True
    return False


def _detect_env_type(data: dict) -> str:
    """Determine environment type from inspection data."""
    prefix = data.get("prefix", "")
    base_prefix = data.get("base_prefix", "")
    if prefix and base_prefix and prefix != base_prefix:
        return "venv"
    if data.get("conda_prefix") or data.get("conda_default_env"):
        return "conda"
    return "system"


def _populate_from_json(info: EnvironmentInfo, data: dict):
    """Fill EnvironmentInfo fields from inspection JSON."""
    info.python_version = data.get("version", "")
    info.executable_valid = data.get("executable_valid", False)
    info.pip_available = data.get("pip_available", False)
    info.torch_installed = data.get("torch_installed", False)
    info.torch_version = data.get("torch_version", "")
    info.cuda_available = data.get("cuda_available", False)
    info.cuda_version = data.get("cuda_version", "")
    info.gpu_name = data.get("gpu_name", "")
    info.ultralytics_installed = data.get("ultralytics_installed", False)
    info.ultralytics_version = data.get("ultralytics_version", "")


_INSPECTION_SCRIPT = r"""
import json, sys

result = {
    "version": sys.version.split()[0],
    "executable": sys.executable,
    "executable_valid": True,
    "prefix": getattr(sys, "prefix", ""),
    "base_prefix": getattr(sys, "base_prefix", ""),
    "pip_available": False,
    "torch_installed": False,
    "torch_version": "",
    "cuda_available": False,
    "cuda_version": "",
    "gpu_name": "",
    "ultralytics_installed": False,
    "ultralytics_version": "",
}

# pip
try:
    import pip  # noqa
    result["pip_available"] = True
except Exception:
    pass

# torch
try:
    import torch
    result["torch_installed"] = True
    result["torch_version"] = torch.__version__
    try:
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            result["cuda_version"] = torch.version.cuda or ""
            try:
                result["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
    except Exception:
        pass
except Exception:
    pass

# ultralytics
try:
    import ultralytics
    result["ultralytics_installed"] = True
    result["ultralytics_version"] = ultralytics.__version__
except Exception:
    pass

# conda detection
try:
    result["conda_prefix"] = os.environ.get("CONDA_PREFIX", "")
    result["conda_default_env"] = os.environ.get("CONDA_DEFAULT_ENV", "")
except Exception:
    pass

import os as _os
sys.stdout.write(json.dumps(result))
"""
