"""TrainLens Environment Wizard — operations (venv creation, pip, requirements)."""

import subprocess
from pathlib import Path
from typing import Optional, Callable, Tuple

from .detector import _validate_python


def create_project_venv(
    project_dir: Path,
    system_python: Path,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, Path, str]:
    """Create .venv in project_dir using system_python.

    Returns (success, venv_python_path, message).
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)

    if not project_dir.is_dir():
        return False, Path(), f"Project directory does not exist: {project_dir}"

    venv_dir = project_dir / ".venv"
    if platform.system() == "Windows":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    # Check existing .venv
    if venv_dir.exists():
        if _validate_python(venv_python):
            log(f"Existing .venv is valid: {venv_python}")
            return True, venv_python, "Existing .venv is valid"
        else:
            return False, Path(), (
                f".venv exists but Python is invalid at {venv_python}. "
                "Remove it manually or choose a different project."
            )

    # Create venv
    log(f"Creating .venv in {project_dir} ...")
    log(f"Using system Python: {system_python}")
    try:
        result = subprocess.run(
            [str(system_python), "-m", "venv", str(venv_dir)],
            capture_output=True, text=True, timeout=120,
            cwd=str(project_dir),
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            return False, Path(), f"venv creation failed (exit {result.returncode}): {stderr[:500]}"
    except subprocess.TimeoutExpired:
        return False, Path(), "venv creation timed out (120s)"
    except Exception as e:
        return False, Path(), f"venv creation error: {e}"

    # Validate
    if not _validate_python(venv_python):
        return False, Path(), f"venv created but Python is not executable: {venv_python}"

    log(f"venv created successfully: {venv_python}")

    # Check pip
    if not _pip_available(venv_python):
        log("pip not found, running ensurepip ...")
        ok, msg = ensure_pip(venv_python, log_callback)
        if not ok:
            return True, venv_python, f"venv created but pip fix failed: {msg}"
        log("pip is now available")

    return True, venv_python, "Environment created successfully"


def ensure_pip(
    python_path: Path,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """Ensure pip is available in the given Python environment."""
    def log(msg: str):
        if log_callback:
            log_callback(msg)

    if _pip_available(python_path):
        return True, "pip is already available"

    log(f"Installing pip via ensurepip: {python_path}")
    try:
        result = subprocess.run(
            [str(python_path), "-m", "ensurepip", "--upgrade"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return False, f"ensurepip failed (exit {result.returncode}): {result.stderr[:300]}"
    except subprocess.TimeoutExpired:
        return False, "ensurepip timed out (60s)"
    except Exception as e:
        return False, f"ensurepip error: {e}"

    if not _pip_available(python_path):
        return False, "ensurepip ran but pip is still not available"

    return True, "pip installed successfully"


def install_requirements(
    python_path: Path,
    requirements_path: Path,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """Install packages from a requirements.txt file using Popen for real-time output.

    Returns (success, message).
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)

    if not requirements_path.exists():
        return False, f"requirements.txt not found: {requirements_path}"

    if not _pip_available(python_path):
        return False, "pip is not available in the selected Python environment"

    log(f"Installing requirements from: {requirements_path}")
    log(f"Python: {python_path}")
    log("Running pip install (output shown after completion)...")

    cmd = [str(python_path), "-m", "pip", "install", "-r", str(requirements_path)]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(requirements_path.parent),
        )
        try:
            out, err = proc.communicate(timeout=600)
            if out:
                lines = out.splitlines()
                if len(lines) > 200:
                    log(f"Output truncated ({len(lines)} lines total, showing last 200)")
                    lines = lines[-200:]
                for line in lines:
                    log(line)
            if err and proc.returncode != 0:
                err_lines = err.splitlines()
                for line in err_lines[-20:]:
                    log(f"STDERR: {line}")
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return False, "pip install timed out (600s)"

        if proc.returncode != 0:
            return False, f"pip install failed (exit {proc.returncode})"
    except Exception as e:
        return False, f"pip install error: {e}"

    log("Requirements installed successfully")
    return True, "Requirements installed successfully"


# ── helpers ────────────────────────────────────────────────────────────


def _pip_available(python_path: Path) -> bool:
    """Check if pip is available via `python -m pip --version`."""
    try:
        result = subprocess.run(
            [str(python_path), "-m", "pip", "--version"],
            capture_output=True, timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


import platform  # noqa: E402 (used in create_project_venv)
