"""TrainLens — GPU Runtime Installer.

Manages creation of isolated venv-based GPU training runtimes.
Uses QThread for background pip install — never blocks GUI.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal

_logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────

RUNTIME_ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "TrainLens" / "runtimes"
CACHE_ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "TrainLens" / "cache" / "runtimes"
NVIDIA_RUNTIME_DIR = RUNTIME_ROOT / "nvidia-cuda"

REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "torchaudio",
    "ultralytics",
]


@dataclass
class RuntimeInfo:
    """Information about an installed training runtime."""
    runtime_id: str = ""
    runtime_path: str = ""
    python_path: str = ""
    torch_version: str = ""
    torch_cuda_version: str = ""
    ultralytics_version: str = ""
    installed_at: str = ""
    install_status: str = ""  # "ready", "failed", "incomplete"
    verification_status: str = ""
    gpu_test_result: str = ""


def get_runtime_info(runtime_dir: str | None = None) -> RuntimeInfo | None:
    """Load runtime.json from the runtime directory. Returns None if not found."""
    if runtime_dir is None:
        runtime_dir = str(NVIDIA_RUNTIME_DIR)
    info_file = Path(runtime_dir) / "runtime.json"
    if not info_file.is_file():
        return None
    try:
        with open(info_file) as f:
            data = json.load(f)
        return RuntimeInfo(**data)
    except Exception:
        return None


def save_runtime_info(info: RuntimeInfo, runtime_dir: str | None = None):
    """Save runtime.json to the runtime directory."""
    if runtime_dir is None:
        runtime_dir = str(NVIDIA_RUNTIME_DIR)
    Path(runtime_dir).mkdir(parents=True, exist_ok=True)
    info_file = Path(runtime_dir) / "runtime.json"
    with open(info_file, "w") as f:
        json.dump(info.__dict__, f, indent=2, ensure_ascii=False)


def _mark_install_status(status: str, runtime_dir: str):
    """Update runtime.json status without full save."""
    info = get_runtime_info(runtime_dir)
    if info is None:
        info = RuntimeInfo(runtime_id="nvidia-cuda", runtime_path=runtime_dir)
    info.install_status = status
    save_runtime_info(info, runtime_dir)


def _find_system_python() -> str:
    """Find a suitable Python executable for creating the venv.

    Prefers the current Python (development mode), falls back to system python.
    """
    # Use current Python if available
    current = sys.executable
    if current and os.path.isfile(current):
        return current

    # Fall back to python3 / python
    for name in ["python3", "python"]:
        try:
            result = subprocess.run(
                [name, "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return name
        except Exception:
            pass
    return "python"


def create_venv(venv_dir: str, python_exe: str | None = None) -> tuple[bool, str]:
    """Create a virtual environment. Returns (success, message)."""
    python_exe = python_exe or _find_system_python()
    venv_path = Path(venv_dir)
    try:
        venv_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [python_exe, "-m", "venv", str(venv_path), "--clear"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return False, result.stderr or "venv creation failed"
        return True, ""
    except Exception as e:
        return False, str(e)


def get_venv_python(venv_dir: str) -> str:
    """Get the python executable path inside a venv."""
    if sys.platform == "win32":
        return str(Path(venv_dir) / "Scripts" / "python.exe")
    return str(Path(venv_dir) / "bin" / "python")


class RuntimeInstallerWorker(QObject):
    """Background worker for pip install into a venv. Emits signals for UI updates."""

    # Progress signals
    stage_changed = pyqtSignal(str)       # CHECKING, DOWNLOADING, INSTALLING, VERIFYING, COMPLETED, FAILED
    progress_text = pyqtSignal(str)        # detailed log line
    progress_percent = pyqtSignal(int)     # 0-100 approximation
    finished = pyqtSignal(bool, str)       # success, message
    elapsed_changed = pyqtSignal(str)      # elapsed time string

    def __init__(self, venv_dir: str, parent=None):
        super().__init__(parent)
        self.venv_dir = venv_dir
        self.python_exe = _find_system_python()
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        """Execute the full install workflow."""
        t0 = time.time()
        self._emit_stage("CHECKING")

        # 1. Check prerequisites
        venv_python = get_venv_python(self.venv_dir)
        self.progress_text.emit(f"Target Python: {self.python_exe}")
        self.progress_text.emit(f"Target venv: {self.venv_dir}")

        # 2. Create venv
        self._emit_stage("DOWNLOADING")
        self.progress_text.emit("Creating virtual environment...")
        ok, msg = create_venv(self.venv_dir, self.python_exe)
        if not ok:
            self._emit_stage("FAILED")
            self.finished.emit(False, f"Failed to create venv: {msg}")
            return

        if self._cancelled:
            self._emit_stage("FAILED")
            self.finished.emit(False, "Cancelled by user")
            return

        # 3. Upgrade pip
        self.progress_text.emit("Upgrading pip...")
        self._run_pip(venv_python, ["install", "--upgrade", "pip"])

        # 4. Install PyTorch with CUDA
        self._emit_stage("INSTALLING")
        self.progress_text.emit("Installing PyTorch with CUDA support...")
        self.progress_text.emit("This may take several minutes...")

        for i, pkg in enumerate(REQUIRED_PACKAGES):
            if self._cancelled:
                self._emit_stage("FAILED")
                self.finished.emit(False, "Cancelled by user")
                return
            pct = int((i / len(REQUIRED_PACKAGES)) * 80) + 10
            self.progress_percent.emit(pct)
            self.progress_text.emit(f"  Installing {pkg}...")
            ok = self._run_pip(venv_python, ["install", pkg])
            if not ok:
                self._emit_stage("FAILED")
                self.finished.emit(False, f"Failed to install {pkg}")
                return

        # 5. Verify
        self._emit_stage("VERIFYING")
        self.progress_percent.emit(90)
        self.progress_text.emit("Verifying GPU runtime...")

        verdict, details = self._verify_runtime(venv_python)
        if verdict:
            self._emit_stage("COMPLETED")
            elapsed = time.time() - t0
            self.elapsed_changed.emit(f"{int(elapsed // 60)}m {int(elapsed % 60)}s")
            self.finished.emit(True, details)
        else:
            self._emit_stage("FAILED")
            self.finished.emit(False, details)

    def _run_pip(self, python_exe: str, args: list) -> bool:
        """Run pip command, streaming output. Returns success."""
        cmd = [python_exe, "-m", "pip"] + args + ["--no-input", "--quiet"]
        try:
            # Use Popen for streaming
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if proc.stdout:
                for line in proc.stdout:
                    if self._cancelled:
                        proc.terminate()
                        return False
                    line = line.strip()
                    if line:
                        self.progress_text.emit(f"    {line[:120]}")
            proc.wait()
            return proc.returncode == 0
        except Exception as e:
            self.progress_text.emit(f"    Error: {e}")
            return False

    def _verify_runtime(self, python_exe: str) -> tuple[bool, str]:
        """Run CUDA verification script. Returns (passed, details)."""
        verify_script = """
import sys, torch
print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"Torch CUDA: {getattr(torch.version, 'cuda', 'N/A')}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    t = torch.randn(100, 100, device='cuda:0')
    t = t @ t.T
    torch.cuda.synchronize()
    del t
    torch.cuda.empty_cache()
    print("CUDA tensor test: PASS")
    sys.exit(0)
else:
    print("CUDA tensor test: SKIP (CUDA not available)")
    sys.exit(1)
"""
        try:
            result = subprocess.run(
                [python_exe, "-c", verify_script],
                capture_output=True, text=True, timeout=30,
            )
            output = result.stdout.strip()
            if result.returncode == 0 and "CUDA tensor test: PASS" in output:
                return True, output
            return False, output or result.stderr or "Verification failed"
        except Exception as e:
            return False, str(e)

    def _emit_stage(self, stage: str):
        self.stage_changed.emit(stage)
        _mark_install_status(stage, self.venv_dir)
