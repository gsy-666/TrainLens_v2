"""TrainLens — Environment Scanner.

Auto-discovers existing CUDA Python environments and diagnoses them.
Uses QThread for background scanning — never blocks GUI.
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

# ── Paths ───────────────────────────────────────────────────────────

REGISTERED_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "TrainLens" / "runtimes" / "registered"

# ── Env State ───────────────────────────────────────────────────────

class EnvStatus(str):
    READY = "ready"
    CUDA_READY_MISSING_PACKAGES = "cuda_ready_missing_packages"
    CPU_ONLY = "cpu_only"
    TORCH_MISSING = "torch_missing"
    CUDA_INIT_FAILED = "cuda_init_failed"
    PYTHON_UNAVAILABLE = "python_unavailable"
    INCOMPATIBLE = "incompatible"


# ── Data models ─────────────────────────────────────────────────────

@dataclass
class EnvironmentInfo:
    """Diagnosed external Python environment."""
    python_path: str = ""
    python_version: str = ""
    environment_root: str = ""
    env_name: str = ""
    source: str = ""  # "conda", "venv", "runtime", "manual"
    torch_installed: bool = False
    torch_version: str = ""
    torch_cuda_version: str = ""
    cuda_available: bool = False
    gpu_count: int = 0
    gpu_names: list = field(default_factory=list)
    gpu_memory_gb: list = field(default_factory=list)
    tensor_test_passed: bool = False
    torchvision_installed: bool = False
    torchvision_version: str = ""
    ultralytics_installed: bool = False
    ultralytics_version: str = ""
    status: str = ""
    error: str = ""
    verified_at: str = ""

    @property
    def is_cuda_ready(self) -> bool:
        return self.status == EnvStatus.READY

    @property
    def needs_packages(self) -> bool:
        return self.status == EnvStatus.CUDA_READY_MISSING_PACKAGES

    def to_registry(self) -> dict:
        return {
            "runtime_id": self._make_runtime_id(),
            "runtime_type": "external",
            "python_path": self.python_path,
            "environment_root": self.environment_root,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "torchvision_version": self.torchvision_version,
            "ultralytics_version": self.ultralytics_version,
            "gpu_names": self.gpu_names,
            "gpu_memory_gb": self.gpu_memory_gb,
            "gpu_count": self.gpu_count,
            "cuda_available": self.cuda_available,
            "tensor_test_passed": self.tensor_test_passed,
            "verification_status": self.status,
            "verified_at": self.verified_at or time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def _make_runtime_id(self) -> str:
        import hashlib
        h = hashlib.sha256(self.python_path.encode()).hexdigest()[:12]
        return f"ext-{h}"


# ── Scanning ────────────────────────────────────────────────────────

def find_candidate_pythons() -> list[str]:
    """Discover candidate Python executables on the system.

    Returns deduplicated list of absolute paths.
    """
    candidates: set[str] = set()

    # 1. Current Python
    if sys.executable and os.path.isfile(sys.executable):
        candidates.add(os.path.abspath(sys.executable))

    # 2. Project .venv
    for base in [
        Path.cwd() / ".venv" / "Scripts" / "python.exe",
        Path(__file__).parent.parent.parent.parent.parent / ".venv" / "Scripts" / "python.exe",
    ]:
        if base.is_file():
            candidates.add(str(base.resolve()))

    # 3. Conda environments (via conda env list)
    for env_dir in _get_conda_env_dirs():
        py = Path(env_dir) / "python.exe"
        if py.is_file():
            candidates.add(str(py.resolve()))

    # 4. Common conda locations (even if conda not on PATH)
    userprofile = os.environ.get("USERPROFILE", str(Path.home()))
    for base in [
        Path(userprofile) / ".conda" / "envs",
        Path("D:\\") / "Anaconda" / "envs",
        Path("D:\\") / "Miniconda" / "envs",
        Path("C:\\") / "Anaconda" / "envs",
        Path("C:\\") / "Miniconda" / "envs",
        Path(userprofile) / "Anaconda" / "envs",
        Path(userprofile) / "Miniconda" / "envs",
    ]:
        if base.is_dir():
            for env in sorted(base.iterdir()):
                py = env / "python.exe"
                if py.is_file():
                    candidates.add(str(py.resolve()))

    # 5. TrainLens runtimes
    from anylabeling.services.training_center.runtime_installer import RUNTIME_ROOT
    if RUNTIME_ROOT.is_dir():
        for runtime_dir in RUNTIME_ROOT.iterdir():
            if runtime_dir.is_dir():
                py = Path(get_venv_python(str(runtime_dir)))
                if py.is_file():
                    candidates.add(str(py))

    # 6. Registered external envs
    if REGISTERED_DIR.is_dir():
        for reg_file in REGISTERED_DIR.glob("*.json"):
            try:
                with open(reg_file) as f:
                    data = json.load(f)
                py = data.get("python_path", "")
                if py and os.path.isfile(py):
                    candidates.add(os.path.abspath(py))
            except Exception:
                pass

    # Filter: python.exe must exist and be accessible
    valid = []
    for c in sorted(candidates):
        try:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                valid.append(os.path.abspath(c))
        except Exception:
            pass

    return valid


def _get_conda_env_dirs() -> list[str]:
    """Get conda environment directories via 'conda env list --json'."""
    dirs: list[str] = []
    # Try JSON format first
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"], capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            envs = data.get("envs", [])
            for env_path in envs:
                if os.path.isdir(env_path):
                    dirs.append(env_path)
            _logger.debug("conda env list --json: found %d envs", len(dirs))
            return dirs
    except Exception:
        pass

    # Fallback: text parsing
    try:
        result = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    env_path = parts[-1]
                    if os.path.isdir(env_path):
                        dirs.append(env_path)
    except Exception:
        pass

    # Also query conda info for base/envs dirs
    try:
        result = subprocess.run(
            ["conda", "info", "--json"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            # Scan envs_dirs from conda config
            for envs_dir in data.get("envs_dirs", []):
                if os.path.isdir(envs_dir):
                    for env in Path(envs_dir).iterdir():
                        py = env / "python.exe"
                        if py.is_file():
                            dirs.append(str(env))
    except Exception:
        pass

    _logger.debug("_get_conda_env_dirs: found %d total", len(dirs))
    return dirs


def get_venv_python(venv_dir: str) -> str:
    """Get python.exe path in a venv."""
    if sys.platform == "win32":
        return str(Path(venv_dir) / "Scripts" / "python.exe")
    return str(Path(venv_dir) / "bin" / "python")


# ── Diagnosis ───────────────────────────────────────────────────────

DIAGNOSIS_SCRIPT = """
import json, sys, traceback
result = {
    "python_version": sys.version.split()[0],
    "torch_installed": False, "torch_version": "", "torch_cuda_version": "",
    "cuda_available": False, "gpu_count": 0, "gpu_names": [], "gpu_memory": [],
    "tensor_test_passed": False,
    "torchvision_installed": False, "torchvision_version": "",
    "ultralytics_installed": False, "ultralytics_version": "",
    "error": "",
}
try:
    # Torch
    try:
        import torch
        result["torch_installed"] = True
        result["torch_version"] = torch.__version__
        result["torch_cuda_version"] = getattr(torch.version, "cuda", None) or ""
        if result["torch_cuda_version"] and torch.cuda.is_available():
            result["cuda_available"] = True
            result["gpu_count"] = torch.cuda.device_count()
            for i in range(result["gpu_count"]):
                result["gpu_names"].append(torch.cuda.get_device_name(i))
                props = torch.cuda.get_device_properties(i)
                result["gpu_memory"].append(round(props.total_memory / (1024**3), 1))
            t = torch.randn(100, 100, device="cuda:0")
            t = t @ t.T
            torch.cuda.synchronize()
            del t
            torch.cuda.empty_cache()
            result["tensor_test_passed"] = True
    except ImportError:
        pass
    except Exception as e:
        result["error"] = f"torch error: {e}"

    # Torchvision
    try:
        import torchvision
        result["torchvision_installed"] = True
        result["torchvision_version"] = torchvision.__version__
    except ImportError:
        pass

    # Ultralytics
    try:
        import ultralytics
        result["ultralytics_installed"] = True
        result["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        pass

except Exception as e:
    result["error"] = str(e)
    traceback.print_exc()

print(json.dumps(result, ensure_ascii=False))
"""


def diagnose_python(python_path: str, timeout: float = 20.0) -> EnvironmentInfo:
    """Run diagnosis script on a Python executable. Returns EnvironmentInfo."""
    info = EnvironmentInfo(python_path=python_path)

    if not os.path.isfile(python_path) or not os.access(python_path, os.X_OK):
        info.status = EnvStatus.PYTHON_UNAVAILABLE
        info.error = "python.exe not found or not executable"
        return info

    try:
        result = subprocess.run(
            [python_path, "-c", DIAGNOSIS_SCRIPT],
            capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout.strip()
        if not output:
            info.status = EnvStatus.PYTHON_UNAVAILABLE
            info.error = result.stderr.strip() or "No output"
            return info

        data = json.loads(output)
        info.python_version = data.get("python_version", "")
        info.torch_installed = data.get("torch_installed", False)
        info.torch_version = data.get("torch_version", "")
        info.torch_cuda_version = data.get("torch_cuda_version", "")
        info.cuda_available = data.get("cuda_available", False)
        info.gpu_count = data.get("gpu_count", 0)
        info.gpu_names = data.get("gpu_names", [])
        info.gpu_memory_gb = data.get("gpu_memory", [])
        info.tensor_test_passed = data.get("tensor_test_passed", False)
        info.torchvision_installed = data.get("torchvision_installed", False)
        info.torchvision_version = data.get("torchvision_version", "")
        info.ultralytics_installed = data.get("ultralytics_installed", False)
        info.ultralytics_version = data.get("ultralytics_version", "")
        info.error = data.get("error", "")

        # Determine environment root
        info.environment_root = str(Path(python_path).parent)
        info.env_name = Path(info.environment_root).name
        info.source = _guess_env_source(python_path)

        # Classify status
        info.status = _classify_env(info)

    except json.JSONDecodeError:
        info.status = EnvStatus.PYTHON_UNAVAILABLE
        info.error = f"Invalid JSON: {output[:200]}"
    except subprocess.TimeoutExpired:
        info.status = EnvStatus.PYTHON_UNAVAILABLE
        info.error = f"Timeout ({timeout}s)"
    except Exception as e:
        info.status = EnvStatus.PYTHON_UNAVAILABLE
        info.error = str(e)

    return info


def _classify_env(info: EnvironmentInfo) -> str:
    if not info.torch_installed:
        return EnvStatus.TORCH_MISSING
    if not info.torch_cuda_version:
        return EnvStatus.CPU_ONLY
    if not info.cuda_available or not info.tensor_test_passed:
        return EnvStatus.CUDA_INIT_FAILED
    if not info.ultralytics_installed:
        return EnvStatus.CUDA_READY_MISSING_PACKAGES
    return EnvStatus.READY


def _guess_env_source(python_path: str) -> str:
    p = python_path.lower()
    if "runtimes" in p and "trainlens" in p:
        return "runtime"
    if "conda" in p or "anaconda" in p or "miniconda" in p:
        return "conda"
    if ".venv" in p or "venv" in p:
        return "venv"
    return "system"


# ── Registration ────────────────────────────────────────────────────

def register_external_env(info: EnvironmentInfo):
    """Save external environment metadata to registered/<id>.json."""
    REGISTERED_DIR.mkdir(parents=True, exist_ok=True)
    reg_data = info.to_registry()
    reg_file = REGISTERED_DIR / f"{reg_data['runtime_id']}.json"
    with open(reg_file, "w") as f:
        json.dump(reg_data, f, indent=2, ensure_ascii=False)
    _logger.info("Registered external env: %s → %s", reg_data["runtime_id"], info.python_path)


def get_registered_envs() -> list[dict]:
    """Load all registered external environment metadata.

    Normalizes verification_status to lowercase 'ready' for compatibility.
    """
    envs = []
    if not REGISTERED_DIR.is_dir():
        return envs
    for reg_file in sorted(REGISTERED_DIR.glob("*.json")):
        try:
            with open(reg_file) as f:
                data = json.load(f)
            # Normalize: accept 'ready', 'READY', 'EnvironmentStatus.READY', etc.
            raw_status = str(data.get("verification_status", "")).strip().lower()
            if raw_status in ("ready", "environmentstatus.ready", "verificationstatus.ready"):
                data["verification_status"] = "ready"
            else:
                data["verification_status"] = raw_status
            # Ensure python_path is absolute
            if data.get("python_path"):
                data["python_path"] = os.path.abspath(data["python_path"])
            _logger.debug(
                "Registered runtime: id=%s python=%s root=%s status=%s",
                data.get("runtime_id"), data.get("python_path"),
                data.get("environment_root"), data.get("verification_status"),
            )
            envs.append(data)
        except Exception as e:
            _logger.warning("Failed to load registered env %s: %s", reg_file.name, e)
    return envs


def unregister_env(runtime_id: str):
    """Remove a registered environment entry."""
    reg_file = REGISTERED_DIR / f"{runtime_id}.json"
    if reg_file.is_file():
        reg_file.unlink()


# ── QThread Scanner ─────────────────────────────────────────────────

class EnvironmentScannerWorker(QObject):
    """Background scanner — discovers and diagnoses Python environments."""

    progress_text = pyqtSignal(str)
    progress_count = pyqtSignal(int, int)  # current, total
    env_found = pyqtSignal(object)  # EnvironmentInfo
    finished = pyqtSignal(list)     # list[EnvironmentInfo]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        results: list[EnvironmentInfo] = []

        self.progress_text.emit("Discovering Python environments...")
        candidates = find_candidate_pythons()
        self.progress_text.emit(f"Found {len(candidates)} candidate environments")

        self.progress_text.emit("Diagnosing environments...")
        for i, py_path in enumerate(candidates):
            if self._cancelled:
                break
            self.progress_count.emit(i + 1, len(candidates))
            _logger.info("Scanning env %d/%d: %s", i + 1, len(candidates), py_path)
            self.progress_text.emit(f"Validating environment {i+1}/{len(candidates)}...")
            info = diagnose_python(py_path, timeout=45.0)
            _logger.info(
                "Env %s: status=%s torch=%s cuda=%s ultralytics=%s error=%s",
                Path(py_path).parent.name,
                info.status, info.torch_version, info.torch_cuda_version,
                info.ultralytics_installed, info.error[:80] if info.error else "",
            )
            results.append(info)
            self.env_found.emit(info)

        self.finished.emit(results)


# ── Ranking ─────────────────────────────────────────────────────────

def rank_environments(envs: list[EnvironmentInfo]) -> list[EnvironmentInfo]:
    """Sort environments by preference: READY CUDA > missing pkgs > CPU > missing torch > unavailable."""
    order = {
        EnvStatus.READY: 0,
        EnvStatus.CUDA_READY_MISSING_PACKAGES: 1,
        EnvStatus.CPU_ONLY: 2,
        EnvStatus.TORCH_MISSING: 3,
        EnvStatus.CUDA_INIT_FAILED: 4,
        EnvStatus.PYTHON_UNAVAILABLE: 5,
        EnvStatus.INCOMPATIBLE: 6,
    }
    return sorted(envs, key=lambda e: (order.get(e.status, 99), not e.ultralytics_installed))


def find_best_env(envs: list[EnvironmentInfo]) -> EnvironmentInfo | None:
    """Find the best ready CUDA environment."""
    ranked = rank_environments(envs)
    for e in ranked:
        if e.is_cuda_ready:
            return e
    for e in ranked:
        if e.needs_packages:
            return e
    return None
