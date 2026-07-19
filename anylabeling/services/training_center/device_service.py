"""TrainLens — Unified device detection and selection service.

Provides local GPU detection via PyTorch/CUDA and a stable
device abstraction for future remote training support.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

_logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Environment State
# ═══════════════════════════════════════════════════════════════════

class EnvironmentState(str, Enum):
    """GPU training environment readiness."""
    GPU_READY = "gpu_ready"
    GPU_PRESENT_CPU_TORCH = "gpu_present_cpu_torch"
    NO_NVIDIA_GPU = "no_nvidia_gpu"
    DRIVER_UNAVAILABLE = "driver_unavailable"
    CUDA_INIT_FAILED = "cuda_init_failed"
    RUNTIME_NOT_INSTALLED = "runtime_not_installed"
    INSTALLING = "installing"
    INSTALL_FAILED = "install_failed"


@dataclass
class EnvironmentDiagnosis:
    """Complete GPU environment assessment."""

    state: EnvironmentState = EnvironmentState.RUNTIME_NOT_INSTALLED
    gpu_name: str = ""
    gpu_count: int = 0
    driver_version: str = ""
    driver_cuda_version: str = ""
    torch_version: str = ""
    torch_cuda_version: str = ""
    cuda_available: bool = False
    current_runtime: str = ""
    recommended_action: str = ""
    diagnostic_message: str = ""


def diagnose_environment() -> EnvironmentDiagnosis:
    """Perform full environment diagnosis.

    Checks nvidia-smi, PyTorch build, and CUDA tensor test.
    """
    d = EnvironmentDiagnosis()

    # ── Check nvidia-smi ──
    d.gpu_name, d.gpu_count, d.driver_version, d.driver_cuda_version = (
        _probe_nvidia_smi()
    )

    # ── Check PyTorch ──
    try:
        import torch
        d.torch_version = torch.__version__
        d.torch_cuda_version = getattr(torch.version, "cuda", None) or ""
    except ImportError:
        d.torch_version = "not installed"
        d.state = EnvironmentState.RUNTIME_NOT_INSTALLED
        if d.gpu_count > 0:
            d.recommended_action = "Install PyTorch with CUDA support"
            d.diagnostic_message = (
                f"NVIDIA {d.gpu_name} detected but PyTorch is not installed."
            )
        return d

    # ── Determine state ──
    if d.gpu_count == 0:
        d.state = EnvironmentState.NO_NVIDIA_GPU
        d.recommended_action = "Use CPU training"
        d.diagnostic_message = "No NVIDIA GPU detected. GPU training unavailable."
    elif not d.torch_cuda_version:
        d.state = EnvironmentState.GPU_PRESENT_CPU_TORCH
        d.recommended_action = "Install CUDA-enabled PyTorch runtime"
        d.diagnostic_message = (
            f"NVIDIA {d.gpu_name} detected but current PyTorch is CPU-only."
        )
    elif torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda:0")
            del t
            torch.cuda.empty_cache()
            d.cuda_available = True
            d.state = EnvironmentState.GPU_READY
            d.recommended_action = "GPU training ready"
            d.diagnostic_message = (
                f"NVIDIA {d.gpu_name} — PyTorch {d.torch_version} "
                f"CUDA {d.torch_cuda_version} — GPU training available."
            )
        except Exception as e:
            d.state = EnvironmentState.CUDA_INIT_FAILED
            d.cuda_available = False
            d.recommended_action = "Check CUDA installation"
            d.diagnostic_message = f"CUDA initialization failed: {e}"
    else:
        d.state = EnvironmentState.DRIVER_UNAVAILABLE
        d.recommended_action = "Install/update NVIDIA driver"
        d.diagnostic_message = (
            f"NVIDIA driver issue: CUDA reported as unavailable."
        )

    return d


def _probe_nvidia_smi() -> tuple[str, int, str, str]:
    """Query nvidia-smi for GPU info. Returns (name, count, driver_ver, cuda_ver)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return ("", 0, "", "")
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if not lines:
            return ("", 0, "", "")
        # First GPU
        parts = [p.strip() for p in lines[0].split(",")]
        name = parts[0] if len(parts) > 0 else ""
        driver_ver = parts[1] if len(parts) > 1 else ""

        # Driver's max supported CUDA version
        try:
            r2 = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10,
            )
            import re
            m = re.search(r"CUDA Version:\s*([\d.]+)", r2.stdout)
            cuda_ver = m.group(1) if m else ""
        except Exception:
            cuda_ver = ""

        return (name, len(lines), driver_ver, cuda_ver)
    except FileNotFoundError:
        return ("", 0, "", "")
    except Exception:
        return ("", 0, "", "")


# ═══════════════════════════════════════════════════════════════════
# Device Info (original)
# ═══════════════════════════════════════════════════════════════════


@dataclass
class DeviceInfo:
    """Unified device descriptor."""

    backend: str  # "cpu", "cuda", "mps", "remote"
    index: int  # device index (0 for CPU)
    display_name: str  # e.g. "GPU 0 — NVIDIA GeForce RTX 3060 · 12.0 GB"
    training_value: str  # internal: "auto", "cpu", "cuda:0", "cuda:1"
    available: bool = True
    total_memory_bytes: int = 0
    capability: str = ""  # compute capability string
    reason_unavailable: str = ""
    execution_location: str = "local"  # "local" | "remote"
    # Runtime binding (for external CUDA environments)
    runtime_id: str = ""
    runtime_python: str = ""
    device_name: str = ""  # raw GPU name from nvidia-smi/registered env


def detect_local_devices() -> list[DeviceInfo]:
    """Detect all available local training devices.

    Always returns at least [CPU]. GPU entries added when CUDA is available.
    Never raises — failures are logged and CPU is always returned.
    """
    devices: list[DeviceInfo] = []

    # ── CPU (always available) ──
    devices.append(DeviceInfo(
        backend="cpu",
        index=0,
        display_name="CPU",
        training_value="cpu",
        available=True,
        execution_location="local",
    ))

    # ── CUDA GPUs ──
    try:
        import torch

        torch_version = getattr(torch, "__version__", "unknown")
        cuda_version = getattr(torch.version, "cuda", None)

        if cuda_version is None:
            _logger.info(
                "Installed PyTorch %s is CPU-only. Install a CUDA-enabled "
                "PyTorch build for NVIDIA GPU training.", torch_version
            )
            # DO NOT return early — fall through to external runtime detection

        if not torch.cuda.is_available():
            _logger.warning(
                "torch.cuda.is_available()=False (PyTorch %s, CUDA %s). "
                "Check NVIDIA driver and CUDA toolkit.", torch_version, cuda_version
            )
            # DO NOT return early — fall through to external runtime detection

        gpu_count = torch.cuda.device_count() if cuda_version else 0
        if gpu_count == 0 and cuda_version:
            _logger.warning("torch.cuda.device_count()=0 despite is_available()=True")
            # Still fall through to external runtimes

        for i in range(gpu_count):
            try:
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / (1024 ** 3)
                cc = f"{props.major}.{props.minor}"

                devices.append(DeviceInfo(
                    backend="cuda",
                    index=i,
                    display_name=f"GPU {i} — {name} · {mem_gb:.1f} GB",
                    training_value=f"cuda:{i}",
                    available=True,
                    total_memory_bytes=props.total_memory,
                    capability=cc,
                    execution_location="local",
                ))
            except Exception as e:
                _logger.warning("Failed to query GPU %d: %s", i, e)
                devices.append(DeviceInfo(
                    backend="cuda",
                    index=i,
                    display_name=f"GPU {i} — Unknown",
                    training_value=f"cuda:{i}",
                    available=False,
                    reason_unavailable=str(e),
                ))

    except ImportError:
        _logger.info("torch not installed — CPU only.")
    except Exception as e:
        _logger.warning("Device detection failed: %s", e)

    _logger.info("After in-process detection: %d devices", len(devices))

    # ── Fallback: external CUDA runtimes and registered envs ──
    if len(devices) <= 1:  # Only CPU found
        _name, gpu_count, _drv, _cuda = _probe_nvidia_smi()
        _logger.info("Fallback: nvidia-smi gpu_count=%d devices=%d", gpu_count, len(devices))
        if gpu_count > 0:
            # 1. Try TrainLens runtimes
            try:
                from anylabeling.services.training_center.runtime_installer import (
                    detect_runtimes, query_runtime_devices,
                )
                runtimes = detect_runtimes()
                _logger.info("TrainLens runtimes found: %d", len(runtimes))
                for rt in runtimes:
                    if rt.install_status == "ready" and rt.verification_status == "PASS":
                        gpu_devices = query_runtime_devices(rt)
                        _logger.info("Runtime %s provides %d GPU devices", rt.runtime_id, len(gpu_devices))
                        for gd in gpu_devices:
                            devices.append(_make_gpu_device(gd))
                        break
            except Exception as e:
                _logger.warning("Runtime scan failed: %s", e)

            # 2. Try registered external environments (fast: use stored GPU info)
            if len(devices) <= 1:
                _logger.info("Trying registered envs (devices=%d)...", len(devices))
                try:
                    from anylabeling.services.training_center.environment_scanner import (
                        get_registered_envs,
                    )
                    regs = get_registered_envs()
                    _logger.info("Registered envs found: %d", len(regs))
                    for reg in regs:
                        status = str(reg.get("verification_status", "")).strip().lower()
                        py = reg.get("python_path", "")
                        py_ok = bool(py and os.path.isfile(py))
                        gpu_names = reg.get("gpu_names", [])
                        gpu_mem = reg.get("gpu_memory_gb", [])
                        _logger.info(
                            "Reg env %s: status=%s py=%s py_ok=%s gpus=%s mem=%s",
                            reg.get("runtime_id"), status, py[:60], py_ok, gpu_names, gpu_mem,
                        )
                        if status != "ready":
                            _logger.debug("Reg env skipped: status=%s", status)
                            continue
                        if not py_ok:
                            _logger.warning("Reg env python missing: %s", py)
                            continue
                        if not gpu_names:
                            _logger.debug("Reg env has no gpu_names")
                            continue
                        for i, name in enumerate(gpu_names):
                            mem = gpu_mem[i] if i < len(gpu_mem) else 0
                            _logger.info("Adding GPU device %d: %s %sGB runtime=%s", i, name, mem, os.path.basename(py))
                            devices.append(DeviceInfo(
                                backend="cuda", index=i,
                                display_name=f"GPU {i} — {name} · {mem:.1f} GB",
                                training_value=f"cuda:{i}",
                                available=True,
                                total_memory_bytes=int(mem * (1024**3)),
                                execution_location="local",
                                runtime_id=reg.get("runtime_id", ""),
                                runtime_python=py,
                                device_name=name,
                            ))
                        break  # Only use first ready registered env
                except Exception as e:
                    _logger.warning("Registered env scan failed: %s", e)

    _logger.info("detect_local_devices returning %d devices", len(devices))

    return devices


def _make_gpu_device(gd: dict) -> DeviceInfo:
    idx = gd["index"]
    name = gd["name"]
    mem = gd.get("total_memory_gb", 0)
    return DeviceInfo(
        backend="cuda", index=idx,
        display_name=f"GPU {idx} — {name} · {mem:.1f} GB",
        training_value=f"cuda:{idx}",
        available=True,
        total_memory_bytes=int(mem * (1024**3)),
        capability=gd.get("capability", ""),
        execution_location="local",
    )


def resolve_training_device(device_value: str) -> str:
    """Convert internal device value to Ultralytics-compatible string.

    Mapping:
        "auto"  + CUDA  → "0"
        "auto"  + no CUDA → "cpu"
        "cpu"            → "cpu"
        "cuda:0"         → "0"
        "cuda:1"         → "1"

    Also checks external runtimes for CUDA availability.
    """
    device_value = str(device_value).strip().lower()

    # Normalize legacy formats
    if device_value in ("0", "1", "2", "3"):
        return device_value
    if device_value == "cuda":
        return "0"

    if device_value.startswith("cuda:"):
        idx = device_value.split(":")[-1]
        return idx

    if device_value == "auto":
        # Check in-process CUDA first
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return "0"
        except Exception:
            pass
        # Check external runtimes
        try:
            from anylabeling.services.training_center.runtime_installer import (
                detect_runtimes, query_runtime_devices,
            )
            runtimes = detect_runtimes()
            for rt in runtimes:
                gpus = query_runtime_devices(rt)
                if gpus:
                    return "0"
        except Exception:
            pass
        return "cpu"

    if device_value.startswith("cuda:"):
        idx = device_value.split(":")[-1]
        return idx  # "cuda:0" → "0"

    if device_value == "auto":
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return "0"
        except Exception:
            pass
        return "cpu"

    if device_value == "cpu":
        return "cpu"

    # Unknown — fallback CPU
    _logger.warning("Unknown device value '%s', falling back to cpu", device_value)
    return "cpu"


def get_device_diagnostics() -> dict:
    """Return diagnostic information about the local training environment.

    Safe to call at any time — never raises.
    """
    info: dict = {
        "torch_version": "not installed",
        "cuda_version": None,
        "cuda_available": False,
        "gpu_count": 0,
        "gpus": [],
        "pytorch_cpu_only": True,
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["pytorch_cpu_only"] = info["cuda_version"] is None

        if not info["pytorch_cpu_only"] and torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_count"] = torch.cuda.device_count()
            for i in range(info["gpu_count"]):
                try:
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    info["gpus"].append({
                        "index": i,
                        "name": name,
                        "total_memory_gb": round(props.total_memory / (1024**3), 1),
                        "capability": f"{props.major}.{props.minor}",
                    })
                except Exception:
                    pass
    except ImportError:
        pass
    except Exception as e:
        info["error"] = str(e)
    return info


def test_gpu_quick(device_value: str = "cuda:0") -> dict:
    """Run a quick CUDA tensor test on the specified device.

    Returns dict with status, device_name, memory info, and timing.
    Never raises — failures are captured in the result dict.
    """
    result = {
        "status": "FAILED",
        "device_value": device_value,
        "device_name": "N/A",
        "total_memory_gb": 0.0,
        "free_memory_gb": 0.0,
        "cuda_version": None,
        "torch_version": "N/A",
        "elapsed_ms": 0.0,
        "error": None,
    }
    import time
    try:
        import torch
        result["torch_version"] = torch.__version__
        result["cuda_version"] = getattr(torch.version, "cuda", None)

        if result["cuda_version"] is None:
            result["error"] = "PyTorch is CPU-only. Install CUDA-enabled build."
            return result

        if not torch.cuda.is_available():
            result["error"] = "CUDA is not available (driver/CUDA toolkit issue)."
            return result

        # Resolve device index
        idx = resolve_training_device(device_value)
        if idx == "cpu":
            result["status"] = "PASS"
            result["device_name"] = "CPU"
            result["error"] = None
            return result

        device_idx = int(idx)
        if device_idx >= torch.cuda.device_count():
            result["error"] = f"GPU {device_idx} does not exist (only {torch.cuda.device_count()} GPUs)."
            return result

        name = torch.cuda.get_device_name(device_idx)
        props = torch.cuda.get_device_properties(device_idx)
        result["device_name"] = name
        result["total_memory_gb"] = round(props.total_memory / (1024**3), 1)

        # Quick tensor test
        t0 = time.perf_counter()
        t = torch.randn(100, 100, device=f"cuda:{device_idx}")
        t = t @ t.T
        torch.cuda.synchronize(device_idx)
        elapsed = time.perf_counter() - t0
        result["elapsed_ms"] = round(elapsed * 1000, 1)

        # Free memory (after sync)
        free_mem, total = torch.cuda.mem_get_info(device_idx)
        result["free_memory_gb"] = round(free_mem / (1024**3), 1)
        result["total_memory_gb"] = round(total / (1024**3), 1)

        del t
        torch.cuda.empty_cache()

        result["status"] = "PASS"
        result["error"] = None

    except ImportError:
        result["error"] = "torch not installed."
    except Exception as e:
        result["error"] = str(e)

    return result


def migrate_legacy_device(value: str) -> str:
    """Convert legacy device strings to the internal format.

    "cpu"     → "cpu"
    "cuda"    → "cuda:0"
    "0"       → "cuda:0"
    "1"       → "cuda:1"
    "0,1"     → "cuda:0"  (take first)
    "auto"    → "auto"
    """
    if not value:
        return "auto"
    v = str(value).strip().lower()
    if v == "auto":
        return "auto"
    if v == "cpu":
        return "cpu"
    if v == "cuda":
        return "cuda:0"
    if v.isdigit():
        return f"cuda:{v}"
    if "," in v:
        first = v.split(",")[0].strip()
        if first.isdigit():
            return f"cuda:{first}"
    if v.startswith("cuda:"):
        return v
    _logger.warning("Unknown legacy device value '%s', defaulting to auto", value)
    return "auto"


def validate_device(value: str) -> tuple[bool, str]:
    """Check whether a device value is usable.

    Returns (ok, message).
    """
    value = migrate_legacy_device(value)
    if value == "auto":
        return True, "Auto-select device"
    if value == "cpu":
        return True, "CPU training"
    if value.startswith("cuda:"):
        try:
            idx = int(value.split(":")[-1])
            import torch
            if not torch.cuda.is_available():
                return False, "CUDA not available (check PyTorch build and drivers)"
            if idx >= torch.cuda.device_count():
                return False, f"GPU {idx} not found (only {torch.cuda.device_count()} GPU(s))"
            return True, f"CUDA GPU {idx} available"
        except ImportError:
            return False, "torch not installed"
        except Exception as e:
            return False, str(e)
    return False, f"Unknown device: {value}"
