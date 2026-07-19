"""TrainLens — Unified device detection and selection service.

Provides local GPU detection via PyTorch/CUDA and a stable
device abstraction for future remote training support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

_logger = logging.getLogger(__name__)


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
            return devices

        if not torch.cuda.is_available():
            _logger.warning(
                "torch.cuda.is_available()=False (PyTorch %s, CUDA %s). "
                "Check NVIDIA driver and CUDA toolkit.", torch_version, cuda_version
            )
            return devices

        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            _logger.warning("torch.cuda.device_count()=0 despite is_available()=True")
            return devices

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

    return devices


def resolve_training_device(device_value: str) -> str:
    """Convert internal device value to Ultralytics-compatible string.

    Mapping:
        "auto"  + CUDA  → "0"
        "auto"  + no CUDA → "cpu"
        "cpu"            → "cpu"
        "cuda:0"         → "0"
        "cuda:1"         → "1"
    """
    device_value = str(device_value).strip().lower()

    # Normalize legacy formats
    if device_value in ("0", "1", "2", "3"):
        # Old bare index → cuda:N
        return device_value  # Ultralytics accepts bare "0", "1"
    if device_value == "cuda":
        return "0"  # Legacy "cuda" → GPU 0

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
