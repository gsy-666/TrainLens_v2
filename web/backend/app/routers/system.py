"""System info endpoints (device detection for training defaults)."""

import subprocess

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

_cache: dict = {}


class SetDeviceRequest(BaseModel):
    device: str  # "cpu" | "gpu" | "auto"


def _detect_device() -> dict:
    gpus = []
    # nvidia-smi gives names even without torch.cuda built with CUDA
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                idx, name, mem = [p.strip() for p in line.split(",")]
                gpus.append({"index": int(idx), "name": name, "memory_mb": int(mem)})
    except Exception:
        pass

    cuda_available = False
    if gpus:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

    recommended = "0" if (gpus and cuda_available) else "cpu"
    return {
        "cuda_available": cuda_available,
        "gpus": gpus,
        "recommended": recommended,
    }


@router.get("/system/device")
def get_device():
    if "device" not in _cache:
        _cache["device"] = _detect_device()
    return _cache["device"]


@router.get("/system/device/inference")
def get_inference_device():
    """Get current inference device preference (CPU / GPU / auto)."""
    from anylabeling.views.common.device_manager import device_manager

    dm = device_manager
    current = dm._preferred_device or "auto"
    available = dm.get_available_devices()
    return {
        "current": current,
        "available": available,
    }


@router.post("/system/device/inference")
def set_inference_device(req: SetDeviceRequest):
    """Set inference device preference."""
    from anylabeling.views.common.device_manager import device_manager

    dm = device_manager
    device = req.device.lower()
    if device == "auto":
        dm.reset_device_preference()
    elif device in ("cpu", "gpu"):
        dm.set_device(device.upper())
    else:
        return {"ok": False, "error": f"Invalid device: {req.device}"}
    return {"ok": True, "current": dm._preferred_device or "auto"}
