"""System info endpoints (device detection for training defaults)."""

import subprocess

from fastapi import APIRouter

router = APIRouter()

_cache: dict = {}


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
