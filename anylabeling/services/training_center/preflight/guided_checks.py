"""TrainLens Preflight — Guided Training checks (model, YAML, device)."""

import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from . import (
    PreflightIssue, PreflightResult, PreflightSeverity,
    SUPPORTED_MODEL_EXTS,
)

# ── Python / Package checks ─────────────────────────────────────────

def check_python_packages(result: PreflightResult):
    """Check ultralytics and torch are importable."""
    try:
        import ultralytics  # noqa: F401
        result.add(PreflightIssue(
            code="ULTRALYTICS_OK", severity=PreflightSeverity.PASS,
            title="Ultralytics is available",
            message="Ultralytics package is installed and importable.",
        ))
    except ImportError:
        result.add(PreflightIssue(
            code="ULTRALYTICS_MISSING", severity=PreflightSeverity.ERROR,
            title="Ultralytics not installed",
            message="Cannot import ultralytics. Install with: pip install ultralytics",
            suggestion="Run: pip install ultralytics",
        ))
        return  # torch check depends on ultralytics

    try:
        import torch  # noqa: F401
        result.add(PreflightIssue(
            code="TORCH_OK", severity=PreflightSeverity.PASS,
            title="PyTorch is available",
            message="PyTorch is installed and importable.",
        ))
    except ImportError:
        result.add(PreflightIssue(
            code="TORCH_MISSING", severity=PreflightSeverity.ERROR,
            title="PyTorch not installed",
            message="Cannot import torch. Install with: pip install torch",
            suggestion="Run: pip install torch",
        ))


# ── Model checks ────────────────────────────────────────────────────

def check_model(result: PreflightResult, model_path: str, task_type: str = ""):
    """Check model file exists, is readable, has valid extension."""
    if not model_path or not model_path.strip():
        result.add(PreflightIssue(
            code="MODEL_EMPTY", severity=PreflightSeverity.ERROR,
            title="Model path is empty",
            message="No model file specified.",
            field_name="model",
        ))
        return

    path = Path(model_path)

    if not path.exists():
        result.add(PreflightIssue(
            code="MODEL_NOT_FOUND", severity=PreflightSeverity.ERROR,
            title="Model file not found",
            message=f"Model file does not exist: {model_path}",
            field_name="model", path=model_path,
            suggestion="Verify the model path or download the model file.",
        ))
        return

    if not path.is_file():
        result.add(PreflightIssue(
            code="MODEL_NOT_FILE", severity=PreflightSeverity.ERROR,
            title="Model path is not a file",
            message=f"Model path is not a regular file: {model_path}",
            field_name="model", path=model_path,
        ))
        return

    # Check readability
    if not os.access(str(path), os.R_OK):
        result.add(PreflightIssue(
            code="MODEL_NOT_READABLE", severity=PreflightSeverity.ERROR,
            title="Model file is not readable",
            message=f"Cannot read model file: {model_path}",
            field_name="model", path=model_path,
        ))
        return

    # Check extension
    ext = path.suffix.lower()
    if ext not in SUPPORTED_MODEL_EXTS:
        result.add(PreflightIssue(
            code="MODEL_EXT_UNSUPPORTED", severity=PreflightSeverity.WARNING,
            title="Model extension may not be supported",
            message=f"Model has extension '{ext}'. Supported: {', '.join(sorted(SUPPORTED_MODEL_EXTS))}",
            field_name="model", path=model_path,
        ))

    # Try to read model metadata for task compatibility
    _check_model_task_compat(result, path, task_type)

    result.add(PreflightIssue(
        code="MODEL_OK", severity=PreflightSeverity.PASS,
        title="Model file is valid",
        message=f"Model: {path.name} ({_format_size(path.stat().st_size)})",
        path=model_path,
    ))


def _check_model_task_compat(result: PreflightResult, model_path: Path, task_type: str):
    """Check model task type matches configured task (best-effort)."""
    if not task_type:
        return

    try:
        import torch
        ckpt = torch.load(str(model_path), map_location="cpu", weights_only=True)
    except Exception:
        return  # Cannot read model, skip task check

    model_task = None
    # Ultralytics models store task in various places
    if isinstance(ckpt, dict):
        model_task = (
            ckpt.get("train_args", {}).get("task")
            or (ckpt.get("model", None) or {}).__dict__.get("task")
        )

    if model_task and task_type.lower() != str(model_task).lower():
        # Only WARNING — model task inference is unreliable
        result.add(PreflightIssue(
            code="MODEL_TASK_MISMATCH", severity=PreflightSeverity.WARNING,
            title="Model task may not match configured task",
            message=f"Model was trained for '{model_task}' but you selected '{task_type}'. "
                    f"Training may produce unexpected results.",
            field_name="model",
            suggestion=f"Use a {task_type} pre-trained model for best results.",
        ))


# ── Device checks ───────────────────────────────────────────────────

def check_device(result: PreflightResult, device: str):
    """Check device configuration is valid.

    For CPU: always passes.
    For CUDA: checks in-process PyTorch (for when GUI has CUDA).
    For external runtime: call check_device_with_runtime instead.
    """
    if not device or not device.strip():
        result.add(PreflightIssue(
            code="DEVICE_EMPTY", severity=PreflightSeverity.WARNING,
            title="Device not specified",
            message="No device configured. Defaulting to CPU.",
            field_name="device",
        ))
        return

    device_lower = device.lower()

    if device_lower == "cpu":
        result.add(PreflightIssue(
            code="DEVICE_CPU_OK", severity=PreflightSeverity.PASS,
            title="Device: CPU",
            message="Training will run on CPU.",
            field_name="device",
        ))
        return

    if device_lower == "auto":
        # Auto — will be resolved later; just note it
        result.add(PreflightIssue(
            code="DEVICE_AUTO", severity=PreflightSeverity.PASS,
            title="Device: Auto",
            message="Device will be auto-selected at training start.",
            field_name="device",
        ))
        return

    if device_lower.startswith("cuda") or device.isdigit():
        _check_cuda_inprocess(result, device, device_lower)
        return

    result.add(PreflightIssue(
        code="DEVICE_UNKNOWN", severity=PreflightSeverity.WARNING,
        title=f"Unknown device: {device}",
        message=f"Device '{device}' is not recognized.",
        field_name="device",
    ))


def _check_cuda_inprocess(result: PreflightResult, device: str, device_lower: str):
    """Check CUDA availability in the current Python process."""
    try:
        import torch
        if not torch.cuda.is_available():
            result.add(PreflightIssue(
                code="CUDA_UNAVAILABLE", severity=PreflightSeverity.ERROR,
                title="CUDA is not available",
                message=f"Device '{device}' requires CUDA, but CUDA is not available.",
                field_name="device",
                suggestion="Install CUDA or switch to 'cpu' device.",
            ))
            return
    except ImportError:
        result.add(PreflightIssue(
            code="TORCH_MISSING", severity=PreflightSeverity.ERROR,
            title="Cannot check CUDA",
            message="PyTorch is not installed. Cannot verify CUDA availability.",
            field_name="device",
        ))
        return

    if device_lower.startswith("cuda:"):
        try:
            idx = int(device.split(":")[1])
        except (IndexError, ValueError):
            result.add(PreflightIssue(
                code="DEVICE_INVALID", severity=PreflightSeverity.ERROR,
                title="Invalid device format",
                message=f"Device '{device}' is not a valid CUDA device specifier.",
                field_name="device",
            ))
            return
        if idx >= torch.cuda.device_count():
            result.add(PreflightIssue(
                code="CUDA_INDEX_OOB", severity=PreflightSeverity.ERROR,
                title="CUDA device index out of range",
                message=f"Device index {idx} >= available GPUs ({torch.cuda.device_count()}).",
                field_name="device",
                suggestion=f"Use one of: 0-{torch.cuda.device_count() - 1}.",
            ))
            return

    # All good
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = "Unknown"
    result.add(PreflightIssue(
        code="CUDA_AVAILABLE", severity=PreflightSeverity.PASS,
        title=f"CUDA GPU available: {gpu_name}",
        message=f"Device '{device}' is valid. GPU: {gpu_name}.",
        field_name="device",
    ))


def check_device_with_runtime(result: PreflightResult, ctx) -> None:
    """Check CUDA using an external training runtime (subprocess).

    Use when GUI PyTorch is CPU-only but an external CUDA runtime is registered.
    Accesses ctx.runtime_python, ctx.requested_device, ctx.resolved_device.

    Args:
        result: PreflightResult to append issues to
        ctx: GuidedPreflightContext with runtime fields
    """
    device = ctx.device or ""
    device_lower = device.lower()

    if device_lower == "cpu":
        _check_cuda_inprocess(result, device, device_lower)
        return

    if device_lower == "auto":
        runtime_python = getattr(ctx, "runtime_python", "") or ""
        if runtime_python and os.path.isfile(runtime_python):
            # Has external CUDA runtime — will auto-select GPU
            info = _query_runtime_cuda(runtime_python)
            if info and info.get("cuda_available"):
                gpu_names = info.get("gpu_names", ["Unknown"])
                result.add(PreflightIssue(
                    code="CUDA_AUTO_RUNTIME", severity=PreflightSeverity.PASS,
                    title=f"Auto: will use GPU via external runtime",
                    message=f"External runtime provides: {', '.join(gpu_names)}.",
                    field_name="device",
                ))
            else:
                result.add(PreflightIssue(
                    code="CUDA_AUTO_FALLBACK", severity=PreflightSeverity.WARNING,
                    title="Auto: no CUDA runtime, will use CPU",
                    message="No ready external CUDA runtime found. Training will use CPU.",
                    field_name="device",
                ))
        else:
            result.add(PreflightIssue(
                code="CUDA_AUTO_FALLBACK", severity=PreflightSeverity.WARNING,
                title="Auto: no CUDA runtime, will use CPU",
                message="No external runtime configured. Training will use CPU.",
                field_name="device",
            ))
        return

    if not device_lower.startswith("cuda") and not device.isdigit():
        _check_cuda_inprocess(result, device, device_lower)
        return

    # ── CUDA device explicitly selected ──
    runtime_python = getattr(ctx, "runtime_python", "") or ""
    runtime_id = getattr(ctx, "runtime_id", "") or ""

    if not runtime_python or not os.path.isfile(runtime_python):
        result.add(PreflightIssue(
            code="CUDA_RUNTIME_MISSING", severity=PreflightSeverity.ERROR,
            title="Registered CUDA runtime missing",
            message=f"Device '{device}' requires an external CUDA runtime, "
                    f"but the registered Python is not available: {runtime_python}",
            field_name="device",
            suggestion="Rescan environments or install a GPU runtime.",
        ))
        return

    # Query runtime for CUDA status
    info = _query_runtime_cuda(runtime_python)
    if info is None:
        result.add(PreflightIssue(
            code="CUDA_RUNTIME_QUERY_FAILED", severity=PreflightSeverity.ERROR,
            title="Failed to query CUDA runtime",
            message=f"Could not query CUDA status from: {runtime_python}",
            field_name="device",
        ))
        return

    if not info.get("cuda_available"):
        result.add(PreflightIssue(
            code="CUDA_RUNTIME_UNAVAILABLE", severity=PreflightSeverity.ERROR,
            title="CUDA not available in training runtime",
            message=f"External runtime reports CUDA unavailable. "
                    f"torch.version.cuda={info.get('torch_cuda_version', 'N/A')}",
            field_name="device",
            suggestion="Check the external environment's CUDA installation.",
        ))
        return

    gpu_names = info.get("gpu_names", [])
    try:
        idx = int(device.split(":")[1]) if ":" in device else 0
    except (IndexError, ValueError):
        idx = 0
    if idx >= len(gpu_names):
        result.add(PreflightIssue(
            code="CUDA_INDEX_OOB", severity=PreflightSeverity.ERROR,
            title="Selected GPU index unavailable",
            message=f"GPU index {idx} not found in runtime ({len(gpu_names)} GPU(s)).",
            field_name="device",
        ))
        return

    # All checks passed
    result.add(PreflightIssue(
        code="CUDA_RUNTIME_READY", severity=PreflightSeverity.PASS,
        title=f"CUDA GPU available via external runtime: {gpu_names[idx]}",
        message=f"Runtime {runtime_id or 'external'} provides: "
                f"torch {info.get('torch_version')} · CUDA {info.get('torch_cuda_version')}",
        field_name="device",
    ))


def _query_runtime_cuda(python_path: str) -> dict | None:
    """Run CUDA diagnostic script via subprocess. Returns dict or None."""
    import json, subprocess
    script = """
import json, torch
info = {
    "torch_version": torch.__version__,
    "torch_cuda_version": getattr(torch.version, "cuda", None),
    "cuda_available": False, "gpu_count": 0, "gpu_names": [],
}
if info["torch_cuda_version"] and torch.cuda.is_available():
    info["cuda_available"] = True
    info["gpu_count"] = torch.cuda.device_count()
    for i in range(info["gpu_count"]):
        info["gpu_names"].append(torch.cuda.get_device_name(i))
print(json.dumps(info))
"""
    try:
        result = subprocess.run(
            [python_path, "-c", script],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode == 0:
            return json.loads(result.stdout.strip())
    except Exception:
        pass
    return None


# ── Helpers ─────────────────────────────────────────────────────────

def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
