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
    """Check device configuration is valid."""
    if not device or not device.strip():
        result.add(PreflightIssue(
            code="DEVICE_EMPTY", severity=PreflightSeverity.WARNING,
            title="Device not specified",
            message="No device configured. Defaulting to CPU.",
            field_name="device",
        ))
        return

    device_lower = device.lower()

    # CPU is always valid
    if device_lower == "cpu":
        result.add(PreflightIssue(
            code="DEVICE_CPU_OK", severity=PreflightSeverity.PASS,
            title="Device: CPU",
            message="Training will run on CPU.",
            field_name="device",
        ))
        return

    # CUDA device check
    if device_lower.startswith("cuda") or device.isdigit():
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

        # Check device index
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
        elif device.isdigit():
            idx = int(device)
            if idx >= torch.cuda.device_count():
                result.add(PreflightIssue(
                    code="CUDA_INDEX_OOB", severity=PreflightSeverity.ERROR,
                    title="CUDA device index out of range",
                    message=f"GPU {idx} not found. Available: 0-{torch.cuda.device_count() - 1}.",
                    field_name="device",
                ))
                return

        gpu_name = torch.cuda.get_device_name(idx if 'idx' in dir() else 0)
        result.add(PreflightIssue(
            code="DEVICE_CUDA_OK", severity=PreflightSeverity.PASS,
            title=f"Device: {device} ({gpu_name})",
            message=f"CUDA is available. GPU: {gpu_name}.",
            field_name="device",
        ))
        return

    # Unknown/auto device
    result.add(PreflightIssue(
        code="DEVICE_UNKNOWN", severity=PreflightSeverity.WARNING,
        title=f"Unknown device: {device}",
        message=f"Device '{device}' may not be valid. Verify before training.",
        field_name="device",
    ))


# ── Helpers ─────────────────────────────────────────────────────────

def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
