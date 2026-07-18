"""TrainLens Preflight — training pre-flight checks.

Checks are performed before reserve_job / start_training to catch
configuration errors, missing data, and invalid paths early.
"""

# Models
from .models import (
    PreflightSeverity, PreflightIssue, PreflightResult,
    GuidedPreflightContext, CustomPreflightContext,
    DISK_SPACE_ERROR_BYTES, DISK_SPACE_WARNING_BYTES,
    MAX_SAMPLE_IMAGES_TRAIN, MAX_SAMPLE_IMAGES_VAL,
    MIN_IMAGE_DIMENSION, SUPPORTED_IMAGE_EXTS, SUPPORTED_MODEL_EXTS,
    CLASS_IMBALANCE_WARNING_RATIO, FEW_INSTANCES_THRESHOLD,
    EMPTY_LABEL_HIGH_RATIO,
)

# Worker
from .worker import PreflightWorker, PreflightTask

__all__ = [
    "PreflightSeverity", "PreflightIssue", "PreflightResult",
    "GuidedPreflightContext", "CustomPreflightContext",
    "DISK_SPACE_ERROR_BYTES", "DISK_SPACE_WARNING_BYTES",
    "MAX_SAMPLE_IMAGES_TRAIN", "MAX_SAMPLE_IMAGES_VAL",
    "MIN_IMAGE_DIMENSION", "SUPPORTED_IMAGE_EXTS", "SUPPORTED_MODEL_EXTS",
    "CLASS_IMBALANCE_WARNING_RATIO", "FEW_INSTANCES_THRESHOLD",
    "EMPTY_LABEL_HIGH_RATIO",
    "PreflightWorker", "PreflightTask",
]
