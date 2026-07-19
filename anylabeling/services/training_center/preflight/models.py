"""TrainLens Preflight — data-only models (no Qt dependency)."""

import hashlib
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class PreflightSeverity(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class PreflightIssue:
    """A single preflight finding."""
    code: str
    severity: PreflightSeverity
    title: str
    message: str
    details: str = ""
    field_name: Optional[str] = None
    path: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code, "severity": self.severity.value,
            "title": self.title, "message": self.message,
            "details": self.details, "field_name": self.field_name,
            "path": self.path, "suggestion": self.suggestion,
        }


@dataclass
class PreflightResult:
    """Aggregated preflight check results."""
    mode: str
    issues: List[PreflightIssue] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    config_fingerprint: str = ""
    cancelled: bool = False

    @property
    def pass_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == PreflightSeverity.PASS)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == PreflightSeverity.WARNING)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == PreflightSeverity.ERROR)

    @property
    def can_start(self) -> bool:
        return self.error_count == 0 and not self.cancelled

    @property
    def has_warnings(self) -> bool:
        return self.warning_count > 0

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def add(self, issue: PreflightIssue):
        self.issues.append(issue)

    def errors(self) -> List[PreflightIssue]:
        return [i for i in self.issues if i.severity == PreflightSeverity.ERROR]

    def warnings(self) -> List[PreflightIssue]:
        return [i for i in self.issues if i.severity == PreflightSeverity.WARNING]

    def summary(self) -> str:
        label = "Data Check" if self.mode == "data_check" else "Preflight"
        return (
            f"{label}: {self.pass_count} passed, "
            f"{self.warning_count} warnings, {self.error_count} errors"
        )


@dataclass
class GuidedPreflightContext:
    """Immutable snapshot of Guided training config for preflight."""
    task_type: str = ""
    model_path: str = ""
    dataset_yaml: str = ""
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    device: str = "cpu"
    output_dir: str = ""
    job_name: str = ""
    classes: str = ""

    # Runtime (external GPU environment)
    runtime_id: str = ""
    runtime_python: str = ""
    requested_device: str = ""
    resolved_device: str = ""
    execution_mode: str = ""
    runtime_type: str = ""
    device_name: str = ""

    def fingerprint(self) -> str:
        parts = [
            self.task_type, self.model_path, self.dataset_yaml,
            str(self.epochs), str(self.batch), str(self.imgsz),
            self.device, self.output_dir, self.job_name, self.classes,
        ]
        for p in (self.model_path, self.dataset_yaml):
            if p and os.path.isfile(p):
                try:
                    st = os.stat(p)
                    parts.append(f"{p}:{st.st_mtime}:{st.st_size}")
                except OSError:
                    parts.append(f"{p}:missing")
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class CustomPreflightContext:
    """Immutable snapshot of Custom Project config for preflight."""
    project_dir: str = ""
    python_path: str = ""
    script_path: str = ""
    arguments: str = ""
    working_directory: str = ""
    output_dir: str = ""
    environment_busy: bool = False

    def fingerprint(self) -> str:
        parts = [
            self.project_dir, self.python_path, self.script_path,
            self.arguments, self.working_directory, self.output_dir,
            str(self.environment_busy),
        ]
        for p in (self.python_path, self.script_path):
            if p and os.path.isfile(p):
                try:
                    st = os.stat(p)
                    parts.append(f"{p}:{st.st_mtime}:{st.st_size}")
                except OSError:
                    parts.append(f"{p}:missing")
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Constants ───────────────────────────────────────────────────────

DISK_SPACE_ERROR_BYTES = 1 * 1024 * 1024 * 1024
DISK_SPACE_WARNING_BYTES = 5 * 1024 * 1024 * 1024
MAX_SAMPLE_IMAGES_TRAIN = 100
MAX_SAMPLE_IMAGES_VAL = 100
MIN_IMAGE_DIMENSION = 16
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SUPPORTED_MODEL_EXTS = {".pt", ".pth", ".onnx", ".engine", ".mlmodel"}
CLASS_IMBALANCE_WARNING_RATIO = 20
FEW_INSTANCES_THRESHOLD = 5
EMPTY_LABEL_HIGH_RATIO = 0.5
