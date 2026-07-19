"""Training Center - Unified Event Protocol

Defines the unified event format for all training events.
Adapters convert system-specific events to this format.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any


class TrainingEventType(Enum):
    """Unified training event types"""
    JOB_CREATED = "job_created"
    PREPARING = "preparing"
    PROCESS_STARTED = "process_started"
    WORKER_READY = "worker_ready"
    CONSOLE_OUTPUT = "console_output"
    RESOURCE_SAMPLE = "resource_sample"
    EPOCH_METRICS = "epoch_metrics"
    CHECKPOINT_SAVED = "checkpoint_saved"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class TrainingEvent:
    """Unified training event"""

    schema_version: int
    job_id: str
    event_type: TrainingEventType
    timestamp: float
    payload: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None  # "ultralytics" or "custom_script"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingEvent":
        """Deserialize from dictionary"""
        return cls(
            schema_version=data["schema_version"],
            job_id=data["job_id"],
            event_type=TrainingEventType(data["event_type"]),
            timestamp=data["timestamp"],
            payload=data.get("payload", {}),
            source=data.get("source"),
        )


def create_job_created_event(
    job_id: str,
    timestamp: float,
    mode: str,
    source: Optional[str] = None,
    **kwargs
) -> TrainingEvent:
    """Create job_created event"""
    return TrainingEvent(
        schema_version=1,
        job_id=job_id,
        event_type=TrainingEventType.JOB_CREATED,
        timestamp=timestamp,
        payload={"mode": mode, **kwargs},
        source=source,
    )


def create_process_started_event(
    job_id: str,
    timestamp: float,
    pid: Optional[int] = None,
    source: Optional[str] = None,
    **kwargs
) -> TrainingEvent:
    """Create process_started event"""
    payload = {**kwargs}
    if pid is not None:
        payload["pid"] = pid

    return TrainingEvent(
        schema_version=1,
        job_id=job_id,
        event_type=TrainingEventType.PROCESS_STARTED,
        timestamp=timestamp,
        payload=payload,
        source=source,
    )


def create_console_output_event(
    job_id: str,
    timestamp: float,
    message: str,
    stream: str = "stdout",
    source: Optional[str] = None,
) -> TrainingEvent:
    """Create console_output event"""
    return TrainingEvent(
        schema_version=1,
        job_id=job_id,
        event_type=TrainingEventType.CONSOLE_OUTPUT,
        timestamp=timestamp,
        payload={"message": message, "stream": stream},
        source=source,
    )


def create_completed_event(
    job_id: str,
    timestamp: float,
    source: Optional[str] = None,
    **kwargs
) -> TrainingEvent:
    """Create completed event"""
    return TrainingEvent(
        schema_version=1,
        job_id=job_id,
        event_type=TrainingEventType.COMPLETED,
        timestamp=timestamp,
        payload=kwargs,
        source=source,
    )


def create_failed_event(
    job_id: str,
    timestamp: float,
    error: str,
    source: Optional[str] = None,
    **kwargs
) -> TrainingEvent:
    """Create failed event"""
    return TrainingEvent(
        schema_version=1,
        job_id=job_id,
        event_type=TrainingEventType.FAILED,
        timestamp=timestamp,
        payload={"error": error, **kwargs},
        source=source,
    )


def create_stopped_event(
    job_id: str,
    timestamp: float,
    source: Optional[str] = None,
    **kwargs
) -> TrainingEvent:
    """Create stopped event"""
    return TrainingEvent(
        schema_version=1,
        job_id=job_id,
        event_type=TrainingEventType.STOPPED,
        timestamp=timestamp,
        payload=kwargs,
        source=source,
    )


def create_preparing_event(
    job_id: str,
    timestamp: float,
    source: Optional[str] = None,
    **kwargs
) -> TrainingEvent:
    """Create preparing event"""
    return TrainingEvent(
        schema_version=1,
        job_id=job_id,
        event_type=TrainingEventType.PREPARING,
        timestamp=timestamp,
        payload=kwargs,
        source=source,
    )


def create_resource_sample_event(
    job_id: str,
    timestamp: float,
    cpu_percent: Optional[float] = None,
    memory_mb: Optional[float] = None,
    gpu_percent: Optional[float] = None,
    vram_mb: Optional[float] = None,
    source: Optional[str] = None,
    **kwargs
) -> TrainingEvent:
    """Create resource_sample event"""
    payload = {**kwargs}
    if cpu_percent is not None:
        payload["cpu_percent"] = cpu_percent
    if memory_mb is not None:
        payload["memory_mb"] = memory_mb
    if gpu_percent is not None:
        payload["gpu_percent"] = gpu_percent
    if vram_mb is not None:
        payload["vram_mb"] = vram_mb

    return TrainingEvent(
        schema_version=1,
        job_id=job_id,
        event_type=TrainingEventType.RESOURCE_SAMPLE,
        timestamp=timestamp,
        payload=payload,
        source=source,
    )


def create_epoch_metrics_event(
    job_id: str,
    timestamp: float,
    epoch: int,
    metrics: Dict[str, Any],
    source: Optional[str] = None,
    **kwargs
) -> TrainingEvent:
    """Create epoch_metrics event"""
    return TrainingEvent(
        schema_version=1,
        job_id=job_id,
        event_type=TrainingEventType.EPOCH_METRICS,
        timestamp=timestamp,
        payload={"epoch": epoch, "metrics": metrics, **kwargs},
        source=source,
    )


def create_checkpoint_saved_event(
    job_id: str,
    timestamp: float,
    checkpoint_path: str,
    source: Optional[str] = None,
    **kwargs
) -> TrainingEvent:
    """Create checkpoint_saved event"""
    return TrainingEvent(
        schema_version=1,
        job_id=job_id,
        event_type=TrainingEventType.CHECKPOINT_SAVED,
        timestamp=timestamp,
        payload={"checkpoint_path": checkpoint_path, **kwargs},
        source=source,
    )
