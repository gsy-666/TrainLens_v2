"""Training Center Package

Unified training management layer for X-AnyLabeling.
Provides adapters for Ultralytics and Custom Script training modes.
"""

from .models import TrainingMode, TrainingStatus, TrainingJob
from .event_protocol import TrainingEventType, TrainingEvent
from .job_manager import JobManager, get_job_manager
from .adapters import TrainingAdapter, UltralyticsAdapter, CustomScriptAdapter

__all__ = [
    "TrainingMode",
    "TrainingStatus",
    "TrainingJob",
    "TrainingEventType",
    "TrainingEvent",
    "JobManager",
    "get_job_manager",
    "TrainingAdapter",
    "UltralyticsAdapter",
    "CustomScriptAdapter",
]
