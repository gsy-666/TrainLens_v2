"""Training Center - Unified Domain Models

This module defines the unified domain models for the Training Center.
All training jobs (Ultralytics and Custom Script) are represented using these models.
Adapters map from system-specific formats to these unified models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any


class TrainingMode(Enum):
    """Training execution mode"""
    GUIDED_ULTRALYTICS = "guided_ultralytics"
    CUSTOM_SCRIPT = "custom_script"


class TrainingStatus(Enum):
    """Unified training job status"""
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

    def is_terminal(self) -> bool:
        """Check if status is terminal (job finished)"""
        return self in (
            TrainingStatus.COMPLETED,
            TrainingStatus.FAILED,
            TrainingStatus.STOPPED,
        )

    def is_active(self) -> bool:
        """Check if status is active (job in progress)"""
        return self in (
            TrainingStatus.PREPARING,
            TrainingStatus.RUNNING,
            TrainingStatus.STOPPING,
        )


# ── Execution mode normalization ────────────────────────────────────

_EXECUTION_MODE_ALIASES = {
    "local": "local",
    "remote_ssh": "remote_ssh",
    # Legacy aliases (stripped on normalization, never persisted)
    "remote": "remote_ssh",
    "ssh_remote": "remote_ssh",
}


def normalize_execution_mode(value) -> str:
    """Convert any execution mode representation to canonical value.

    Only valid canonical values: 'local', 'remote_ssh'.
    Raises ValueError for unknown values.
    NEVER returns LocalRunner as a default.
    """
    if value is None:
        return "local"
    raw = str(value).strip().lower()
    if raw not in _EXECUTION_MODE_ALIASES:
        raise ValueError(
            f"Unsupported execution mode: {value!r}. "
            f"Valid values: {list(_EXECUTION_MODE_ALIASES.keys())}"
        )
    return _EXECUTION_MODE_ALIASES[raw]


@dataclass
class TrainingJob:
    """Unified training job representation"""

    job_id: str
    mode: TrainingMode
    status: TrainingStatus = TrainingStatus.IDLE

    # Timestamps
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Paths
    workspace: Optional[Path] = None
    output_directory: Optional[Path] = None

    # Display
    display_name: Optional[str] = None
    framework: Optional[str] = None

    # Execution
    python_executable: Optional[Path] = None
    command: Optional[list] = None

    # Runtime (GPU environment)
    runtime_id: Optional[str] = None
    runtime_python: Optional[str] = None
    requested_device: Optional[str] = None
    resolved_device: Optional[str] = None
    execution_mode: Optional[str] = None  # "local" | "remote"

    # Remote SSH
    remote_profile_id: Optional[str] = None
    remote_host: Optional[str] = None
    remote_port: Optional[int] = None
    remote_username: Optional[str] = None
    remote_workspace: Optional[str] = None
    remote_python: Optional[str] = None
    remote_job_dir: Optional[str] = None
    remote_output_directory: Optional[str] = None

    # Session-only (not persisted)
    _session_password: Optional[str] = None

    # Training config (for Guided history display)
    task: Optional[str] = None
    model: Optional[str] = None
    data: Optional[str] = None
    project: Optional[str] = None
    name: Optional[str] = None

    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "job_id": self.job_id,
            "mode": self.mode.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "workspace": str(self.workspace) if self.workspace else None,
            "output_directory": str(self.output_directory) if self.output_directory else None,
            "display_name": self.display_name,
            "framework": self.framework,
            "python_executable": str(self.python_executable) if self.python_executable else None,
            "command": self.command,
            "metadata": self.metadata,
            "error_message": self.error_message,
            "runtime_id": self.runtime_id,
            "runtime_python": self.runtime_python,
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "execution_mode": self.execution_mode,
            "remote_profile_id": self.remote_profile_id,
            "remote_host": self.remote_host,
            "remote_port": self.remote_port,
            "remote_workspace": self.remote_workspace,
            "remote_python": self.remote_python,
            "remote_output_directory": self.remote_output_directory,
        }

    @property
    def python_exe_path(self) -> Optional[Path]:
        """Resolved Python executable for training subprocess."""
        if self.runtime_python:
            return Path(self.runtime_python)
        if self.python_executable:
            return self.python_executable
        return None
