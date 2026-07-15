"""Data models for Run Monitor"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunStatus(Enum):
    """Training run status"""
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class Run:
    """Represents a single training execution"""
    run_id: str
    workspace_path: Path
    script_path: Path
    python_path: Path
    arguments: List[str] = field(default_factory=list)
    framework: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    exit_code: Optional[int] = None
    status: RunStatus = RunStatus.IDLE
    pid: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "run_id": self.run_id,
            "workspace_path": str(self.workspace_path),
            "script_path": str(self.script_path),
            "python_path": str(self.python_path),
            "arguments": self.arguments,
            "framework": self.framework,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "exit_code": self.exit_code,
            "status": self.status.value,
            "pid": self.pid,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Run":
        """Create from dictionary"""
        return cls(
            run_id=data["run_id"],
            workspace_path=Path(data["workspace_path"]),
            script_path=Path(data["script_path"]),
            python_path=Path(data["python_path"]),
            arguments=data.get("arguments", []),
            framework=data.get("framework"),
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            exit_code=data.get("exit_code"),
            status=RunStatus(data.get("status", "idle")),
            pid=data.get("pid"),
        )


@dataclass
class DetectedScript:
    """Represents a detected training script"""
    path: Path
    framework: Optional[str] = None
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "path": str(self.path),
            "framework": self.framework,
            "confidence": self.confidence,
            "reasons": self.reasons,
        }


@dataclass
class PythonEnvironment:
    """Represents a Python installation or virtual environment"""
    python_path: Path
    version: str
    env_type: str  # system, venv, conda, uv
    env_path: Optional[Path] = None
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "python_path": str(self.python_path),
            "version": self.version,
            "env_type": self.env_type,
            "env_path": str(self.env_path) if self.env_path else None,
            "is_valid": self.is_valid,
        }


@dataclass
class Workspace:
    """Represents a training project workspace"""
    path: Path
    detected_scripts: List[DetectedScript] = field(default_factory=list)
    detected_environments: List[PythonEnvironment] = field(default_factory=list)
    scan_timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "path": str(self.path),
            "detected_scripts": [s.to_dict() for s in self.detected_scripts],
            "detected_environments": [e.to_dict() for e in self.detected_environments],
            "scan_timestamp": self.scan_timestamp.isoformat() if self.scan_timestamp else None,
        }


@dataclass
class TrainingEvent:
    """Structured event emitted during training"""
    schema_version: int = 1
    run_id: str = ""
    event: str = ""
    timestamp: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL serialization"""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "event": self.event,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingEvent":
        """Create from dictionary"""
        return cls(
            schema_version=data.get("schema_version", 1),
            run_id=data.get("run_id", ""),
            event=data.get("event", ""),
            timestamp=data.get("timestamp", 0.0),
            payload=data.get("payload", {}),
        )
