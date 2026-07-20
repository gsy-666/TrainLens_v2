"""Remote training profile and diagnostic models."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List


class AuthMethod(str, Enum):
    SSH_KEY = "ssh_key"
    PASSWORD = "password"


class DiagnosticStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    PENDING = "PENDING"


@dataclass
class DiagnosticItem:
    """A single diagnostic check result."""
    stage: str  # e.g. "connection", "python", "pytorch", "gpu"
    label: str  # e.g. "SSH Connection", "Python Version"
    status: DiagnosticStatus = DiagnosticStatus.PENDING
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RemoteProfile:
    """SSH remote training host profile.

    Passwords are NEVER persisted to disk. password is only held
    in memory during the current session.
    """

    profile_id: str
    name: str = ""

    # Connection
    host: str = ""
    port: int = 22
    username: str = ""

    # Auth
    auth_method: AuthMethod = AuthMethod.SSH_KEY
    private_key_path: str = ""  # SSH key file path
    password: str = ""  # NEVER persisted; session-only

    # Remote environment
    remote_workspace: str = ""
    remote_python: str = ""

    # Security
    known_host_fingerprint: str = ""

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict. Password is excluded."""
        d = asdict(self)
        d.pop("password", None)
        d["auth_method"] = self.auth_method.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RemoteProfile":
        """Deserialize from dict. Password always empty on load."""
        data = dict(data)
        data["auth_method"] = AuthMethod(data.get("auth_method", "ssh_key"))
        data.pop("password", None)
        return cls(**data)

    def to_json_safe(self) -> Dict[str, Any]:
        """Override to handle dataclass serialization edge cases."""
        return self.to_dict()
