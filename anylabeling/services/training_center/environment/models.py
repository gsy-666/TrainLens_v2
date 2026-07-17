"""TrainLens Environment Wizard — domain models."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class EnvironmentStatus(str, Enum):
    NOT_CHECKED = "not_checked"
    CHECKING = "checking"
    NOT_FOUND = "not_found"
    CREATING = "creating"
    INSTALLING = "installing"
    READY = "ready"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class EnvironmentInfo:
    """Snapshot of a Python environment."""

    python_path: str = ""
    python_version: str = ""
    executable_valid: bool = False
    pip_available: bool = False
    environment_type: str = ""  # venv, conda, system
    torch_installed: bool = False
    torch_version: str = ""
    cuda_available: bool = False
    cuda_version: str = ""
    gpu_name: str = ""
    ultralytics_installed: bool = False
    ultralytics_version: str = ""
    requirements_path: str = ""
    status: EnvironmentStatus = EnvironmentStatus.NOT_CHECKED
    warning_message: str = ""
    error_message: str = ""

    @property
    def summary(self) -> str:
        lines = []
        if self.python_version:
            lines.append(f"Python {self.python_version}")
        if self.environment_type:
            lines.append(f"Type: {self.environment_type}")
        if self.executable_valid:
            lines.append("pip: Available" if self.pip_available else "pip: MISSING")
        if self.torch_installed:
            line = f"PyTorch {self.torch_version}"
            if self.cuda_available:
                line += f" · CUDA {self.cuda_version}"
                if self.gpu_name:
                    line += f" · GPU {self.gpu_name}"
            lines.append(line)
        if self.ultralytics_installed:
            lines.append(f"Ultralytics {self.ultralytics_version}")
        return "\n".join(lines)
