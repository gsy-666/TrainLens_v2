"""TrainLens Training Runners — unified execution layer.

Runners encapsulate all process-level concerns:
  - Python runtime selection
  - Subprocess lifecycle (start, monitor, stop, kill)
  - Event forwarding (worker_ready, log, progress, completed, failed)

JobManager interacts only with the abstract TrainingRunner interface.
"""

from .base import TrainingRunner
from .factory import RunnerFactory, get_runner

__all__ = [
    "TrainingRunner",
    "RunnerFactory",
    "get_runner",
]
