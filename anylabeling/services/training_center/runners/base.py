"""TrainingRunner — abstract interface for all training execution backends."""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional, Tuple

from ..models import TrainingJob
from ..event_protocol import TrainingEvent


@dataclass
class RunnerCapability:
    """Describes what a runner backend supports."""
    name: str
    execution_mode: str  # "local" | "remote"
    max_workers: int = 1
    supports_gpu: bool = True
    supports_cancel: bool = True


class TrainingRunner(ABC):
    """Abstract training execution backend.

    Implementations:
      - LocalRunner  — subprocess on the same machine
      - SSHRemoteRunner — subprocess on a remote host (future)

    Each runner owns the full lifecycle of ONE training worker at a time.
    """

    def __init__(self):
        self._event_callbacks: List[Callable[[TrainingEvent], None]] = []
        self._callback_lock = threading.Lock()

    # ── metadata ──

    @property
    @abstractmethod
    def runner_id(self) -> str:
        """Unique runner instance identifier."""
        ...

    @property
    @abstractmethod
    def execution_mode(self) -> str:
        """'local' or 'remote'."""
        ...

    @property
    @abstractmethod
    def capability(self) -> RunnerCapability:
        """Static capability descriptor."""
        ...

    # ── lifecycle ──

    @abstractmethod
    def prepare(self, job: TrainingJob, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate runtime and prepare for execution.

        Called before start(). Must verify:
          - Python executable exists (GPU jobs)
          - Worker script exists
          - Config is valid for this runner

        Returns:
            (ok, message)
        """
        ...

    @abstractmethod
    def start(self, job: TrainingJob, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Launch the training worker process.

        The runner begins emitting events (worker_ready, log, ...)
        via its event callback chain. Must be non-blocking —
        the actual process runs in a background thread/QProcess.

        Returns:
            (ok, message)
        """
        ...

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Request graceful shutdown. Returns True if signal sent."""
        ...

    @abstractmethod
    def force_stop(self, job_id: str) -> bool:
        """Force-kill the worker after cancel timeout. Returns True."""
        ...

    @abstractmethod
    def is_running(self, job_id: str) -> bool:
        """Check if a worker process is active."""
        ...

    @abstractmethod
    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Return dict with pid, exit_code, runtime info, etc."""
        ...

    @abstractmethod
    def cleanup(self, job_id: str):
        """Release resources after job terminal state. Idempotent."""
        ...

    # ── events ──

    def subscribe(self, callback: Callable[[TrainingEvent], None]):
        """Register an event callback."""
        with self._callback_lock:
            if callback not in self._event_callbacks:
                self._event_callbacks.append(callback)

    def unsubscribe(self, callback: Callable[[TrainingEvent], None]):
        """Remove an event callback."""
        with self._callback_lock:
            if callback in self._event_callbacks:
                self._event_callbacks.remove(callback)

    def _emit_event(self, event: TrainingEvent):
        """Send event to all subscribers (thread-safe)."""
        with self._callback_lock:
            cbs = list(self._event_callbacks)
        for cb in cbs:
            try:
                cb(event)
            except Exception:
                pass
