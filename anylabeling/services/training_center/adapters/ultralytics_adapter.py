"""Training Center - Ultralytics Adapter

Wraps the existing Ultralytics TrainingManager without modifying it.
Maps Ultralytics-specific events to unified TrainingEvents.
"""

import time
from typing import Callable, Dict, Any, Tuple, List

from .base import TrainingAdapter
from ..models import TrainingJob
from ..event_protocol import (
    TrainingEvent,
    create_process_started_event,
    create_console_output_event,
    create_completed_event,
    create_failed_event,
    create_stopped_event,
)


class UltralyticsAdapter(TrainingAdapter):
    """Adapter for Ultralytics TrainingManager

    Uses composition to wrap existing TrainingManager.
    Does NOT modify TrainingManager internal implementation.
    """

    def __init__(self):
        """Initialize adapter with existing TrainingManager singleton"""
        from anylabeling.services.auto_training.ultralytics.trainer import (
            get_training_manager
        )

        self.manager = get_training_manager()
        self._callbacks: List[Callable] = []
        self._current_job_id: str = None
        self._original_callbacks = self.manager.callbacks.copy()
        self._registered_callback = self._on_training_event

        # Register our event mapper
        self.manager.callbacks.append(self._registered_callback)

    def can_start(self) -> Tuple[bool, str]:
        """Check if Ultralytics training can start"""
        if self.manager.is_training:
            return False, "Ultralytics training already in progress"
        return True, ""

    def start(self, job: TrainingJob, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Start Ultralytics training

        Args:
            job: TrainingJob with metadata (runtime_python, etc.)
            config: train_args dict expected by TrainingManager.start_training()

        Returns:
            (success, message)
        """
        if not isinstance(config, dict):
            return False, "Config must be a dict of train_args"

        self._current_job_id = job.job_id
        self._current_runtime_python = getattr(job, 'runtime_python', None) or None

        python_executable = self._current_runtime_python

        # Diagnostic: log the runtime being used
        if python_executable:
            _log = __import__('logging').getLogger(__name__)
            _log.info(
                "Adapter received runtime_python: %s (job_id=%s, device=%s)",
                python_executable, job.job_id, getattr(job, 'requested_device', '?'),
            )

        success, message = self.manager.start_training(config, python_executable=python_executable)
        return success, message

    def stop(self) -> bool:
        """Stop Ultralytics training"""
        return self.manager.stop_training()

    def is_running(self) -> bool:
        """Check if Ultralytics training is running"""
        return self.manager.is_training

    def subscribe(self, callback: Callable) -> None:
        """Subscribe to unified training events"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from training events"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _on_training_event(self, event_type: str, data: dict):
        """Map Ultralytics events to unified TrainingEvents

        Ultralytics event types:
        - training_started: {"total_epochs": int}
        - training_log: {"message": str}
        - training_completed: {"results": str}
        - training_error: {"error": str}
        - training_stopped: {}
        """
        if not self._current_job_id:
            return

        timestamp = time.time()
        unified_event = None

        if event_type == "training_started":
            unified_event = create_process_started_event(
                job_id=self._current_job_id,
                timestamp=timestamp,
                source="ultralytics",
                total_epochs=data.get("total_epochs"),
            )

        elif event_type == "training_log":
            unified_event = create_console_output_event(
                job_id=self._current_job_id,
                timestamp=timestamp,
                message=data.get("message", ""),
                stream="stdout",
                source="ultralytics",
            )

        elif event_type == "training_completed":
            unified_event = create_completed_event(
                job_id=self._current_job_id,
                timestamp=timestamp,
                source="ultralytics",
                results=data.get("results"),
                save_dir=data.get("save_dir", ""),
            )

        elif event_type == "training_error":
            unified_event = create_failed_event(
                job_id=self._current_job_id,
                timestamp=timestamp,
                error=data.get("error", "Unknown error"),
                source="ultralytics",
                save_dir=data.get("save_dir", ""),
            )

        elif event_type == "training_stopped":
            unified_event = create_stopped_event(
                job_id=self._current_job_id,
                timestamp=timestamp,
                source="ultralytics",
                save_dir=data.get("save_dir", ""),
            )

        elif event_type == "worker_ready":
            # Emit runtime info as console output
            worker_info_msg = (
                f"Runtime Python:\n{data.get('sys_executable', '?')}\n\n"
                f"Python:\n{data.get('python_version', '?')}\n\n"
                f"Torch:\n{data.get('torch_version', '?')}\n\n"
                f"Torch CUDA:\n{data.get('torch_cuda_version', '?')}\n\n"
                f"CUDA available:\n{data.get('cuda_available')}\n\n"
                f"GPU:\n{data.get('gpu_name', 'N/A')}"
            )
            console_event = create_console_output_event(
                job_id=self._current_job_id,
                timestamp=timestamp,
                message=worker_info_msg,
                stream="stdout",
                source="ultralytics",
            )
            self._emit_event(console_event)

            # Also emit a dedicated worker_ready event for UI state transitions
            unified_event = TrainingEvent(
                schema_version=1,
                job_id=self._current_job_id,
                event_type=TrainingEventType.WORKER_READY,
                timestamp=timestamp,
                payload={
                    "sys_executable": data.get("sys_executable", ""),
                    "python_version": data.get("python_version", ""),
                    "torch_version": data.get("torch_version", ""),
                    "torch_cuda_version": data.get("torch_cuda_version"),
                    "cuda_available": data.get("cuda_available", False),
                    "gpu_count": data.get("gpu_count", 0),
                    "gpu_name": data.get("gpu_name", ""),
                    "gpu_names": data.get("gpu_names", []),
                    "requested_device": data.get("requested_device", ""),
                    "ultralytics_device": data.get("ultralytics_device", ""),
                },
                source="ultralytics",
            )

        if unified_event:
            self._emit_event(unified_event)

    def _emit_event(self, event: TrainingEvent):
        """Emit unified event to all subscribers"""
        for callback in self._callbacks[:]:  # Copy to avoid modification during iteration
            try:
                callback(event)
            except Exception:
                # Silently ignore callback exceptions to prevent breaking other callbacks
                pass

    def shutdown(self):
        """Clean up: remove registered callback from TrainingManager.

        Must be called explicitly — do NOT rely on __del__.
        Idempotent: safe to call multiple times.
        """
        if hasattr(self, 'manager') and hasattr(self, '_registered_callback'):
            try:
                if self._registered_callback in self.manager.callbacks:
                    self.manager.callbacks.remove(self._registered_callback)
            except (ValueError, AttributeError):
                pass
        self._callbacks.clear()

    def __del__(self):
        """Fallback cleanup — prefer explicit shutdown()."""
        self.shutdown()
