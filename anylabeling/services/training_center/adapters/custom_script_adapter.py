"""Training Center - Custom Script Adapter

Wraps the existing Run Monitor ProcessManager without modifying it.
Maps ProcessManager signals to unified TrainingEvents.
"""

import time
from typing import Callable, Dict, Any, Tuple, List
from pathlib import Path

from PyQt6.QtCore import QObject

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


class CustomScriptAdapter(TrainingAdapter):
    """Adapter for Run Monitor ProcessManager

    Uses composition to wrap existing ProcessManager.
    Does NOT modify ProcessManager internal implementation.
    """

    def __init__(self):
        """Initialize adapter with new ProcessManager instance"""
        from anylabeling.services.run_monitor.process_manager import ProcessManager

        self.manager = ProcessManager()
        self._callbacks: List[Callable] = []
        self._current_job_id: str = None
        self._current_job: TrainingJob = None

        # Connect to ProcessManager signals
        self.manager.process_started.connect(self._on_process_started)
        self.manager.process_finished.connect(self._on_process_finished)
        self.manager.stdout_ready.connect(self._on_stdout)
        self.manager.stderr_ready.connect(self._on_stderr)

    def can_start(self) -> Tuple[bool, str]:
        """Check if custom script can start"""
        if self.manager.is_running():
            return False, "Custom script already in progress"
        return True, ""

    def start(self, job: TrainingJob, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Start custom script training

        Args:
            job: TrainingJob with workspace, python_executable, command
            config: Dict with 'script_path', 'arguments' (optional)

        Returns:
            (success, message)
        """
        if not isinstance(config, dict):
            return False, "Config must be a dict"

        script_path = config.get('script_path')
        if not script_path:
            return False, "Config must include 'script_path'"

        arguments = config.get('arguments', [])

        # Create Run object for ProcessManager
        from anylabeling.services.run_monitor.models import Run

        run = Run(
            run_id=job.job_id,
            workspace_path=job.workspace or Path.cwd(),
            script_path=Path(script_path),
            python_path=job.python_executable or Path('python'),
            arguments=arguments if isinstance(arguments, list) else [],
        )

        self._current_job_id = job.job_id
        self._current_job = job

        success = self.manager.start(run)
        if success:
            return True, "Process started"
        else:
            return False, "Failed to start process"

    def stop(self) -> bool:
        """Stop custom script training"""
        return self.manager.stop()

    def is_running(self) -> bool:
        """Check if custom script is running"""
        return self.manager.is_running()

    def subscribe(self, callback: Callable) -> None:
        """Subscribe to unified training events"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from training events"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _on_process_started(self, pid: int):
        """Map ProcessManager process_started to unified event"""
        if not self._current_job_id:
            return

        event = create_process_started_event(
            job_id=self._current_job_id,
            timestamp=time.time(),
            pid=pid,
            source="custom_script",
        )
        self._emit_event(event)

    def _on_process_finished(self, pid: int, exit_code: int):
        """Map ProcessManager process_finished to completed/failed event"""
        if not self._current_job_id:
            return

        timestamp = time.time()

        if exit_code == 0:
            event = create_completed_event(
                job_id=self._current_job_id,
                timestamp=timestamp,
                source="custom_script",
                exit_code=exit_code,
            )
        else:
            event = create_failed_event(
                job_id=self._current_job_id,
                timestamp=timestamp,
                error=f"Process exited with code {exit_code}",
                source="custom_script",
                exit_code=exit_code,
            )

        self._emit_event(event)

    def _on_stdout(self, line: str):
        """Map stdout to console_output event"""
        if not self._current_job_id:
            return

        event = create_console_output_event(
            job_id=self._current_job_id,
            timestamp=time.time(),
            message=line,
            stream="stdout",
            source="custom_script",
        )
        self._emit_event(event)

    def _on_stderr(self, line: str):
        """Map stderr to console_output event"""
        if not self._current_job_id:
            return

        event = create_console_output_event(
            job_id=self._current_job_id,
            timestamp=time.time(),
            message=line,
            stream="stderr",
            source="custom_script",
        )
        self._emit_event(event)

    def _emit_event(self, event: TrainingEvent):
        """Emit unified event to all subscribers"""
        for callback in self._callbacks[:]:
            try:
                callback(event)
            except Exception:
                pass
