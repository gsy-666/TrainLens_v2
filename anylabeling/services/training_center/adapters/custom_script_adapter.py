"""Training Center - Custom Script Adapter

Wraps the existing Run Monitor ProcessManager without modifying it.
Maps ProcessManager signals to unified TrainingEvents.
"""

import time
import os
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
        self._error_buffer: List[str] = []

        # Connect to ProcessManager signals
        self.manager.process_started.connect(self._on_process_started)
        self.manager.process_finished.connect(self._on_process_finished)
        self.manager.stdout_ready.connect(self._on_stdout)
        self.manager.stderr_ready.connect(self._on_stderr_buffered)

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

        workspace_path = job.workspace or Path.cwd()
        script_path_obj = Path(script_path)
        python_path = job.python_executable or Path('python')

        # Clear error buffer before starting
        self._error_buffer.clear()

        # Pre-validation: check if paths exist and are valid
        diagnostics = []
        diagnostics.append(f"Python executable: {python_path}")
        diagnostics.append(f"Script path: {script_path_obj}")
        diagnostics.append(f"Working directory: {workspace_path}")
        diagnostics.append(f"Arguments: {arguments}")

        # Build command for diagnostics
        cmd_parts = [str(python_path), str(script_path_obj)] + (arguments if isinstance(arguments, list) else [])
        diagnostics.append(f"Command: {' '.join(cmd_parts)}")

        # Validate python executable exists
        if not os.path.exists(python_path):
            error_msg = f"Python executable not found\n" + "\n".join(diagnostics)
            return False, error_msg

        # Validate script exists
        if not script_path_obj.exists():
            error_msg = f"Script file not found\n" + "\n".join(diagnostics)
            return False, error_msg

        # Validate workspace exists
        if not workspace_path.exists():
            error_msg = f"Workspace directory not found\n" + "\n".join(diagnostics)
            return False, error_msg

        run = Run(
            run_id=job.job_id,
            workspace_path=workspace_path,
            script_path=script_path_obj,
            python_path=python_path,
            arguments=arguments if isinstance(arguments, list) else [],
        )

        self._current_job_id = job.job_id
        self._current_job = job

        success = self.manager.start(run)
        if success:
            return True, "Process started"
        else:
            # Collect buffered errors from ProcessManager
            if self._error_buffer:
                error_details = "\n".join(self._error_buffer)
                full_msg = f"Failed to start process\n{error_details}\n" + "\n".join(diagnostics)
                self._error_buffer.clear()
                return False, full_msg
            else:
                error_msg = f"Failed to start process (no error details)\n" + "\n".join(diagnostics)
                return False, error_msg

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
        """Map ProcessManager process_finished to completed/failed/stopped event"""
        if not self._current_job_id:
            return

        timestamp = time.time()

        # Check if user requested stop
        if self.manager._stop_requested:
            # User stopped the process - emit STOPPED regardless of exit code
            event = create_stopped_event(
                job_id=self._current_job_id,
                timestamp=timestamp,
                source="custom_script",
            )
        elif exit_code == 0:
            # Normal completion
            event = create_completed_event(
                job_id=self._current_job_id,
                timestamp=timestamp,
                source="custom_script",
                exit_code=exit_code,
            )
        else:
            # Process failed with non-zero exit
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

    def _on_stderr_buffered(self, line: str):
        """Buffer stderr during start, then forward to event system"""
        # Buffer errors for start() method to use
        self._error_buffer.append(line)

        # Also forward to event system if job is running
        self._on_stderr(line)

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
