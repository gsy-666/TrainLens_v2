"""LocalRunner — QProcess-based training execution on local machine."""

import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback as _traceback
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Tuple

from PyQt6.QtCore import (
    QObject, QProcess, QProcessEnvironment, pyqtSignal, QTimer,
)

from .base import TrainingRunner, RunnerCapability
from ..models import TrainingJob, TrainingStatus
from ..event_protocol import (
    TrainingEvent,
    TrainingEventType,
    create_process_started_event,
    create_console_output_event,
    create_completed_event,
    create_failed_event,
    create_stopped_event,
)

_log = logging.getLogger(__name__)

TRAINING_WORKER_EVENT_PREFIX = "__XANYLABELING_TRAIN_EVENT__="


# ── QProcess bridge (must live on main thread) ────────────────────────

class _LocalProcessBridge(QObject):
    """Thin QObject that owns a QProcess for the training worker.

    Lives on the GUI thread so QProcess signals work correctly.
    All actual logic lives in LocalRunner which communicates via this bridge.
    """

    # Signals (emitted from any thread via pyqtSignal)
    stdout_line = pyqtSignal(str)
    process_done = pyqtSignal(int)  # exit_code
    process_error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: Optional[QProcess] = None
        self._running_job_id: str = ""
        self._partial_line = ""

    def launch(self, job_id: str, command: List[str], env: Dict[str, str] = None) -> bool:
        """Start QProcess with the given command.

        Returns True on success, False on failure (no exception raised).
        """
        if self._proc is not None:
            _log.warning("Process still running; cleaning up first")
            self.cleanup()

        self._running_job_id = job_id
        self._partial_line = ""

        try:
            self._proc = QProcess(self)
            self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

            # Build process environment (system + overrides)
            process_env = QProcessEnvironment.systemEnvironment()
            if env:
                for key, value in env.items():
                    if value is not None:
                        process_env.insert(str(key), str(value))

            self._proc.setProcessEnvironment(process_env)

            # Wire signals
            self._proc.readyReadStandardOutput.connect(self._on_stdout)
            self._proc.finished.connect(self._on_finished)
            self._proc.errorOccurred.connect(self._on_error)

            # Set program and arguments explicitly
            program = command[0]
            args = command[1:] if len(command) > 1 else []
            self._proc.setProgram(program)
            self._proc.setArguments(args)

            # Set working directory to worker script's parent (not cwd-dependent)
            worker_dir = os.path.dirname(os.path.abspath(command[1])) if len(command) > 1 else ""
            if worker_dir and os.path.isdir(worker_dir):
                self._proc.setWorkingDirectory(worker_dir)

            _log.info("LocalRunner: QProcess program=%s args=%s wd=%s", program, args[:3], worker_dir)
            self._proc.start()
            return True

        except (AttributeError, OSError, RuntimeError) as e:
            _log.error("Failed to launch QProcess: %s\n%s", e, _traceback.format_exc())
            self.cleanup()
            self.process_error.emit(f"Failed to start local training worker: {e}")
            return False

    def _on_stdout(self):
        """Read stdout and emit full lines."""
        if not self._proc:
            return
        data = self._proc.readAllStandardOutput()
        if not data:
            return
        try:
            text = data.data().decode("utf-8", errors="replace")
        except Exception:
            text = str(data)

        self._partial_line += text
        while "\n" in self._partial_line:
            line, self._partial_line = self._partial_line.split("\n", 1)
            line = line.strip()
            if line:
                self.stdout_line.emit(line)

    def _on_finished(self, exit_code: int, _exit_status):
        """Flush remaining data and emit done."""
        # Flush remaining partial line
        if self._partial_line.strip():
            self.stdout_line.emit(self._partial_line.strip())
            self._partial_line = ""
        self.process_done.emit(exit_code)

    def _on_error(self, error: QProcess.ProcessError):
        err_msg = {
            QProcess.ProcessError.FailedToStart: "Failed to start",
            QProcess.ProcessError.Crashed: "Crashed",
            QProcess.ProcessError.Timedout: "Timed out",
            QProcess.ProcessError.WriteError: "Write error",
            QProcess.ProcessError.ReadError: "Read error",
            QProcess.ProcessError.UnknownError: "Unknown error",
        }.get(error, f"Process error: {error}")
        self.process_error.emit(err_msg)

    def terminate(self):
        """Send SIGTERM / WM_CLOSE."""
        if self._proc and self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.terminate()

    def kill(self):
        """Force kill."""
        if self._proc and self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.kill()

    def pid(self) -> int:
        """Return OS process ID or 0."""
        if self._proc:
            pid = self._proc.processId()
            if pid:
                return pid
        return 0

    def state(self):
        """Return QProcess state."""
        if self._proc:
            return self._proc.state()
        return QProcess.ProcessState.NotRunning

    def cleanup(self):
        """Disconnect and release QProcess."""
        if self._proc:
            try:
                self._proc.readyReadStandardOutput.disconnect(self._on_stdout)
                self._proc.finished.disconnect(self._on_finished)
                self._proc.errorOccurred.disconnect(self._on_error)
            except (TypeError, RuntimeError):
                pass
            self._proc.deleteLater()
            self._proc = None
        self._running_job_id = ""
        self._partial_line = ""


# ── LocalRunner ───────────────────────────────────────────────────────

class LocalRunner(TrainingRunner):
    """Runs training as a local subprocess via QProcess.

    Responsibilities:
      - Python runtime selection (runtime_python or sys.executable)
      - Worker script resolution
      - QProcess lifecycle (start → monitor → stop → kill)
      - stdout parsing → training events
      - worker_ready validation
      - CUDA mismatch handling
      - cancel / force_stop / cleanup
    """

    _runner_counter = 0

    def __init__(self, parent: QObject = None):
        super().__init__()
        LocalRunner._runner_counter += 1
        self._runner_id = f"local-{LocalRunner._runner_counter}"
        self._bridge = _LocalProcessBridge(parent)
        self._job: Optional[TrainingJob] = None
        self._active_job_id: str = ""
        self._config: Dict[str, Any] = {}

        # Wire bridge signals
        self._bridge.stdout_line.connect(self._on_worker_line)
        self._bridge.process_done.connect(self._on_process_done)
        self._bridge.process_error.connect(self._on_bridge_error)

        # Cancel state
        self._stop_requested = False
        self._cancel_timer: Optional[QTimer] = None
        self._terminal_event_sent = False

    # ── metadata ──

    @property
    def runner_id(self) -> str:
        return self._runner_id

    @property
    def execution_mode(self) -> str:
        return "local"

    @property
    def capability(self) -> RunnerCapability:
        return RunnerCapability(
            name="LocalRunner",
            execution_mode="local",
            max_workers=1,
            supports_gpu=True,
            supports_cancel=True,
        )

    # ── lifecycle ──

    def prepare(self, job: TrainingJob, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate runtime and config before starting."""
        # GPU validation
        requested = getattr(job, "requested_device", "") or ""
        if requested.startswith("cuda") or (
            getattr(job, "resolved_device", "") not in ("cpu", "", None)
        ):
            rp = getattr(job, "runtime_python", None) or ""
            if not rp:
                return False, (
                    f"GPU runtime Python is missing for device '{requested}'. "
                    f"Select a GPU device with a registered CUDA environment."
                )
            if not os.path.isfile(rp):
                return False, f"Runtime Python not found: {rp}"

        # Worker script check
        worker_path = _resolve_worker_script()
        from ..resource_utils import is_frozen
        if not is_frozen():
            if not os.path.isfile(worker_path):
                return False, f"Worker script not found: {worker_path}"

        return True, "ready"

    def start(self, job: TrainingJob, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Launch the training worker.

        Catches all exceptions to prevent crashing the GUI main loop.
        """
        if self._active_job_id:
            return False, "Runner already has an active job"

        self._job = job
        self._active_job_id = job.job_id
        self._config = dict(config)
        self._stop_requested = False
        self._terminal_event_sent = False

        python_exe = _resolve_python_executable(job)

        try:
            # Build payload
            payload_path = _create_training_payload(config)

            # Build command
            worker_script = _resolve_worker_script()
            from ..resource_utils import is_frozen
            if is_frozen() and python_exe == sys.executable:
                # Frozen + bundled CPU: invoke same EXE in training-worker mode
                # Works for both PyInstaller and Nuitka standalone
                command = [sys.executable, "--training-worker", "--payload", payload_path]
            else:
                # Source mode OR external GPU runtime
                command = [python_exe, worker_script, "--payload", payload_path]

            _log.info(
                "Runner: %s\nExecution mode: %s\nRuntime Python: %s\nWorker script: %s\n"
                "Requested device: %s\nResolved device: %s\nFrozen: %s",
                self.runner_id, self.execution_mode, python_exe, worker_script,
                getattr(job, "requested_device", ""),
                getattr(job, "resolved_device", ""),
                is_frozen(),
            )

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            ok = self._bridge.launch(job.job_id, command, env)
            if not ok:
                self._active_job_id = ""
                self._job = None
                return False, "Failed to start local training worker (see log for details)"

            return True, "Worker process started"

        except Exception as e:
            _log.error("LocalRunner.start exception: %s\n%s", e, _traceback.format_exc())
            self._active_job_id = ""
            self._job = None
            return False, f"Failed to start local training worker: {e}"

    def cancel(self, job_id: str) -> bool:
        """Request graceful shutdown."""
        if job_id != self._active_job_id:
            return False
        self._stop_requested = True

        # Phase 1: terminate
        self._bridge.terminate()

        # Phase 2: force-kill after timeout (3s)
        if self._cancel_timer:
            self._cancel_timer.stop()
        from PyQt6.QtCore import QTimer
        self._cancel_timer = QTimer()
        self._cancel_timer.setSingleShot(True)
        self._cancel_timer.timeout.connect(lambda: self.force_stop(job_id))
        self._cancel_timer.start(3000)

        return True

    def force_stop(self, job_id: str) -> bool:
        """Force-kill the worker process."""
        if job_id != self._active_job_id:
            return False
        self._bridge.kill()

        if not self._terminal_event_sent:
            self._emit_event(create_stopped_event(
                job_id=job_id, timestamp=time.time(),
                source="local_runner",
            ))
            self._terminal_event_sent = True

        return True

    def is_running(self, job_id: str) -> bool:
        if job_id != self._active_job_id:
            return False
        state = self._bridge.state()
        return state != QProcess.ProcessState.NotRunning

    def get_status(self, job_id: str) -> Dict[str, Any]:
        return {
            "active_job_id": self._active_job_id,
            "pid": self._bridge.pid(),
            "is_running": self.is_running(job_id),
            "stop_requested": self._stop_requested,
        }

    def cleanup(self, job_id: str):
        """Release resources. Idempotent."""
        if job_id != self._active_job_id:
            return
        if self._cancel_timer:
            self._cancel_timer.stop()
            self._cancel_timer = None
        self._bridge.cleanup()
        self._active_job_id = ""
        self._job = None
        self._config = {}
        self._stop_requested = False

    # ── stdout parsing ─────────────────────────────────────────────────

    def _on_worker_line(self, line: str):
        """Parse a single stdout line from the worker."""
        if not line:
            return
        if not self._active_job_id:
            return

        if line.startswith(TRAINING_WORKER_EVENT_PREFIX):
            payload_text = line[len(TRAINING_WORKER_EVENT_PREFIX):]
            try:
                data = json.loads(payload_text)
            except json.JSONDecodeError:
                self._emit_event(create_console_output_event(
                    job_id=self._active_job_id, timestamp=time.time(),
                    message=line, source="local_runner",
                ))
                return
            self._handle_worker_event(data)
        else:
            # Plain log line
            self._emit_event(create_console_output_event(
                job_id=self._active_job_id, timestamp=time.time(),
                message=line, stream="stdout", source="local_runner",
            ))

    def _handle_worker_event(self, data: Dict[str, Any]):
        """Process a structured worker event."""
        event_type = data.pop("event", "")
        job_id = self._active_job_id
        ts = time.time()

        if event_type == "worker_ready":
            self._on_worker_ready(job_id, ts, data)
        elif event_type == "training_log":
            self._emit_event(create_console_output_event(
                job_id=job_id, timestamp=ts,
                message=data.get("message", ""),
                stream="stdout", source="local_runner",
            ))
        elif event_type == "training_completed":
            if not self._terminal_event_sent:
                self._emit_event(create_completed_event(
                    job_id=job_id, timestamp=ts,
                    source="local_runner",
                    results=data.get("results"),
                    save_dir=data.get("save_dir", ""),
                ))
                self._terminal_event_sent = True
        elif event_type == "training_error":
            if not self._terminal_event_sent:
                self._emit_event(create_failed_event(
                    job_id=job_id, timestamp=ts,
                    error=data.get("error", "Unknown error"),
                    source="local_runner",
                    save_dir=data.get("save_dir", ""),
                ))
                self._terminal_event_sent = True
        elif event_type == "training_stopped":
            if not self._terminal_event_sent:
                self._emit_event(create_stopped_event(
                    job_id=job_id, timestamp=ts,
                    source="local_runner",
                    save_dir=data.get("save_dir", ""),
                ))
                self._terminal_event_sent = True

    def _on_worker_ready(self, job_id: str, ts: float, data: Dict[str, Any]):
        """Handle worker_ready event — validation + forwarding."""
        import time as _time

        # Emit runtime info as console output
        worker_info_msg = (
            f"Runtime Python:\n{data.get('sys_executable', '?')}\n\n"
            f"Python:\n{data.get('python_version', '?')}\n\n"
            f"Torch:\n{data.get('torch_version', '?')}\n\n"
            f"Torch CUDA:\n{data.get('torch_cuda_version', '?')}\n\n"
            f"CUDA available:\n{data.get('cuda_available')}\n\n"
            f"GPU:\n{data.get('gpu_name', 'N/A')}"
        )
        self._emit_event(create_console_output_event(
            job_id=job_id, timestamp=ts,
            message=worker_info_msg, source="local_runner",
        ))

        # Emit WORKER_READY event
        self._emit_event(TrainingEvent(
            schema_version=1,
            job_id=job_id,
            event_type=TrainingEventType.WORKER_READY,
            timestamp=ts,
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
            source="local_runner",
        ))

    def _on_process_done(self, exit_code: int):
        """Called when QProcess finishes."""
        if not self._active_job_id:
            return
        if not self._terminal_event_sent:
            if exit_code == 0:
                self._emit_event(create_completed_event(
                    job_id=self._active_job_id, timestamp=time.time(),
                    source="local_runner",
                ))
            else:
                self._emit_event(create_failed_event(
                    job_id=self._active_job_id, timestamp=time.time(),
                    error=f"Training process exited with code {exit_code}",
                    source="local_runner",
                ))
            self._terminal_event_sent = True

    def _on_bridge_error(self, error_msg: str):
        """Called when QProcess encounters an error."""
        if not self._active_job_id:
            return
        if not self._terminal_event_sent:
            self._emit_event(create_failed_event(
                job_id=self._active_job_id, timestamp=time.time(),
                error=error_msg, source="local_runner",
            ))
            self._terminal_event_sent = True


# ── helpers ───────────────────────────────────────────────────────────

def _resolve_python_executable(job: TrainingJob) -> str:
    """Select the correct Python for this job."""
    rp = getattr(job, "runtime_python", None) or None
    if rp:
        return rp
    return sys.executable


def _resolve_worker_script() -> str:
    """Return absolute path to the standalone training worker script.

    In frozen (PyInstaller) mode, the worker is compiled into the EXE
    and invoked via `sys.executable --training-worker`. This function
    returns the path only for source-mode usage.
    """
    from ..resource_utils import resource_path, is_frozen
    if is_frozen():
        # Frozen: worker is part of the EXE, return sentinel
        return ":frozen:"
    return str(resource_path(
        "anylabeling/services/auto_training/ultralytics/training_worker.py"
    ))


def _create_training_payload(train_args: Dict[str, Any]) -> str:
    """Create a temporary JSON payload file for the worker."""
    from anylabeling.services.auto_training.ultralytics.trainer import (
        resolve_training_model_path,
    )

    payload_train_args = dict(train_args)
    payload_train_args["model"] = resolve_training_model_path(
        payload_train_args.get("model", "")
    )
    fd, payload_path = tempfile.mkstemp(
        prefix="xanylabeling-train-", suffix=".json"
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload_train_args, f, ensure_ascii=False)
    return payload_path
