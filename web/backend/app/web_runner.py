"""Threading-based TrainingRunner for the web backend.

The desktop's LocalRunner drives the training worker through QProcess,
which requires a Qt event loop that does not exist in the FastAPI
process. WebLocalRunner implements the same TrainingRunner contract
with plain subprocess + reader threads, and is registered into the
global RunnerFactory as the "local" runner for the web backend.
"""

import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

from anylabeling.services.training_center.event_protocol import (
    create_completed_event,
    create_console_output_event,
    create_failed_event,
    create_process_started_event,
    create_stopped_event,
)
from anylabeling.services.training_center.models import TrainingJob
from anylabeling.services.training_center.runners.base import (
    RunnerCapability,
    TrainingRunner,
)
from anylabeling.services.training_center.runners.local import (
    TRAINING_WORKER_EVENT_PREFIX,
    LocalRunner,
    _create_training_payload,
    _resolve_python_executable,
    _resolve_worker_script,
)


class WebLocalRunner(LocalRunner):
    """LocalRunner subclass passing the job_manager's isinstance check,
    but drives the worker with plain threads (no QProcess/Qt loop)."""

    def __init__(self):
        # deliberately skip LocalRunner.__init__ (it builds a QProcess
        # bridge); only the ABC's callback plumbing is needed
        TrainingRunner.__init__(self)
        self._proc: Optional[subprocess.Popen] = None
        self._active_job_id = ""
        self._job: Optional[TrainingJob] = None
        self._stop_requested = False
        self._terminal_event_sent = False
        self._lock = threading.RLock()

    # ---- metadata -------------------------------------------------------
    @property
    def runner_id(self) -> str:
        return "web-local-1"

    @property
    def execution_mode(self) -> str:
        return "local"

    @property
    def capability(self) -> RunnerCapability:
        return RunnerCapability(
            name="WebLocalRunner",
            execution_mode="local",
            max_workers=1,
            supports_gpu=True,
            supports_cancel=True,
        )

    # ---- lifecycle --------------------------------------------------------
    def prepare(self, job: TrainingJob, config: Dict[str, Any]) -> Tuple[bool, str]:
        worker_path = _resolve_worker_script()
        if worker_path != ":frozen:" and not os.path.isfile(worker_path):
            return False, f"Worker script not found: {worker_path}"
        return True, "ready"

    def start(self, job: TrainingJob, config: Dict[str, Any]) -> Tuple[bool, str]:
        with self._lock:
            if self._active_job_id:
                return False, "Runner already has an active job"
            try:
                python_exe = _resolve_python_executable(job)
                payload_path = _create_training_payload(config)
                worker_script = _resolve_worker_script()
                command = [python_exe, worker_script, "--payload", payload_path]

                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONUNBUFFERED"] = "1"

                self._proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # merged channels like QProcess
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    bufsize=1,
                )
            except Exception as e:  # noqa
                self._proc = None
                return False, f"Failed to start local training worker: {e}"

            self._job = job
            self._active_job_id = job.job_id
            self._stop_requested = False
            self._terminal_event_sent = False

        self._emit_event(
            create_process_started_event(
                job_id=job.job_id, timestamp=time.time(),
                pid=self._proc.pid, source="web_local_runner",
            )
        )
        threading.Thread(target=self._read_output, daemon=True).start()
        threading.Thread(target=self._watch, daemon=True).start()
        return True, "Worker process started"

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id != self._active_job_id or not self._proc:
                return False
            self._stop_requested = True
        self._terminate_tree(force=False)
        # force-kill fallback after 3s, mirroring the desktop runner
        threading.Timer(3.0, lambda: self.force_stop(job_id)).start()
        return True

    def force_stop(self, job_id: str) -> bool:
        with self._lock:
            if job_id != self._active_job_id:
                return False
        self._terminate_tree(force=True)
        self._emit_terminal(
            create_stopped_event(
                job_id=job_id, timestamp=time.time(), source="web_local_runner"
            )
        )
        return True

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            return job_id == self._active_job_id and bool(
                self._proc and self._proc.poll() is None
            )

    def get_status(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            return {
                "active_job_id": self._active_job_id,
                "pid": self._proc.pid if self._proc else None,
                "is_running": self.is_running(job_id),
                "stop_requested": self._stop_requested,
            }

    def cleanup(self, job_id: str):
        with self._lock:
            if job_id != self._active_job_id:
                return
            self._active_job_id = ""
            self._job = None
            self._proc = None
            self._stop_requested = False

    # ---- internals ---------------------------------------------------------
    def _terminate_tree(self, force: bool):
        proc = self._proc
        if not proc or proc.poll() is not None:
            return
        try:
            if sys.platform.startswith("win"):
                subprocess.run(
                    ["taskkill", "/F" if force else "/T", "/PID", str(proc.pid)]
                    if force
                    else ["taskkill", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    timeout=10,
                )
            else:
                proc.kill() if force else proc.terminate()
        except Exception:
            pass

    def _emit_terminal(self, event):
        if not self._terminal_event_sent:
            self._emit_event(event)
            self._terminal_event_sent = True

    def _read_output(self):
        proc = self._proc
        if not proc or not proc.stdout:
            return
        try:
            for line in iter(proc.stdout.readline, ""):
                self._handle_line(line.rstrip("\n"))
        except Exception:
            pass

    def _handle_line(self, line: str):
        if not line or not self._active_job_id:
            return
        ts = time.time()
        if line.startswith(TRAINING_WORKER_EVENT_PREFIX):
            try:
                data = json.loads(line[len(TRAINING_WORKER_EVENT_PREFIX):])
            except json.JSONDecodeError:
                self._emit_event(
                    create_console_output_event(
                        job_id=self._active_job_id, timestamp=ts,
                        message=line, source="web_local_runner",
                    )
                )
                return
            self._handle_worker_event(data, ts)
        else:
            self._emit_event(
                create_console_output_event(
                    job_id=self._active_job_id, timestamp=ts,
                    message=line, stream="stdout", source="web_local_runner",
                )
            )

    def _handle_worker_event(self, data: Dict[str, Any], ts: float):
        event_type = data.pop("event", "")
        job_id = self._active_job_id

        if event_type == "worker_ready":
            msg = (
                f"Runtime Python:\n{data.get('sys_executable', '?')}\n\n"
                f"Python:\n{data.get('python_version', '?')}\n\n"
                f"Torch:\n{data.get('torch_version', '?')}\n\n"
                f"Torch CUDA:\n{data.get('torch_cuda_version', '?')}\n\n"
                f"CUDA available:\n{data.get('cuda_available')}\n\n"
                f"GPU:\n{data.get('gpu_name', 'N/A')}"
            )
            self._emit_event(
                create_console_output_event(
                    job_id=job_id, timestamp=ts, message=msg, source="web_local_runner"
                )
            )
        elif event_type == "training_log":
            self._emit_event(
                create_console_output_event(
                    job_id=job_id, timestamp=ts,
                    message=data.get("message", ""),
                    stream="stdout", source="web_local_runner",
                )
            )
        elif event_type == "training_completed":
            self._emit_terminal(
                create_completed_event(
                    job_id=job_id, timestamp=ts, source="web_local_runner",
                    results=data.get("results"), save_dir=data.get("save_dir", ""),
                )
            )
        elif event_type == "training_error":
            self._emit_terminal(
                create_failed_event(
                    job_id=job_id, timestamp=ts,
                    error=data.get("error", "Unknown error"),
                    source="web_local_runner",
                    save_dir=data.get("save_dir", ""),
                )
            )
        elif event_type == "training_stopped":
            self._emit_terminal(
                create_stopped_event(
                    job_id=job_id, timestamp=ts, source="web_local_runner",
                    save_dir=data.get("save_dir", ""),
                )
            )

    def _watch(self):
        proc = self._proc
        if proc is None:
            return
        exit_code = proc.wait()
        job_id = self._active_job_id
        if not job_id:
            return
        # no terminal event from the worker: synthesize one from the exit code
        if self._stop_requested:
            self._emit_terminal(
                create_stopped_event(
                    job_id=job_id, timestamp=time.time(), source="web_local_runner"
                )
            )
        elif exit_code == 0:
            self._emit_terminal(
                create_completed_event(
                    job_id=job_id, timestamp=time.time(), source="web_local_runner"
                )
            )
        else:
            self._emit_terminal(
                create_failed_event(
                    job_id=job_id, timestamp=time.time(),
                    error=f"Worker exited with code {exit_code}",
                    source="web_local_runner",
                )
            )


def register_web_runner():
    """Register WebLocalRunner as the 'local' runner in the global factory."""
    from anylabeling.services.training_center.runners.factory import RunnerFactory

    factory = RunnerFactory.get_instance()
    factory.register("local", WebLocalRunner())
