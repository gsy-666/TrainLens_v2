"""SSHRemoteRunner — executes training on a remote Linux host via SSH/SFTP.

Implements the TrainingRunner interface for remote execution.
All network I/O runs on background threads — GUI never blocks.
"""

import json
import logging
import os
import posixpath
import tempfile
import threading
import time
import traceback as _traceback
import uuid
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Tuple

from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer

from ..runners.base import TrainingRunner, RunnerCapability
from ..models import TrainingJob, TrainingStatus
from ..event_protocol import (
    TrainingEvent,
    TrainingEventType,
    create_console_output_event,
    create_completed_event,
    create_failed_event,
    create_stopped_event,
)
from ..remote.models import RemoteProfile
from ..remote.storage import get_profile_store
from ..remote.ssh_service import SSHConnectionService

_log = logging.getLogger(__name__)


class _RemoteStreamWorker(QObject):
    """Reads remote stdout line-by-line on a background thread."""

    line_received = pyqtSignal(str)
    stream_ended = pyqtSignal(int)  # exit_code
    stream_error = pyqtSignal(str)

    def __init__(self, channel, parent=None):
        super().__init__(parent)
        self._channel = channel
        self._running = False

    def start_reading(self):
        self._running = True
        while self._running:
            if self._channel.recv_ready():
                try:
                    data = self._channel.recv(4096).decode("utf-8", errors="replace")
                    if data:
                        for line in data.split("\n"):
                            line = line.strip()
                            if line:
                                self.line_received.emit(line)
                except Exception as e:
                    self.stream_error.emit(str(e))
                    break
            if self._channel.exit_status_ready():
                break
            QThread.msleep(50)

        # Flush remaining
        try:
            while self._channel.recv_ready():
                data = self._channel.recv(4096).decode("utf-8", errors="replace")
                if data:
                    for line in data.split("\n"):
                        line = line.strip()
                        if line:
                            self.line_received.emit(line)
        except Exception:
            pass

        exit_code = self._channel.recv_exit_status()
        self.stream_ended.emit(exit_code)

    def stop(self):
        self._running = False


class SSHRemoteRunner(TrainingRunner):
    """Runs training on a remote SSH host."""

    _runner_counter = 0

    def __init__(self, parent: QObject = None):
        super().__init__()
        SSHRemoteRunner._runner_counter += 1
        self._runner_id = f"ssh-remote-{SSHRemoteRunner._runner_counter}"
        self._parent = parent

        self._job: Optional[TrainingJob] = None
        self._active_job_id: str = ""
        self._config: Dict[str, Any] = {}
        self._profile: Optional[RemoteProfile] = None
        self._ssh: Optional[SSHConnectionService] = None
        self._sftp = None

        # Threading
        self._stream_thread: Optional[QThread] = None
        self._stream_worker: Optional[_RemoteStreamWorker] = None

        # State
        self._stop_requested = False
        self._terminal_event_sent = False
        self._remote_job_dir: str = ""
        self._remote_pid: int = 0
        self._password: str = ""  # session-only
        self._cancel_timer: Optional[QTimer] = None

        # Host key interaction (signal-based, thread-safe)
        self.host_key_prompt = pyqtSignal(str, str, str)  # hostname, fingerprint, key_type
        self._host_key_result: Optional[tuple] = None
        self._host_key_event = threading.Event()

    # ── metadata ──

    @property
    def runner_id(self) -> str:
        return self._runner_id

    @property
    def execution_mode(self) -> str:
        return "remote_ssh"

    @property
    def capability(self) -> RunnerCapability:
        return RunnerCapability(
            name="SSHRemoteRunner",
            execution_mode="remote_ssh",
            max_workers=1,
            supports_gpu=True,
            supports_cancel=True,
        )

    # ── Host key callback (thread-safe) ──

    def confirm_host_key(self, trust: bool, save: bool):
        self._host_key_result = (trust, save)
        self._host_key_event.set()

    def _on_host_key(self, hostname, fingerprint, key_type) -> Tuple[bool, bool]:
        """Called from SSH worker thread. Emits signal for GUI thread."""
        self._host_key_result = None
        self._host_key_event.clear()
        self.host_key_prompt.emit(hostname, fingerprint, key_type)
        # Wait for GUI response (with timeout)
        if not self._host_key_event.wait(timeout=30):
            return (False, False)
        return self._host_key_result or (False, False)

    # ── lifecycle ──

    def prepare(self, job: TrainingJob, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate remote profile, SSH credentials, and config."""
        profile_id = getattr(job, "remote_profile_id", None) or ""
        if not profile_id:
            return False, "No remote profile selected"

        store = get_profile_store()
        profile = store.get(profile_id)
        if not profile:
            return False, f"Remote profile not found: {profile_id}"

        if not profile.host:
            return False, "Remote host is not configured"

        if not profile.remote_python:
            return False, "Remote Python path is not configured"

        # Password check for password auth
        if profile.auth_method.value == "password":
            pwd = getattr(job, "_session_password", "") or ""
            if not pwd:
                return False, "Password is required for the selected SSH profile"

        self._profile = profile
        self._password = getattr(job, "_session_password", "") or ""

        return True, "ready"

    def start(self, job: TrainingJob, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Connect SSH, upload, launch remote worker, stream output."""
        if self._active_job_id:
            return False, "Runner already has an active job"

        self._job = job
        self._active_job_id = job.job_id
        self._config = dict(config)
        self._stop_requested = False
        self._terminal_event_sent = False

        def _stage(msg: str):
            _log.info("Remote stage: %s", msg)
            self._emit_event(create_console_output_event(
                job_id=job.job_id, timestamp=time.time(),
                message=f"Remote stage: {msg}", source="ssh_remote",
            ))

        try:
            # 1. Validate profile
            _stage("validating profile")
            if not self._profile:
                raise ValueError("Remote profile not set")

            # 2. Connect SSH
            _stage("connecting SSH")
            self._ssh = SSHConnectionService()
            ok, msg = self._ssh.connect(
                self._profile,
                password=self._password,
                on_host_key=self._on_host_key,
            )
            if not ok:
                raise ConnectionError(f"SSH connection failed: {msg}")

            fingerprint = msg if ":" in msg else ""
            if fingerprint and self._profile:
                self._profile.known_host_fingerprint = fingerprint
                get_profile_store().save(self._profile)

            # 3. Open SFTP
            _stage("opening SFTP")
            import paramiko
            try:
                self._sftp = self._ssh._client.open_sftp()
            except Exception as e:
                raise OSError(f"SFTP open failed: {e}")

            # 4. Create remote job directory structure
            _stage("creating job directory")
            remote_base = self._profile.remote_workspace.rstrip("/")
            self._remote_job_dir = f"{remote_base}/trainlens/jobs/{job.job_id}"

            # Create full directory tree
            dirs = [
                "config", "dataset", "inputs/model",
                "logs", "runs", "status",
            ]
            code, out, err = self._ssh.execute(
                f"mkdir -p " + " ".join(
                    _shquote(posixpath.join(self._remote_job_dir, d)) for d in dirs
                ),
                timeout=10,
            )
            if code != 0:
                raise OSError(f"Failed to create remote job directories: {err}")

            # 5. Rewrite dataset YAML and upload
            _stage("rewriting dataset YAML")
            self._rewrite_data_yaml(config.get("data", ""))

            # 6. Upload dataset
            _stage("uploading dataset")
            data_yaml = config.get("data", "")
            if data_yaml and os.path.isfile(data_yaml):
                dataset_dir = os.path.dirname(os.path.abspath(data_yaml))
                remote_dataset = posixpath.join(self._remote_job_dir, "dataset")
                _sftp_mkdir_p(self._sftp, remote_dataset)
                self._upload_dir(dataset_dir, remote_dataset)

            # 7. Upload model
            _stage("uploading model")
            model_path = config.get("model", "")
            remote_model_path = ""
            if model_path and os.path.isfile(model_path):
                model_name = os.path.basename(model_path)
                model_size = os.path.getsize(model_path)
                remote_model_dir = posixpath.join(self._remote_job_dir, "inputs", "model")
                remote_model_path = posixpath.join(remote_model_dir, model_name)

                _sftp_mkdir_p(self._sftp, remote_model_dir)

                self._emit_event(create_console_output_event(
                    job_id=job.job_id, timestamp=time.time(),
                    message=(
                        f"Local model source: {model_path}\n"
                        f"Local model exists: True\n"
                        f"Local model size: {model_size}\n"
                        f"Remote model directory: {remote_model_dir}\n"
                        f"Remote model destination: {remote_model_path}"
                    ),
                    source="ssh_remote",
                ))

                # Atomic upload via .part temp file
                remote_part = remote_model_path + ".part"
                try:
                    self._sftp.put(model_path, remote_part)
                    # Verify size
                    remote_stat = self._sftp.stat(remote_part)
                    if remote_stat.st_size != model_size:
                        raise OSError(
                            f"Upload size mismatch: local={model_size} remote={remote_stat.st_size}"
                        )
                    # Rename to final
                    self._sftp.rename(remote_part, remote_model_path)
                    self._emit_event(create_console_output_event(
                        job_id=job.job_id, timestamp=time.time(),
                        message="Model upload completed and verified",
                        source="ssh_remote",
                    ))
                except Exception:
                    # Clean up .part on failure
                    try:
                        self._sftp.remove(remote_part)
                    except Exception:
                        pass
                    raise

            # 8. Upload worker script
            _stage("uploading worker")
            local_worker = str(_resolve_worker_for_upload())
            remote_worker = posixpath.join(self._remote_job_dir, "training_worker.py")
            self._sftp.put(local_worker, remote_worker)

            # 9. Upload job config (with all paths rewritten to Linux)
            _stage("uploading config")
            remote_config = dict(config)
            remote_config["data"] = posixpath.join(self._remote_job_dir, "config", "data.yaml")
            if remote_model_path:
                remote_config["model"] = remote_model_path
            # Rewrite project to remote runs directory
            remote_config["project"] = posixpath.join(self._remote_job_dir, "runs")

            remote_config_dir = posixpath.join(self._remote_job_dir, "config")
            _sftp_mkdir_p(self._sftp, remote_config_dir)

            fd, tmp = tempfile.mkstemp(suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(remote_config, f, ensure_ascii=False)
            remote_config_path = posixpath.join(remote_config_dir, "job_config.json")
            self._sftp.put(tmp, remote_config_path)
            os.unlink(tmp)

            # 10. Launch remote worker
            _stage("launching worker")
            remote_py = self._profile.remote_python or "python3"
            remote_worker = posixpath.join(self._remote_job_dir, "training_worker.py")
            # Worker CLI contract: --payload <json_file_path>
            remote_payload = posixpath.join(self._remote_job_dir, "config", "job_config.json")
            command = (
                f"{_shquote(remote_py)} {_shquote(remote_worker)} "
                f"--payload {_shquote(remote_payload)}"
            )

            self._emit_event(create_console_output_event(
                job_id=job.job_id, timestamp=time.time(),
                message=(
                    f"Worker CLI contract: --payload\n"
                    f"Remote worker: {remote_worker}\n"
                    f"Remote payload: {remote_payload}"
                ),
                source="ssh_remote",
            ))

            _log.info("Remote command: %s", command[:200])
            transport = self._ssh._client.get_transport()
            channel = transport.open_session()
            channel.exec_command(command)

            # 11. Start streaming stdout
            _stage("streaming output")
            self._stream_worker = _RemoteStreamWorker(channel)
            self._stream_thread = QThread(self._parent)
            self._stream_worker.moveToThread(self._stream_thread)

            self._stream_worker.line_received.connect(self._on_remote_line)
            self._stream_worker.stream_ended.connect(self._on_remote_ended)
            self._stream_worker.stream_error.connect(self._on_remote_error)
            self._stream_thread.started.connect(self._stream_worker.start_reading)
            self._stream_thread.start()

            self._emit_event(create_console_output_event(
                job_id=job.job_id, timestamp=time.time(),
                message=f"Runner: {self.runner_id}\n"
                        f"Execution mode: remote\n"
                        f"Remote host: {self._profile.host}:{self._profile.port}\n"
                        f"Remote Python: {remote_py}\n"
                        f"Remote job dir: {self._remote_job_dir}",
                source="ssh_remote",
            ))

            return True, "Remote worker process started"

        except Exception as e:
            error_msg = str(e)
            # Sanitize: never log passwords
            if self._password and self._password in error_msg:
                error_msg = error_msg.replace(self._password, "***")
            _log.error("SSHRemoteRunner.start: %s", error_msg, exc_info=True)

            self._emit_event(create_failed_event(
                job_id=job.job_id, timestamp=time.time(),
                error=error_msg,
                source="ssh_remote",
            ))
            self._terminal_event_sent = True
            self._cleanup_resources()
            return False, error_msg


    def _upload_dir(self, local_dir: str, remote_dir: str):
        """Recursively upload a directory via SFTP."""
        sftp = self._sftp
        _sftp_mkdir_p(sftp, remote_dir)

        for item in os.listdir(local_dir):
            local_path = os.path.join(local_dir, item)
            remote_path = posixpath.join(remote_dir, item)
            if os.path.isfile(local_path):
                sftp.put(local_path, remote_path)
            elif os.path.isdir(local_path):
                self._upload_dir(local_path, remote_path)

    def _rewrite_data_yaml(self, local_yaml_path: str) -> str:
        """Rewrite data.yaml with remote paths and upload it."""
        import yaml as _yaml_lib
        with open(local_yaml_path, "r", encoding="utf-8") as f:
            data = _yaml_lib.safe_load(f)

        remote_dataset = posixpath.join(self._remote_job_dir, "dataset")
        data["path"] = remote_dataset
        if "train" in data:
            data["train"] = "images/train"
        if "val" in data:
            data["val"] = "images/val"
        if "test" in data:
            data["test"] = "images/test"

        remote_yaml = posixpath.join(self._remote_job_dir, "config", "data.yaml")
        _sftp_mkdir_p(self._sftp, posixpath.dirname(remote_yaml))
        fd, tmp = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            _yaml_lib.dump(data, f, default_flow_style=False)
        self._sftp.put(tmp, remote_yaml)
        os.unlink(tmp)

        return remote_yaml

    # ── stdout parsing ──

    def _on_remote_line(self, line: str):
        """Parse a line from remote worker stdout."""
        if not self._active_job_id:
            return
        prefix = "__XANYLABELING_TRAIN_EVENT__="
        if line.startswith(prefix):
            try:
                data = json.loads(line[len(prefix):])
            except json.JSONDecodeError:
                self._emit_event(create_console_output_event(
                    job_id=self._active_job_id, timestamp=time.time(),
                    message=line, source="ssh_remote",
                ))
                return
            self._handle_worker_event(data)
        else:
            self._emit_event(create_console_output_event(
                job_id=self._active_job_id, timestamp=time.time(),
                message=line, stream="stdout", source="ssh_remote",
            ))

    def _handle_worker_event(self, data: Dict[str, Any]):
        event_type = data.pop("event", "")
        job_id = self._active_job_id
        ts = time.time()

        if event_type == "worker_ready":
            self._on_worker_ready(job_id, ts, data)
        elif event_type in ("training_log", "log"):
            self._emit_event(create_console_output_event(
                job_id=job_id, timestamp=ts,
                message=data.get("message", ""),
                source="ssh_remote",
            ))
        elif event_type == "epoch_metrics":
            # Forward as structured EPOCH_METRICS event for progress + metrics
            self._emit_event(TrainingEvent(
                schema_version=1, job_id=job_id,
                event_type=TrainingEventType.EPOCH_METRICS,
                timestamp=ts,
                payload=data, source="ssh_remote",
            ))
        elif event_type == "training_completed":
            if not self._terminal_event_sent:
                save_dir = data.get("save_dir", "")
                # Download artifacts before marking completed
                self._emit_event(create_console_output_event(
                    job_id=job_id, timestamp=ts,
                    message=f"Remote stage: downloading artifacts\nRemote save directory: {save_dir}",
                    source="ssh_remote",
                ))
                local_dir = self._download_results(job_id, save_dir)

                self._emit_event(create_completed_event(
                    job_id=job_id, timestamp=ts,
                    source="ssh_remote",
                    results=data.get("results"),
                    save_dir=local_dir or save_dir,
                ))
                self._terminal_event_sent = True
        elif event_type == "training_error":
            if not self._terminal_event_sent:
                self._emit_event(create_failed_event(
                    job_id=job_id, timestamp=ts,
                    error=data.get("error", "Unknown error"),
                    source="ssh_remote",
                ))
                self._terminal_event_sent = True
        elif event_type == "training_stopped":
            if not self._terminal_event_sent:
                self._emit_event(create_stopped_event(
                    job_id=job_id, timestamp=ts,
                    source="ssh_remote",
                ))
                self._terminal_event_sent = True

    def _on_worker_ready(self, job_id: str, ts: float, data: Dict[str, Any]):
        worker_info = (
            f"Runtime Python:\n{data.get('sys_executable', '?')}\n\n"
            f"Python:\n{data.get('python_version', '?')}\n\n"
            f"Torch:\n{data.get('torch_version', '?')}\n\n"
            f"Torch CUDA:\n{data.get('torch_cuda_version', '?')}\n\n"
            f"CUDA available:\n{data.get('cuda_available')}\n\n"
            f"GPU:\n{', '.join(data.get('gpu_names', ['N/A']))}"
        )
        self._emit_event(create_console_output_event(
            job_id=job_id, timestamp=ts,
            message=worker_info, source="ssh_remote",
        ))
        self._emit_event(TrainingEvent(
            schema_version=1, job_id=job_id,
            event_type=TrainingEventType.WORKER_READY,
            timestamp=ts,
            payload=data, source="ssh_remote",
        ))

    def _on_remote_ended(self, exit_code: int):
        if not self._active_job_id:
            return
        if not self._terminal_event_sent:
            if exit_code == 0:
                self._emit_event(create_completed_event(
                    job_id=self._active_job_id, timestamp=time.time(),
                    source="ssh_remote",
                ))
            else:
                self._emit_event(create_failed_event(
                    job_id=self._active_job_id, timestamp=time.time(),
                    error=(
                        f"Remote worker exited before ready with code {exit_code}.\n"
                        f"Check the remote log for details:\n"
                        f"  ssh {self._profile.username}@{self._profile.host} "
                        f"'cat {self._remote_job_dir}/logs/*.log'"
                    ),
                    source="ssh_remote",
                ))
            self._terminal_event_sent = True

    def _on_remote_error(self, err: str):
        if not self._active_job_id:
            return
        if not self._terminal_event_sent:
            self._emit_event(create_failed_event(
                job_id=self._active_job_id, timestamp=time.time(),
                error=f"Remote stream error: {err}",
                source="ssh_remote",
            ))
            self._terminal_event_sent = True

    # ── download ──

    def _download_results(self, job_id: str, remote_save_dir: str = "") -> str:
        """Download artifacts from remote to local store.

        Returns the local directory path, or empty string on failure.
        """
        if not remote_save_dir:
            remote_save_dir = posixpath.join(self._remote_job_dir, "runs")

        local_base = os.path.join(
            os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
            "TrainLens", "remote_runs", job_id,
        )
        os.makedirs(local_base, exist_ok=True)

        sftp = self._sftp
        if not sftp:
            _log.warning("No SFTP connection for artifact download")
            return ""

        downloaded = []
        patterns = [
            "results.csv", "args.yaml",
            "weights/best.pt", "weights/last.pt",
        ]
        optional = [
            "results.png", "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png",
        ]

        def _try_dl(rel: str):
            remote_path = posixpath.join(remote_save_dir, rel)
            local_path = os.path.join(local_base, os.path.basename(rel))
            try:
                sftp.stat(remote_path)
                sftp.get(remote_path, local_path)
                downloaded.append(rel)
                self._emit_event(create_console_output_event(
                    job_id=job_id, timestamp=time.time(),
                    message=f"Downloaded {rel}",
                    source="ssh_remote",
                ))
            except Exception:
                pass

        for pattern in patterns:
            _try_dl(pattern)

        for pattern in optional:
            _try_dl(pattern)

        # Also try to find results.csv in a subdirectory (runs/expN/results.csv)
        try:
            # Look for a subdirectory under remote_save_dir that has results.csv
            # If remote_save_dir itself doesn't have results.csv
            if "results.csv" not in downloaded:
                for entry in sftp.listdir(remote_save_dir):
                    sub = posixpath.join(remote_save_dir, entry)
                    try:
                        sftp.stat(posixpath.join(sub, "results.csv"))
                        # Found a run subdirectory — download from there
                        for pattern in patterns:
                            remote_path = posixpath.join(sub, pattern)
                            local_path = os.path.join(local_base, os.path.basename(pattern))
                            try:
                                sftp.get(remote_path, local_path)
                                downloaded.append(f"{entry}/{pattern}")
                            except Exception:
                                pass
                        for pattern in optional:
                            remote_path = posixpath.join(sub, pattern)
                            local_path = os.path.join(local_base, os.path.basename(pattern))
                            try:
                                sftp.get(remote_path, local_path)
                                downloaded.append(f"{entry}/{pattern}")
                            except Exception:
                                pass
                        break
                    except IOError:
                        continue
        except Exception as e:
            _log.warning("Remote directory listing failed: %s", e)

        self._emit_event(create_console_output_event(
            job_id=job_id, timestamp=time.time(),
            message=(
                f"Remote artifact synchronization completed\n"
                f"Downloaded: {len(downloaded)} files\n"
                f"Local path: {local_base}"
            ),
            source="ssh_remote",
        ))

        return local_base

    # ── cancel ──

    def cancel(self, job_id: str) -> bool:
        if job_id != self._active_job_id:
            return False
        self._stop_requested = True

        if self._ssh and self._ssh.is_connected:
            # Get remote PID from status
            code, out, err = self._ssh.execute(
                f"cat {_shquote(self._remote_job_dir)}/status/status.json 2>/dev/null",
                timeout=5,
            )
            if code == 0:
                try:
                    status = json.loads(out)
                    pid = status.get("pid", 0)
                    if pid:
                        self._ssh.execute(f"kill -TERM {pid}", timeout=5)
                        time.sleep(2)
                        code2, _, _ = self._ssh.execute(f"kill -0 {pid} 2>/dev/null", timeout=5)
                        if code2 == 0:
                            self._ssh.execute(f"kill -KILL {pid}", timeout=5)
                except Exception:
                    pass

        return True

    def force_stop(self, job_id: str) -> bool:
        return self.cancel(job_id)

    def is_running(self, job_id: str) -> bool:
        return self._active_job_id == job_id and not self._terminal_event_sent

    def get_status(self, job_id: str) -> Dict[str, Any]:
        return {
            "active_job_id": self._active_job_id,
            "remote_host": self._profile.host if self._profile else "",
            "remote_job_dir": self._remote_job_dir,
            "is_running": self.is_running(job_id),
        }

    def cleanup(self, job_id: str):
        self._cleanup_resources()

    def _cleanup_resources(self):
        if self._stream_worker:
            self._stream_worker.stop()
        if self._stream_thread:
            self._stream_thread.quit()
            self._stream_thread.wait(3000)
            self._stream_thread = None
        self._stream_worker = None
        if self._sftp:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        if self._ssh:
            try:
                self._ssh.close()
            except Exception:
                pass
            self._ssh = None
        self._active_job_id = ""
        self._job = None
        self._config = {}


def _sftp_mkdir_p(sftp, remote_dir: str):
    """Recursively create a remote directory (like mkdir -p).

    Uses posixpath to split the path and create each level.
    Directory already exists → no error.
    """
    remote_dir = remote_dir.rstrip("/")
    if not remote_dir:
        return
    parts = remote_dir.split("/")
    current = ""
    for part in parts:
        if not part:
            continue
        current = current + "/" + part
        try:
            sftp.stat(current)
        except IOError:
            try:
                sftp.mkdir(current)
            except IOError:
                pass  # race: another thread/session created it


def _shquote(s: str) -> str:
    """Shell-quote a single argument for SSH remote execution."""
    return "'" + str(s).replace("'", "'\\''") + "'"


def _resolve_worker_for_upload() -> str:
    """Return local path to training_worker.py for SSH upload.

    In frozen (PyInstaller) mode, the worker is bundled as a data file.
    In source mode, it's resolved relative to the project root.
    """
    from ..resource_utils import resource_path
    return str(resource_path(
        "anylabeling/services/auto_training/ultralytics/training_worker.py"
    ))
