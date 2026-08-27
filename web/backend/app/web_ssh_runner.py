"""Qt-free SSHRemoteRunner for the web backend.

The desktop's SSHRemoteRunner streams remote stdout through a QThread
worker and blocks host-key confirmation on a pyqtSignal round-trip —
both require a Qt event loop that does not exist in the FastAPI process.
WebSSHRemoteRunner subclasses SSHRemoteRunner (the JobManager enforces
an isinstance check) and swaps only the Qt-bound pieces:

- stdout/stderr streaming: plain threading.Thread reading the paramiko
  channel line-by-line into ``_on_remote_line``.
- host-key confirmation: TOFU against the profile's saved fingerprint,
  otherwise fail fast — the web has no interactive confirmation path.
- cancel: set the ``_stop_event`` flag first (the inherited synchronous
  start() polls it at stage boundaries: post-connect, per upload entry),
  then close the session channel (remote worker dies on SIGPIPE), a
  best-effort remote ``pkill`` with a kill -0-style verification, and
  finally close the SFTP/SSH handles so an in-flight upload raises
  immediately instead of blocking for minutes.

Everything else (connect/SFTP/upload/config rewrite/artifact download)
is inherited unchanged.
"""

import logging
import os
import posixpath
import threading
import time
from typing import Optional, Tuple

from anylabeling.services.training_center.event_protocol import (
    create_console_output_event,
    create_stopped_event,
)
from anylabeling.services.training_center.runners.base import TrainingRunner
from anylabeling.services.training_center.runners.ssh_remote import (
    SSHRemoteRunner,
    _sftp_mkdir_p,
    _shquote,
)

_log = logging.getLogger(__name__)


class _StopRequestedError(Exception):
    """Raised inside start() hooks when cancel() lands mid-prepare."""


class WebSSHRemoteRunner(SSHRemoteRunner):
    """SSHRemoteRunner subclass passing the job_manager's isinstance check,
    but streams output with plain threads (no QThread/Qt event loop)."""

    def __init__(self):
        # SSHRemoteRunner.__init__ only allocates plain attributes (the
        # pyqtSignal it creates is never emitted on this subclass — the
        # host-key wait below replaces that flow entirely).
        super().__init__(parent=None)
        self._runner_id = f"web-ssh-remote-{SSHRemoteRunner._runner_counter}"
        # Neutralize Qt-only handles so inherited helpers never touch them.
        self.host_key_prompt = None
        self._stream_thread = None
        self._stream_worker = None
        self._cancel_timer = None
        # Cancellation flag polled at start() stage boundaries. Separate
        # from the parent's plain-bool ``_stop_requested`` (which the
        # parent start() rebinds to False on entry).
        self._stop_event = threading.Event()
        # Web streaming state.
        self._channel = None
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()

    # ---- start: per-job re-arm + stop checks at stage boundaries -----------

    def start(self, job, config):
        """Re-arm the per-job stop flag, then run the inherited start.

        A stop() that landed while the job sat in PREPARING (reserve →
        daemon-thread start) has already finalized the job — don't even
        connect in that case.
        """
        self._stop_event.clear()
        # Honor the job's local output directory as the artifact download
        # destination (falls back to the default remote_runs cache).
        self._active_output_dir = str(getattr(job, "output_directory", "") or "")
        status = getattr(job, "status", None)
        if status is not None and status.is_terminal():
            _log.info(
                "WebSSHRemoteRunner: job %s already %s before start; aborting",
                getattr(job, "job_id", "?"), status,
            )
            return False, f"job stopped before remote start ({status})"
        return super().start(job, config)

    def _local_download_base(self, job_id: str) -> str:
        """Download artifacts into the job's configured output directory."""
        base = getattr(self, "_active_output_dir", "") or ""
        if base:
            return base
        return super()._local_download_base(job_id)

    def _check_stop_requested(self):
        if self._stop_event.is_set():
            raise _StopRequestedError(
                f"远程任务在准备阶段被用户停止 (job {self._active_job_id})"
            )

    def _rewrite_data_yaml(self, local_yaml_path: str) -> str:
        # Stage boundary: SSH connected, uploads about to begin.
        self._check_stop_requested()
        return super()._rewrite_data_yaml(local_yaml_path)

    def _upload_dir(self, local_dir: str, remote_dir: str):
        """Same recursion as the parent, but polls the stop flag per entry
        so cancel() interrupts multi-minute dataset uploads (PREPARING)."""
        self._check_stop_requested()
        sftp = self._sftp
        _sftp_mkdir_p(sftp, remote_dir)
        for item in os.listdir(local_dir):
            self._check_stop_requested()
            local_path = os.path.join(local_dir, item)
            remote_path = posixpath.join(remote_dir, item)
            if os.path.isfile(local_path):
                sftp.put(local_path, remote_path)
            elif os.path.isdir(local_path):
                self._upload_dir(local_path, remote_path)

    # ---- host key: TOFU + fail fast (no interactive path in the web) -------

    def _on_host_key(self, hostname, fingerprint, key_type) -> Tuple[bool, bool]:
        """Called from the SSH connect thread.

        Trust-on-first-use policy:
          - fingerprint matches the profile's saved one → auto-trust;
          - otherwise reject immediately. Nothing in the web backend ever
            calls confirm_host_key(), so waiting would just stall the
            connect for 30s before failing anyway.
        """
        saved = ""
        if self._profile is not None:
            saved = (self._profile.known_host_fingerprint or "").strip()
        if saved and saved == fingerprint:
            return (True, False)

        _log.warning(
            "WebSSHRemoteRunner: rejecting unknown host key for %s (%s, %s)",
            hostname, key_type, fingerprint,
        )
        if self._active_job_id:
            self._emit_event(
                create_console_output_event(
                    job_id=self._active_job_id, timestamp=time.time(),
                    message=(
                        f"未知的主机密钥: {hostname} {key_type}\n"
                        f"指纹: {fingerprint}\n"
                        "已拒绝连接。请先在「执行位置 → 管理服务器」中测试连接并确认指纹后重试。"
                    ),
                    source="web_ssh_remote",
                )
            )
        return (False, False)

    # ---- stdout streaming: plain thread (replaces _RemoteStreamWorker) -------

    def _start_output_stream(self, channel):
        """Read the remote worker's stdout/stderr on a daemon thread."""
        self._channel = channel
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(
            target=self._read_channel,
            args=(channel,),
            daemon=True,
            name=f"{self._runner_id}-reader",
        )
        self._reader_thread.start()

    def _read_channel(self, channel):
        stdout_buf = ""
        stderr_buf = ""
        exit_code = -1
        stream_error: Optional[str] = None
        try:
            while not self._reader_stop.is_set():
                got = False
                if channel.recv_ready():
                    data = channel.recv(4096).decode("utf-8", errors="replace")
                    if data:
                        got = True
                        stdout_buf += data
                        while "\n" in stdout_buf:
                            line, stdout_buf = stdout_buf.split("\n", 1)
                            line = line.strip()
                            if line:
                                self._on_remote_line(line)
                if channel.recv_stderr_ready():
                    data = channel.recv_stderr(4096).decode("utf-8", errors="replace")
                    if data:
                        got = True
                        stderr_buf += data
                        while "\n" in stderr_buf:
                            line, stderr_buf = stderr_buf.split("\n", 1)
                            line = line.strip()
                            if line:
                                self._on_remote_line(line)
                if channel.exit_status_ready() and not (
                    channel.recv_ready() or channel.recv_stderr_ready()
                ):
                    break
                if not got:
                    time.sleep(0.05)
            if channel.exit_status_ready():
                try:
                    exit_code = channel.recv_exit_status()
                except Exception:
                    exit_code = -1
            # else: channel closed locally before an exit status arrived —
            # do NOT call recv_exit_status() here, it would block forever.
        except Exception as e:  # channel closed mid-read, transport drop, ...
            if not self._stop_requested:
                stream_error = str(e)

        job_id = self._active_job_id
        try:
            if self._stop_requested:
                if not self._terminal_event_sent and job_id:
                    self._emit_event(
                        create_stopped_event(
                            job_id=job_id, timestamp=time.time(),
                            source="web_ssh_remote",
                        )
                    )
                    self._terminal_event_sent = True
            elif stream_error is not None:
                self._on_remote_error(stream_error)
            else:
                self._on_remote_ended(exit_code)
        finally:
            # JobManager never calls runner.cleanup(); release the SSH/SFTP
            # handles and re-arm the runner for the next job ourselves.
            self._reader_stop.set()
            self._cleanup_resources()

    def _cleanup_resources(self):
        self._reader_stop.set()
        self._channel = None
        super()._cleanup_resources()  # closes sftp/ssh, resets job state
        self._reader_thread = None

    # ---- cancel ---------------------------------------------------------------

    def cancel(self, job_id: str) -> bool:
        if job_id != self._active_job_id:
            return False
        # 0) Set the stop flags first — the inherited start() polls
        #    _stop_event at stage boundaries (post-connect, per upload
        #    entry) and aborts promptly when it is set.
        self._stop_requested = True
        self._stop_event.set()
        # Unblock the reader thread even if the channel misbehaves below.
        self._reader_stop.set()

        # 1) Close the session channel: the remote worker's stdout pipe
        #    breaks, which usually terminates it (SIGPIPE). This also
        #    unblocks the reader thread so a STOPPED event is emitted.
        try:
            if self._channel is not None:
                self._channel.close()
        except Exception:
            _log.debug("WebSSHRemoteRunner: channel close failed", exc_info=True)

        # 2) Best-effort remote kill on a fresh exec channel (the payload
        #    path embeds the job_id, so pkill -f matches only this job),
        #    then verify the worker is really gone — same idea as the
        #    desktop runner's kill -0 probe — escalating to SIGKILL and
        #    warning in the event log when it refuses to die.
        ssh = self._ssh
        if ssh is not None and getattr(ssh, "is_connected", False):
            pattern = f"training_worker.py.*{job_id}"
            try:
                ssh.execute(f"pkill -f {_shquote(pattern)} || true", timeout=5)
                time.sleep(0.5)
                # Verify the worker pids are really gone — same idea as
                # the desktop runner's kill -0 probe — escalating to
                # SIGKILL and warning in the event log when they refuse
                # to die.
                _, out, _ = ssh.execute(
                    f"pgrep -f {_shquote(pattern)} 2>/dev/null || true", timeout=5
                )
                alive = []
                for pid in [p for p in str(out).split() if p.isdigit()]:
                    code, _, _ = ssh.execute(f"kill -0 {pid} 2>/dev/null", timeout=5)
                    if code == 0:
                        alive.append(pid)
                if alive:
                    ssh.execute(f"pkill -9 -f {_shquote(pattern)} || true", timeout=5)
                    time.sleep(0.3)
                    _, out2, _ = ssh.execute(
                        f"pgrep -f {_shquote(pattern)} 2>/dev/null || true", timeout=5
                    )
                    still = [p for p in str(out2).split() if p.isdigit()]
                    if still:
                        msg = (
                            f"远程训练进程未能终止 (pid: {', '.join(still)})，"
                            "请登录服务器手动检查清理。"
                        )
                        _log.warning("WebSSHRemoteRunner: %s", msg)
                        self._emit_event(
                            create_console_output_event(
                                job_id=job_id, timestamp=time.time(),
                                message=f"警告: {msg}",
                                source="web_ssh_remote",
                            )
                        )
            except Exception as e:  # noqa - cancel must never raise
                _log.info("WebSSHRemoteRunner: remote pkill fallback failed: %s", e)

        # 3) Close the SFTP/SSH handles so an in-flight upload (start()
        #    runs synchronously on another thread during PREPARING) raises
        #    immediately instead of blocking for minutes.
        for handle in (self._sftp, self._ssh):
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                _log.debug("WebSSHRemoteRunner: handle close failed", exc_info=True)

        return True


def register_web_ssh_runner():
    """Register WebSSHRemoteRunner as the 'remote_ssh' runner in the global factory."""
    from anylabeling.services.training_center.runners.factory import RunnerFactory

    factory = RunnerFactory.get_instance()
    factory.register("remote_ssh", WebSSHRemoteRunner())
