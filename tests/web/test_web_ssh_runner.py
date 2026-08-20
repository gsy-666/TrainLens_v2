"""Tests for WebSSHRemoteRunner — the Qt-free SSH runner used by the web
backend — and for remote live-metric feeding in WebTrainingService.

No real SSH server: channel/ssh objects are fakes. The runner must be
constructible and drivable without a QApplication."""

from __future__ import annotations

import threading
import time
from collections import deque
from types import SimpleNamespace

import pytest

from anylabeling.services.training_center.runners.factory import RunnerFactory
from anylabeling.services.training_center.runners.ssh_remote import SSHRemoteRunner
from web.backend.app.web_ssh_runner import (
    WebSSHRemoteRunner,
    register_web_ssh_runner,
)


# ---- construction + factory registration ----------------------------------------


def test_runner_constructs_without_qapplication():
    runner = WebSSHRemoteRunner()
    assert runner.host_key_prompt is None  # pyqtSignal neutralized
    assert runner.capability.execution_mode == "remote_ssh"
    assert isinstance(runner, SSHRemoteRunner)


def test_register_web_ssh_runner_overrides_factory():
    register_web_ssh_runner()
    runner = RunnerFactory.get_instance().get("remote_ssh")
    assert isinstance(runner, WebSSHRemoteRunner)
    # job_manager's hard isinstance check requires the desktop base class
    assert isinstance(runner, SSHRemoteRunner)


# ---- host key flow (TOFU, fail fast — no interactive path in the web) -------


def test_host_key_auto_trusts_saved_fingerprint():
    runner = WebSSHRemoteRunner()
    runner._profile = SimpleNamespace(known_host_fingerprint="aa:bb")
    assert runner._on_host_key("host", "aa:bb", "ssh-rsa") == (True, False)


def test_host_key_unknown_fails_fast_without_waiting():
    runner = WebSSHRemoteRunner()
    runner._profile = SimpleNamespace(known_host_fingerprint="")
    start = time.monotonic()
    assert runner._on_host_key("host", "ff:ee", "ssh-ed25519") == (False, False)
    # no 30s Event.wait — nothing in the web confirms host keys
    assert time.monotonic() - start < 1.0


def test_host_key_mismatch_fails_fast_and_emits_console_hint():
    runner = WebSSHRemoteRunner()
    runner._profile = SimpleNamespace(known_host_fingerprint="aa:bb")
    runner._active_job_id = "job-1"
    events = []
    runner.subscribe(events.append)
    assert runner._on_host_key("host", "ff:ee", "ssh-ed25519") == (False, False)
    assert len(events) == 1
    event = events[0].to_dict()
    assert event["event_type"] == "console_output"
    assert "ff:ee" in event["payload"]["message"]
    assert "管理服务器" in event["payload"]["message"]


# ---- cancel -------------------------------------------------------------------------


class _FakeChannel:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_cancel_without_active_job_returns_false():
    runner = WebSSHRemoteRunner()
    assert runner.cancel("nope") is False


def test_cancel_closes_channel_and_issues_remote_pkill():
    runner = WebSSHRemoteRunner()
    runner._active_job_id = "guided_20260101_000000_abc123"
    channel = _FakeChannel()
    runner._channel = channel
    calls = []
    runner._ssh = SimpleNamespace(
        is_connected=True,
        execute=lambda cmd, timeout=5: calls.append(cmd) or (0, "", ""),
    )
    assert runner.cancel(runner._active_job_id) is True
    assert runner._stop_requested is True
    assert runner._stop_event.is_set()
    assert runner._reader_stop.is_set()
    assert channel.closed is True
    assert "pkill -f" in calls[0]
    assert "guided_20260101_000000_abc123" in calls[0]
    # the pkill is followed by a liveness verification probe
    assert any("pgrep -f" in c for c in calls[1:])


def test_cancel_verifies_with_kill_probe_and_escalates():
    """A worker that survives pkill gets SIGKILLed; still alive → warn."""
    runner = WebSSHRemoteRunner()
    job_id = "guided_20260101_000000_def456"
    runner._active_job_id = job_id
    runner._channel = None
    events = []
    runner.subscribe(events.append)
    calls = []

    def fake_execute(cmd, timeout=5):
        calls.append(cmd)
        if cmd.startswith("pgrep"):
            return 0, "4321\n", ""
        if cmd.startswith("kill -0"):
            return 0, "", ""  # pid still alive
        return 0, "", ""

    runner._ssh = SimpleNamespace(is_connected=True, execute=fake_execute)
    assert runner.cancel(job_id) is True
    assert any(c.startswith("kill -0 4321") for c in calls)
    assert any(c.startswith("pkill -9") for c in calls)
    warnings = [
        e for e in events
        if "未能终止" in (e.to_dict()["payload"].get("message") or "")
    ]
    assert warnings


def test_cancel_no_escalation_when_worker_gone():
    runner = WebSSHRemoteRunner()
    job_id = "guided_20260101_000000_999999"
    runner._active_job_id = job_id
    runner._channel = None
    calls = []

    def fake_execute(cmd, timeout=5):
        calls.append(cmd)
        if cmd.startswith("pgrep"):
            return 0, "4321\n", ""
        if cmd.startswith("kill -0"):
            return 1, "", ""  # pid already gone
        return 0, "", ""

    runner._ssh = SimpleNamespace(is_connected=True, execute=fake_execute)
    assert runner.cancel(job_id) is True
    assert any(c.startswith("kill -0 4321") for c in calls)
    assert not any(c.startswith("pkill -9") for c in calls)


def test_cancel_tolerates_ssh_failures():
    runner = WebSSHRemoteRunner()
    runner._active_job_id = "job-x"
    runner._channel = None
    runner._ssh = SimpleNamespace(
        is_connected=True,
        execute=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert runner.cancel("job-x") is True  # never raises


# ---- remote live metrics feeding -----------------------------------------------------


def _build_service(fake_jm):
    import importlib

    ts = importlib.import_module("web.backend.app.training_service")
    svc = ts.WebTrainingService.__new__(ts.WebTrainingService)
    svc.jm = fake_jm
    svc.events = deque(maxlen=100)
    svc.seq = 0
    svc.lock = threading.RLock()
    from anylabeling.services.training_center.metrics.store import MetricStore

    svc.metric_store = MetricStore()
    svc.adapter = None
    svc.output_dir = None
    svc.active_job_id = None
    svc._epoch_marks = deque(maxlen=6)
    svc._eta_seconds = None
    return svc


def _epoch_event(job_id: str, epoch: int):
    from anylabeling.services.training_center.event_protocol import (
        TrainingEvent,
        TrainingEventType,
    )

    return TrainingEvent(
        schema_version=1,
        job_id=job_id,
        event_type=TrainingEventType.EPOCH_METRICS,
        timestamp=time.time(),
        payload={
            "epoch": epoch,
            "total_epochs": 3,
            "metrics": {
                "metrics/mAP50(B)": 0.5 + epoch * 0.1,
                "train/box_loss": 1.2 - epoch * 0.1,
                "lr/pg0": 0.001,
                "notes": "not-a-number",  # must be filtered out
            },
            "best_metric": None,
        },
        source="test",
    )


def test_remote_epoch_metrics_are_fed_into_metric_store():
    job = SimpleNamespace(job_id="job-remote", execution_mode="remote_ssh")
    fake_jm = SimpleNamespace(get_current_job=lambda: job)
    svc = _build_service(fake_jm)

    svc._on_event(_epoch_event("job-remote", 1))
    svc._on_event(_epoch_event("job-remote", 2))

    run = svc.metric_store.get_run("job-remote")
    assert run is not None
    assert len(run.samples) == 2
    sample = run.samples[0]
    assert sample.epoch == 1.0
    assert sample.total_epochs == 3
    assert "metrics/mAP50(B)" in sample.values
    assert "train/box_loss" in sample.values
    assert "notes" not in sample.values

    # the metrics endpoint must surface event-fed series (no local CSV)
    out = svc.metrics()
    names = {s["name"] for s in out["series"]}
    assert {"metrics/mAP50(B)", "train/box_loss", "lr/pg0"} <= names
    map_series = next(s for s in out["series"] if s["name"] == "metrics/mAP50(B)")
    assert map_series["group"] == "quality"
    assert map_series["points"] == [(1.0, pytest.approx(0.6)), (2.0, pytest.approx(0.7))]


def test_local_epoch_metrics_are_not_fed():
    """Local jobs rely on the CSV tailer — event feeding would duplicate."""
    job = SimpleNamespace(job_id="job-local", execution_mode="local")
    fake_jm = SimpleNamespace(get_current_job=lambda: job)
    svc = _build_service(fake_jm)

    svc._on_event(_epoch_event("job-local", 1))
    run = svc.metric_store.get_run("job-local")
    assert run is None or len(run.samples) == 0


# ---- PREPARING-phase stop: runner must release the slot for the next job ------


def _wait_for(cond, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


class _BlockingSFTP:
    """SFTP fake whose put() blocks until close() — a very slow upload."""

    def __init__(self, block: bool):
        self.block = block
        self.closed = False
        self.put_entered = threading.Event()

    def stat(self, path):
        raise IOError("not found")

    def mkdir(self, path):
        pass

    def put(self, local, remote):
        self.put_entered.set()
        if self.block:
            while not self.closed:
                time.sleep(0.05)
            raise IOError("sftp session closed")

    def close(self):
        self.closed = True


class _FakeStreamChannel:
    """Session channel that never produces output nor exits."""

    def __init__(self):
        self.closed = False
        self.command = None

    def exec_command(self, cmd):
        self.command = cmd

    def recv_ready(self):
        return False

    def recv_stderr_ready(self):
        return False

    def exit_status_ready(self):
        return False

    def close(self):
        self.closed = True


class _FakeSSHService:
    """Stands in for SSHConnectionService; generation 0 blocks on SFTP put."""

    generation = 0
    sftps = []

    def __init__(self):
        self.is_connected = False
        self._sftp = _BlockingSFTP(block=(type(self).generation == 0))
        type(self).sftps.append(self._sftp)
        self._client = SimpleNamespace(
            open_sftp=lambda: self._sftp,
            get_transport=lambda: SimpleNamespace(
                open_session=lambda: _FakeStreamChannel()
            ),
        )

    def connect(self, profile, password="", on_host_key=None):
        self.is_connected = True
        return True, "aa:bb:cc:dd"

    def execute(self, cmd, timeout=5):
        return 0, "", ""

    def close(self):
        self.is_connected = False
        self._sftp.close()


def _make_remote_job(job_id: str, tmp_path, status):
    import datetime as _dt

    from anylabeling.services.training_center.models import (
        TrainingJob,
        TrainingMode,
    )

    job = TrainingJob(job_id=job_id, mode=TrainingMode.GUIDED_ULTRALYTICS)
    job.status = status
    job.created_at = _dt.datetime.now()
    job.workspace = str(tmp_path)
    job.output_directory = tmp_path / "out"
    job.display_name = "t"
    job.execution_mode = "remote_ssh"
    job.remote_profile_id = "p1"
    job._session_password = ""
    return job


@pytest.fixture()
def remote_env(monkeypatch, tmp_path):
    """ProfileStore redirected to tmp + SSHConnectionService faked out."""
    from anylabeling.services.training_center.remote import storage
    from anylabeling.services.training_center.remote.models import (
        AuthMethod,
        RemoteProfile,
    )
    from anylabeling.services.training_center.remote.storage import ProfileStore
    from anylabeling.services.training_center.runners import (
        ssh_remote as ssh_remote_mod,
    )

    store = ProfileStore(tmp_path / "remote_profiles.json")
    monkeypatch.setattr(storage, "_store", store)
    store.save(
        RemoteProfile(
            profile_id="p1",
            name="gpu",
            host="192.168.1.10",
            port=22,
            username="trainer",
            auth_method=AuthMethod.SSH_KEY,
            private_key_path="",
            remote_workspace="/data/ws",
            remote_python="/usr/bin/python3",
        )
    )

    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "data.yaml").write_text("names: [cat]\n", encoding="utf-8")

    _FakeSSHService.generation = 0
    _FakeSSHService.sftps = []
    monkeypatch.setattr(ssh_remote_mod, "SSHConnectionService", _FakeSSHService)
    return SimpleNamespace(tmp_path=tmp_path, ds=ds)


def test_start_aborts_immediately_when_job_already_terminal(remote_env):
    """stop() landing before the daemon thread entered start() must not connect."""
    from anylabeling.services.training_center.models import TrainingStatus

    runner = WebSSHRemoteRunner()
    job = _make_remote_job("guided_x_aaaaaa", remote_env.tmp_path, TrainingStatus.STOPPED)
    ok, msg = runner.start(job, {"data": str(remote_env.ds / "data.yaml")})
    assert ok is False
    assert runner._active_job_id == ""  # slot never taken


def test_preparing_stop_releases_runner_and_next_remote_job_starts(remote_env):
    from anylabeling.services.training_center.models import TrainingStatus

    runner = WebSSHRemoteRunner()
    job1 = _make_remote_job(
        "guided_20260101_000000_aaaaaa", remote_env.tmp_path, TrainingStatus.PREPARING
    )
    ok, msg = runner.prepare(job1, {})
    assert ok, msg

    holder: dict = {}
    t = threading.Thread(
        target=lambda: holder.update(
            result=runner.start(
                job1, {"data": str(remote_env.ds / "data.yaml"), "model": ""}
            )
        ),
        daemon=True,
    )
    started = time.monotonic()
    t.start()
    sftp0 = _FakeSSHService.sftps[0]
    assert _wait_for(lambda: sftp0.put_entered.is_set()), "upload never started"

    # user hits stop during the (potentially multi-minute) upload phase
    assert runner.cancel(job1.job_id) is True
    t.join(timeout=10)
    assert not t.is_alive(), "start() did not abort after cancel"
    ok, _ = holder["result"]
    assert ok is False
    assert runner._active_job_id == ""  # slot released
    assert time.monotonic() - started < 60  # the review's release budget

    # the next remote job starts normally on the same runner instance
    _FakeSSHService.generation = 1
    job2 = _make_remote_job(
        "guided_20260101_000100_bbbbbb", remote_env.tmp_path, TrainingStatus.PREPARING
    )
    ok, msg = runner.prepare(job2, {})
    assert ok, msg
    ok, msg = runner.start(job2, {"data": str(remote_env.ds / "data.yaml"), "model": ""})
    assert ok is True, msg
    assert runner._active_job_id == job2.job_id

    # stopping the now-running job still emits STOPPED and releases the slot
    events = []
    runner.subscribe(events.append)
    assert runner.cancel(job2.job_id) is True
    assert _wait_for(lambda: runner._active_job_id == "")
    stopped = [e for e in events if e.to_dict()["event_type"] == "stopped"]
    assert stopped
