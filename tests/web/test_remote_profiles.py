"""Tests for /api/remote/profiles CRUD + test/confirm-host-key/diagnostics
endpoints, and remote execution params in guided start.

No real SSH server exists in CI — SSHConnectionService.connect and
run_remote_diagnostics are mocked. The ProfileStore singleton is
redirected to a tmp path per test."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from anylabeling.services.training_center.remote import storage
from anylabeling.services.training_center.remote.models import (
    AuthMethod,
    RemoteProfile,
)
from anylabeling.services.training_center.remote.storage import ProfileStore


@pytest.fixture()
def store(tmp_path, monkeypatch) -> ProfileStore:
    """Redirect the ProfileStore singleton to a tmp file."""
    s = ProfileStore(tmp_path / "remote_profiles.json")
    monkeypatch.setattr(storage, "_store", s)
    return s


@pytest.fixture()
def client(store):
    remote = importlib.import_module("web.backend.app.routers.remote")
    importlib.reload(remote)
    app = FastAPI()
    app.include_router(remote.router, prefix="/api")
    return TestClient(app)


def _payload(**over) -> dict:
    p = {
        "name": "gpu-box",
        "host": "192.168.1.10",
        "port": 22,
        "username": "trainer",
        "auth_method": "ssh_key",
        "private_key_path": "C:/Users/x/.ssh/id_rsa",
        "remote_workspace": "/data/trainlens",
        "remote_python": "/usr/bin/python3",
    }
    p.update(over)
    return p


def _make_profile(store: ProfileStore, **over) -> RemoteProfile:
    p = RemoteProfile(
        profile_id="p1",
        name="gpu-box",
        host="192.168.1.10",
        port=22,
        username="trainer",
        auth_method=AuthMethod.SSH_KEY,
        private_key_path="/keys/id_rsa",
        remote_workspace="/data/trainlens",
        remote_python="/usr/bin/python3",
    )
    for k, v in over.items():
        setattr(p, k, v)
    store.save(p)
    return p


# ---- CRUD ----------------------------------------------------------------------


def test_list_profiles_empty(client):
    r = client.get("/api/remote/profiles")
    assert r.status_code == 200
    assert r.json() == {"profiles": []}


def test_profile_crud_full_path(client, store):
    # create
    r = client.post("/api/remote/profiles", json=_payload())
    assert r.status_code == 200
    profile = r.json()["profile"]
    pid = profile["profile_id"]
    assert pid
    assert profile["host"] == "192.168.1.10"
    assert "password" not in profile  # never serialized

    # list
    r = client.get("/api/remote/profiles")
    assert [p["profile_id"] for p in r.json()["profiles"]] == [pid]

    # update
    r = client.put(f"/api/remote/profiles/{pid}", json=_payload(name="renamed", port=2222))
    assert r.status_code == 200
    assert r.json()["profile"]["name"] == "renamed"
    assert store.get(pid).port == 2222

    # delete
    r = client.delete(f"/api/remote/profiles/{pid}")
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    assert store.get(pid) is None


def test_create_profile_rejects_bad_auth_method(client):
    r = client.post("/api/remote/profiles", json=_payload(auth_method="kerberos"))
    assert r.status_code == 400


def test_create_profile_rejects_empty_host(client):
    r = client.post("/api/remote/profiles", json=_payload(host="  "))
    assert r.status_code == 400


def test_update_missing_profile_returns_404(client):
    r = client.put("/api/remote/profiles/nope", json=_payload())
    assert r.status_code == 404


def test_delete_missing_profile_returns_404(client):
    r = client.delete("/api/remote/profiles/nope")
    assert r.status_code == 404


# ---- test connection + host key confirmation -------------------------------------


def _fake_connect_factory(presented="aa:bb:cc:dd"):
    """Mimic the real connect() host-key flow with a fixed server key."""

    def fake_connect(self, profile, password="", on_host_key=None):
        saved = (profile.known_host_fingerprint or "").strip()
        if saved:
            if saved == presented:
                return True, presented
            if on_host_key:
                trust, _save = on_host_key(profile.host, presented, "ssh-ed25519")
                if trust:
                    return True, presented
            return False, "SSH error: host key changed"
        if on_host_key:
            trust, _save = on_host_key(profile.host, presented, "ssh-ed25519")
            if trust:
                return True, presented
        return False, f"SSH error: Unknown host {profile.host}"

    return fake_connect


def _patch_ssh(monkeypatch, connect_impl):
    ssh_service = importlib.import_module(
        "anylabeling.services.training_center.remote.ssh_service"
    )
    monkeypatch.setattr(ssh_service.SSHConnectionService, "connect", connect_impl)
    # the real service opens no resources in __init__, close() stays as-is


def test_test_endpoint_success_with_saved_fingerprint(client, store, monkeypatch):
    _make_profile(store, known_host_fingerprint="aa:bb:cc:dd")
    _patch_ssh(monkeypatch, _fake_connect_factory())
    r = client.post("/api/remote/profiles/p1/test", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["fingerprint"] == "aa:bb:cc:dd"
    assert body["need_host_key_confirm"] is False


def test_test_endpoint_auth_failure(client, store, monkeypatch):
    _make_profile(store, known_host_fingerprint="aa:bb:cc:dd")

    def fake_connect(self, profile, password="", on_host_key=None):
        return False, "Authentication failed. Check username, key, or password."

    _patch_ssh(monkeypatch, fake_connect)
    r = client.post("/api/remote/profiles/p1/test", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["need_host_key_confirm"] is False
    assert "Authentication failed" in body["error"]


def test_test_endpoint_unknown_host_key_then_confirm(client, store, monkeypatch):
    _make_profile(store)  # no saved fingerprint
    _patch_ssh(monkeypatch, _fake_connect_factory("ff:ee:dd:cc"))

    # first test: fingerprint unknown → ask for confirmation
    r = client.post("/api/remote/profiles/p1/test", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["need_host_key_confirm"] is True
    assert body["fingerprint"] == "ff:ee:dd:cc"

    # confirm the fingerprint → persisted on the profile
    r = client.post(
        "/api/remote/profiles/p1/confirm-host-key",
        json={"fingerprint": "ff:ee:dd:cc"},
    )
    assert r.status_code == 200
    assert r.json()["saved"] is True
    assert store.get("p1").known_host_fingerprint == "ff:ee:dd:cc"

    # second test: now trusted → ok
    r = client.post("/api/remote/profiles/p1/test", json={})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_test_endpoint_fingerprint_mismatch_is_flagged(client, store, monkeypatch):
    _make_profile(store, known_host_fingerprint="00:11:22:33")
    _patch_ssh(monkeypatch, _fake_connect_factory("ff:ee:dd:cc"))
    r = client.post("/api/remote/profiles/p1/test", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["need_host_key_confirm"] is False
    assert "不一致" in body["error"]


def test_confirm_host_key_rejects_empty_fingerprint(client, store):
    _make_profile(store)
    r = client.post("/api/remote/profiles/p1/confirm-host-key", json={"fingerprint": " "})
    assert r.status_code == 400


def test_test_endpoint_missing_profile_returns_404(client):
    r = client.post("/api/remote/profiles/nope/test", json={})
    assert r.status_code == 404


# ---- diagnostics -------------------------------------------------------------------


def _diag_items(*, cuda: bool):
    from anylabeling.services.training_center.remote.models import (
        DiagnosticItem,
        DiagnosticStatus,
    )

    gpus = (
        [{"index": 0, "name": "RTX 4090", "total_memory_gb": 24.0, "compute_capability": "8.9"}]
        if cuda
        else []
    )
    return [
        DiagnosticItem(stage="os", label="Operating System", status=DiagnosticStatus.PASS,
                       message="Linux gpu-box 6.8"),
        DiagnosticItem(stage="pytorch", label="PyTorch", status=DiagnosticStatus.PASS,
                       message="PyTorch 2.4 · CUDA 12.4",
                       details={"installed": True, "version": "2.4.0",
                                "cuda_available": cuda, "gpus": gpus}),
        DiagnosticItem(stage="ultralytics", label="Ultralytics", status=DiagnosticStatus.PASS,
                       message="Version 8.3.0"),
    ]


def _patch_diagnostics(monkeypatch, items):
    diag = importlib.import_module(
        "anylabeling.services.training_center.remote.diagnostics"
    )
    monkeypatch.setattr(diag, "run_remote_diagnostics", lambda *a, **k: items)
    _patch_ssh(monkeypatch, _fake_connect_factory())


def test_diagnostics_gpu_recommends_device_0(client, store, monkeypatch):
    _make_profile(store, known_host_fingerprint="aa:bb:cc:dd")
    _patch_diagnostics(monkeypatch, _diag_items(cuda=True))
    r = client.post("/api/remote/profiles/p1/diagnostics", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["recommended_device"] == "0"
    assert body["gpus"][0]["name"] == "RTX 4090"
    labels = [i["label"] for i in body["items"]]
    assert "PyTorch" in labels and "Ultralytics" in labels
    assert all(i["status"] in ("PASS", "WARNING", "ERROR") for i in body["items"])


def test_diagnostics_cpu_only_recommends_cpu(client, store, monkeypatch):
    _make_profile(store, known_host_fingerprint="aa:bb:cc:dd")
    _patch_diagnostics(monkeypatch, _diag_items(cuda=False))
    r = client.post("/api/remote/profiles/p1/diagnostics", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["recommended_device"] == "cpu"
    assert body["gpus"] == []


def test_diagnostics_connect_failure_returns_400(client, store, monkeypatch):
    _make_profile(store, known_host_fingerprint="aa:bb:cc:dd")

    def fake_connect(self, profile, password="", on_host_key=None):
        return False, "Connection error: timed out"

    _patch_ssh(monkeypatch, fake_connect)
    r = client.post("/api/remote/profiles/p1/diagnostics", json={})
    assert r.status_code == 400
    assert "timed out" in r.json()["detail"]


# ---- guided start: remote params -----------------------------------------------------


def _guided_payload(**over) -> dict:
    p = {
        "task": "detect",
        "model": "yolov8n.pt",
        "data": "data.yaml",
        "project": "runs",
        "name": "train",
    }
    p.update(over)
    return p


def test_guided_start_accepts_remote_params(monkeypatch):
    """Router passes execution_mode/remote_profile_id through to the service."""
    training = importlib.import_module("web.backend.app.routers.training")
    importlib.reload(training)
    captured = {}

    def fake_start(params):
        captured.update(params)
        return {"job": None, "running": True, "output_dir": None, "eta_seconds": None}

    app = FastAPI()
    app.include_router(training.router, prefix="/api")
    monkeypatch.setattr(
        training,
        "get_training_service",
        lambda: SimpleNamespace(start_guided=fake_start),
    )
    client = TestClient(app)
    r = client.post(
        "/api/training/guided/start",
        json=_guided_payload(
            execution_mode="remote",
            remote_profile_id="p1",
            remote_password="s3cret",
        ),
    )
    assert r.status_code == 200
    assert captured["execution_mode"] == "remote"
    assert captured["remote_profile_id"] == "p1"
    assert captured["remote_password"] == "s3cret"


def test_guided_start_defaults_to_local(monkeypatch):
    training = importlib.import_module("web.backend.app.routers.training")
    importlib.reload(training)
    captured = {}

    def fake_start(params):
        captured.update(params)
        return {"job": None, "running": True, "output_dir": None, "eta_seconds": None}

    app = FastAPI()
    app.include_router(training.router, prefix="/api")
    monkeypatch.setattr(
        training,
        "get_training_service",
        lambda: SimpleNamespace(start_guided=fake_start),
    )
    client = TestClient(app)
    r = client.post("/api/training/guided/start", json=_guided_payload())
    assert r.status_code == 200
    assert captured["execution_mode"] == "local"
    assert captured["remote_profile_id"] is None


# ---- WebTrainingService.start_guided remote wiring ------------------------------------


def _build_service(monkeypatch, fake_jm):
    """WebTrainingService without __init__ (no JobManager singleton side effects)."""
    import threading
    from collections import deque

    ts = importlib.import_module("web.backend.app.training_service")
    monkeypatch.setattr(ts, "UltralyticsAdapter", lambda: SimpleNamespace())
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


class _FakeJM:
    def __init__(self):
        self.reserved = None
        self.started = None

    def get_current_job(self):
        return None

    def subscribe_events(self, cb):
        pass

    def reserve_job(self, job, adapter):
        self.reserved = job
        return True, "ok"

    def start_reserved_job(self, job_id, config):
        self.started = (job_id, config)
        return True, "ok"


def _wait_for(cond, timeout=5.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_start_guided_remote_sets_job_fields(store, monkeypatch, tmp_path):
    _make_profile(
        store,
        auth_method=AuthMethod.PASSWORD,
        remote_workspace="/data/ws",
        remote_python="/usr/bin/python3",
    )
    fake_jm = _FakeJM()
    svc = _build_service(monkeypatch, fake_jm)

    out = svc.start_guided(
        _guided_payload(
            project=str(tmp_path / "runs"),
            execution_mode="remote",  # alias → remote_ssh
            remote_profile_id="p1",
            remote_password="s3cret",
        )
    )
    assert out["running"] is False  # fake JM has no current job; reservation happened below
    job = fake_jm.reserved
    assert job is not None
    assert job.execution_mode == "remote_ssh"
    assert job.remote_profile_id == "p1"
    assert job.remote_host == "192.168.1.10"
    assert job.remote_workspace == "/data/ws"
    assert job.remote_python == "/usr/bin/python3"
    assert job._session_password == "s3cret"
    # remote start phase runs on a daemon thread
    assert _wait_for(lambda: fake_jm.started is not None)
    assert fake_jm.started[0] == job.job_id
    assert fake_jm.started[1]["data"] == "data.yaml"


def test_start_guided_remote_missing_profile(store, monkeypatch, tmp_path):
    svc = _build_service(monkeypatch, _FakeJM())
    with pytest.raises(RuntimeError, match="不存在"):
        svc.start_guided(
            _guided_payload(
                project=str(tmp_path / "runs"),
                execution_mode="remote_ssh",
                remote_profile_id="ghost",
            )
        )


def test_start_guided_remote_password_required(store, monkeypatch, tmp_path):
    _make_profile(store, auth_method=AuthMethod.PASSWORD)
    svc = _build_service(monkeypatch, _FakeJM())
    with pytest.raises(RuntimeError, match="密码"):
        svc.start_guided(
            _guided_payload(
                project=str(tmp_path / "runs"),
                execution_mode="remote_ssh",
                remote_profile_id="p1",
            )
        )


def test_start_guided_remote_requires_workspace(store, monkeypatch, tmp_path):
    _make_profile(store, remote_workspace="")
    svc = _build_service(monkeypatch, _FakeJM())
    with pytest.raises(RuntimeError, match="remote_workspace|远程工作目录"):
        svc.start_guided(
            _guided_payload(
                project=str(tmp_path / "runs"),
                execution_mode="remote_ssh",
                remote_profile_id="p1",
            )
        )


def test_run_preflight_remote_short_circuits(monkeypatch):
    svc = _build_service(monkeypatch, _FakeJM())
    out = svc.run_preflight(_guided_payload(execution_mode="remote_ssh"))
    assert out["can_start"] is True
    assert len(out["issues"]) == 1
    assert "远程" in out["issues"][0]["message"]


# ---- host/port change invalidates the saved host-key fingerprint (m12) ------------


def test_update_profile_host_change_clears_fingerprint(client, store):
    _make_profile(store, known_host_fingerprint="aa:bb:cc:dd")
    r = client.put("/api/remote/profiles/p1", json=_payload(host="10.0.0.2"))
    assert r.status_code == 200
    assert store.get("p1").known_host_fingerprint == ""


def test_update_profile_port_change_clears_fingerprint(client, store):
    _make_profile(store, known_host_fingerprint="aa:bb:cc:dd")
    r = client.put("/api/remote/profiles/p1", json=_payload(port=2222))
    assert r.status_code == 200
    assert store.get("p1").known_host_fingerprint == ""


def test_update_profile_same_endpoint_keeps_fingerprint(client, store):
    _make_profile(store, known_host_fingerprint="aa:bb:cc:dd")
    r = client.put("/api/remote/profiles/p1", json=_payload(name="renamed"))
    assert r.status_code == 200
    assert store.get("p1").known_host_fingerprint == "aa:bb:cc:dd"


# ---- proxy field round-trip + worker command env prefix ---------------------------


def test_proxy_field_round_trip(client, store):
    r = client.post("/api/remote/profiles", json=_payload(proxy="http://127.0.0.1:7890"))
    assert r.status_code == 200
    pid = r.json()["profile"]["profile_id"]
    assert store.get(pid).proxy == "http://127.0.0.1:7890"
    listed = client.get("/api/remote/profiles").json()["profiles"]
    assert listed[0]["proxy"] == "http://127.0.0.1:7890"

    r2 = client.put(f"/api/remote/profiles/{pid}", json=_payload(proxy=""))
    assert r2.status_code == 200
    assert store.get(pid).proxy == ""


def test_proxy_defaults_empty_for_legacy_payload(client, store):
    r = client.post("/api/remote/profiles", json=_payload())
    assert r.status_code == 200
    assert r.json()["profile"]["proxy"] == ""
