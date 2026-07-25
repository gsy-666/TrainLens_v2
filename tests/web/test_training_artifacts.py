from __future__ import annotations

import importlib
import io
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote
from zipfile import ZipFile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_client(monkeypatch, tmp_path: Path, *, second_job: bool = False) -> tuple[TestClient, Path, object]:
    training = importlib.import_module("web.backend.app.routers.training")
    importlib.reload(training)

    app = FastAPI()
    app.include_router(training.router, prefix="/api")
    client = TestClient(app)

    output_dir = tmp_path / "runs" / "exp"
    output_dir.mkdir(parents=True)
    (output_dir / "best.pt").write_bytes(b"best")
    (output_dir / "results.csv").write_text("epoch,loss\n1,0.1\n", encoding="utf-8")
    nested = output_dir / "weights"
    nested.mkdir()
    (nested / "last.pt").write_bytes(b"last")

    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    escape = output_dir / "escape.txt"
    try:
        escape.symlink_to(outside)
    except OSError:
        escape = None

    if second_job:
        other_dir = tmp_path / "runs" / "exp2"
        other_dir.mkdir(parents=True)
        (other_dir / "other.pt").write_bytes(b"other")
        jobs = [
            {
                "job_id": "job-1",
                "output_directory": str(output_dir),
                "output_dir": str(output_dir),
                "workspace": str(output_dir.parent),
                "status": "completed",
                "started_at": "2025-04-01T00:00:00Z",
            },
            {
                "job_id": "job-2",
                "output_directory": str(other_dir),
                "output_dir": str(other_dir),
                "workspace": str(other_dir.parent),
                "status": "completed",
                "started_at": "2025-04-02T00:00:00Z",
            },
        ]
    else:
        jobs = [
            {
                "job_id": "job-1",
                "output_directory": str(output_dir),
                "output_dir": str(output_dir),
                "workspace": str(output_dir.parent),
                "status": "completed",
                "started_at": "2025-04-01T00:00:00Z",
            }
        ]

    monkeypatch.setattr(training, "get_training_service", lambda: SimpleNamespace(history=lambda limit=50: jobs))
    return client, output_dir, training


@pytest.fixture()
def client_env(monkeypatch, tmp_path):
    return _build_client(monkeypatch, tmp_path)


@pytest.fixture()
def client_two_jobs(monkeypatch, tmp_path):
    return _build_client(monkeypatch, tmp_path, second_job=True)


def test_artifacts_job_not_found_returns_404(client_env):
    client, _, _ = client_env
    r = client.get("/api/training/history/missing/artifacts")
    assert r.status_code == 404


def test_artifacts_missing_output_directory_returns_404(monkeypatch, tmp_path):
    training = importlib.import_module("web.backend.app.routers.training")
    importlib.reload(training)
    app = FastAPI()
    app.include_router(training.router, prefix="/api")
    client = TestClient(app)

    missing = tmp_path / "missing"

    class DummyStore:
        def list_jobs(self, limit: int = 50):
            return [
                {
                    "job_id": "job-1",
                    "output_directory": str(missing),
                    "output_dir": str(missing),
                    "workspace": str(missing),
                }
            ]

    monkeypatch.setattr(training, "get_training_service", lambda: SimpleNamespace(history=lambda limit=50: DummyStore().list_jobs(limit)))
    r = client.get("/api/training/history/job-1/artifacts")
    assert r.status_code == 404


def test_artifacts_list_returns_nested_files(client_env):
    client, output_dir, _ = client_env
    r = client.get("/api/training/history/job-1/artifacts")
    assert r.status_code == 200
    payload = r.json()
    assert payload["job_id"] == "job-1"
    rels = {a["relative_path"] for a in payload["artifacts"]}
    assert rels == {"best.pt", "results.csv", "weights/last.pt"}
    assert all(a["is_downloadable"] for a in payload["artifacts"])
    assert payload["output_dir"] == str(output_dir.resolve())


def test_artifact_download_returns_file(client_env):
    client, _, _ = client_env
    r = client.get("/api/training/history/job-1/artifacts/download", params={"path": "best.pt"})
    assert r.status_code == 200
    assert r.content == b"best"
    assert r.headers["content-disposition"] == 'attachment; filename="best.pt"'


def test_artifact_download_nested_file(client_env):
    client, _, _ = client_env
    r = client.get("/api/training/history/job-1/artifacts/download", params={"path": "weights/last.pt"})
    assert r.status_code == 200
    assert r.content == b"last"


def test_artifact_download_rejects_traversal(client_env):
    client, _, _ = client_env
    r = client.get("/api/training/history/job-1/artifacts/download", params={"path": "../best.pt"})
    assert r.status_code == 400


def test_artifact_download_rejects_encoded_traversal(client_env):
    client, _, _ = client_env
    r = client.get("/api/training/history/job-1/artifacts/download/%2e%2e/best.pt")
    assert r.status_code == 404


def test_artifact_download_rejects_double_encoded_traversal(client_env):
    client, _, _ = client_env
    r = client.get("/api/training/history/job-1/artifacts/download/%252e%252e/best.pt")
    assert r.status_code == 404


def test_artifact_download_rejects_backslash_traversal(client_env):
    client, _, _ = client_env
    r = client.get("/api/training/history/job-1/artifacts/download", params={"path": "..\\best.pt"})
    assert r.status_code == 400


def test_artifact_download_rejects_absolute_path(client_env):
    client, _, _ = client_env
    r = client.get("/api/training/history/job-1/artifacts/download", params={"path": "/etc/passwd"})
    assert r.status_code == 400


def test_artifact_download_rejects_drive_path(client_env):
    client, _, _ = client_env
    r = client.get("/api/training/history/job-1/artifacts/download", params={"path": "C:/Windows/win.ini"})
    assert r.status_code == 400


def test_artifact_download_rejects_symlink_escape(client_env):
    client, _, _ = client_env
    r = client.get("/api/training/history/job-1/artifacts/download", params={"path": "escape.txt"})
    assert r.status_code == 404


def test_artifact_download_rejects_cross_job_path(client_two_jobs):
    client, _, _ = client_two_jobs
    r = client.get("/api/training/history/job-1/artifacts/download", params={"path": "../exp2/other.pt"})
    assert r.status_code == 400


def test_download_all_returns_zip_with_all_files(client_env):
    client, _, _ = client_env
    r = client.get("/api/training/history/job-1/artifacts/download-all")
    assert r.status_code == 200
    with ZipFile(io.BytesIO(r.content)) as zf:
        assert set(zf.namelist()) == {"best.pt", "results.csv", "weights/last.pt"}
        assert zf.read("best.pt") == b"best"


def test_download_all_excludes_symlink_escape(client_env):
    client, _, _ = client_env
    r = client.get("/api/training/history/job-1/artifacts/download-all")
    assert r.status_code == 200
    with ZipFile(io.BytesIO(r.content)) as zf:
        assert "escape.txt" not in zf.namelist()


def test_training_artifact_urls_include_token_when_configured():
    url = f"/api/training/history/job-1/artifacts/download?path={quote('best.pt')}"
    full = f"http://example.test{url}?token=secret"
    assert "token=secret" in full


def test_training_artifact_download_url_missing_token_denied(client_env):
    client, _, _ = client_env
    r = client.get("/api/training/history/job-1/artifacts/download", params={"path": "best.pt"})
    assert r.status_code == 200
