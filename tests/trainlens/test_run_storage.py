"""Test run storage"""

import json
from pathlib import Path
from datetime import datetime

from anylabeling.services.run_monitor.run_storage import RunStorage
from anylabeling.services.run_monitor import Run, RunStatus, TrainingEvent


def test_run_storage_initialize(tmp_path):
    """Test storage initialization"""
    storage = RunStorage(tmp_path)
    success = storage.initialize()

    assert success
    assert (tmp_path / ".trainlens").exists()
    assert (tmp_path / ".trainlens" / "runs").exists()


def test_save_and_load_run(tmp_path):
    """Test saving and loading run metadata"""
    storage = RunStorage(tmp_path)
    storage.initialize()

    run = Run(
        run_id="test_run",
        workspace_path=tmp_path,
        script_path=tmp_path / "train.py",
        python_path=Path("/usr/bin/python3"),
        status=RunStatus.COMPLETED,
        start_time=datetime.now(),
    )

    success = storage.save_run(run)
    assert success

    loaded_run = storage.load_run("test_run")
    assert loaded_run is not None
    assert loaded_run.run_id == "test_run"
    assert loaded_run.status == RunStatus.COMPLETED


def test_save_console_line(tmp_path):
    """Test saving console output"""
    storage = RunStorage(tmp_path)
    storage.initialize()

    run_id = "test_run"
    lines = ["Line 1", "Line 2", "Line 3"]

    for line in lines:
        storage.save_console_line(run_id, line)

    console_file = tmp_path / ".trainlens" / "runs" / run_id / "console.log"
    assert console_file.exists()

    content = console_file.read_text()
    for line in lines:
        assert line in content


def test_save_event(tmp_path):
    """Test saving training events"""
    storage = RunStorage(tmp_path)
    storage.initialize()

    event = TrainingEvent(
        schema_version=1,
        run_id="test_run",
        event="epoch_metrics",
        timestamp=123456.0,
        payload={"epoch": 1, "loss": 0.5}
    )

    success = storage.save_event(event)
    assert success

    events_file = tmp_path / ".trainlens" / "runs" / "test_run" / "events.jsonl"
    assert events_file.exists()

    # Parse JSONL
    with open(events_file) as f:
        lines = f.readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["run_id"] == "test_run"
        assert parsed["event"] == "epoch_metrics"


def test_save_resource_sample(tmp_path):
    """Test saving resource samples"""
    storage = RunStorage(tmp_path)
    storage.initialize()

    sample = {
        "timestamp": 123456.0,
        "process": {"cpu_percent": 50.0, "memory_percent": 30.0},
        "system": {"cpu_percent": 60.0, "memory_percent": 40.0}
    }

    success = storage.save_resource_sample("test_run", sample)
    assert success

    resources_file = tmp_path / ".trainlens" / "runs" / "test_run" / "resources.jsonl"
    assert resources_file.exists()


def test_storage_handles_errors_gracefully():
    """Test that storage handles errors without crashing"""
    # Use invalid path
    storage = RunStorage(Path("/invalid/path"))

    # Should return False but not crash
    success = storage.initialize()
    # May or may not succeed depending on permissions
    # Just ensure it doesn't crash
    assert isinstance(success, bool)
