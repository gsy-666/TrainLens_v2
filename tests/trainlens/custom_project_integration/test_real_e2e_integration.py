"""Real end-to-end integration test for Custom Project launch

Tests complete chain: JobManager → CustomScriptAdapter → ProcessManager → subprocess
Uses REAL PyQt6 and QApplication (ProcessWatcher requires real Qt)
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

import pytest
from PyQt6.QtWidgets import QApplication

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.services.training_center.job_manager import JobManager
from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter
from anylabeling.services.training_center.models import TrainingJob, TrainingMode, TrainingStatus


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for real Qt tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def job_manager():
    """Create JobManager and ensure cleanup after test"""
    jm = JobManager()

    # Force cleanup any previous state
    if jm._current_adapter:
        try:
            jm._current_adapter.stop()
        except:
            pass
        jm._current_adapter.unsubscribe(jm._on_adapter_event)
    jm._current_adapter = None
    jm._current_job = None

    yield jm

    # Cleanup after test
    if jm._current_adapter:
        try:
            jm._current_adapter.stop()
            time.sleep(0.5)
        except:
            pass
        jm._current_adapter.unsubscribe(jm._on_adapter_event)
    jm._current_adapter = None
    jm._current_job = None


def wait_for_completion(qapp, adapter, max_wait=50):
    """Wait for adapter to finish and process events for signal propagation"""
    for _ in range(max_wait):
        qapp.processEvents()
        time.sleep(0.1)
        if not adapter.is_running():
            break

    # Continue processing events after process stops to ensure signals propagate
    for _ in range(10):
        qapp.processEvents()
        time.sleep(0.1)


def test_full_chain_successful_launch(qapp, job_manager, tmp_path):
    """Test complete chain: JobManager → Adapter → ProcessManager → subprocess"""
    workspace = tmp_path
    script_path = workspace / "train.py"
    script_path.write_text("""
import sys
import time
print("TRAINLENS_START_OK", flush=True)
print("PYTHON=" + sys.executable, flush=True)
print("CWD=" + sys.argv[0], flush=True)
print("ARGS=" + repr(sys.argv[1:]), flush=True)
time.sleep(0.5)
print("TRAINLENS_FINISH_OK", flush=True)
""")

    # Create TrainingJob
    job = TrainingJob(
        job_id="test-e2e-001",
        mode=TrainingMode.CUSTOM_SCRIPT,
        status=TrainingStatus.PREPARING,
        created_at=datetime.now(),
        workspace=workspace,
        display_name="E2E Test",
        framework="custom",
        python_executable=Path(sys.executable),
        command=[str(script_path)],
        metadata={}
    )

    # Create real adapter
    adapter = CustomScriptAdapter()

    # Capture events
    events = []
    def capture_event(event):
        events.append(event)

    adapter.subscribe(capture_event)

    # Config for adapter
    config = {
        'script_path': str(script_path),
        'arguments': ['--test-arg', 'value'],
    }

    # Start via JobManager
    success, message = job_manager.request_start(job=job, adapter=adapter, config=config)

    assert success is True, f"Start failed: {message}"

    # Wait for process to complete with signal propagation
    wait_for_completion(qapp, adapter)

    # Check process completed
    assert not adapter.is_running(), "Process should have completed"

    # Verify events
    event_types = [e.event_type.value for e in events]

    # Should have: process_started, console_output (multiple), completed
    assert 'process_started' in event_types, "Missing process_started event"
    assert 'completed' in event_types or 'failed' in event_types, "Missing completion event"

    # Check console output
    stdout_events = [e for e in events if e.event_type.value == 'console_output' and e.payload.get('stream') == 'stdout']
    stdout_text = '\n'.join([e.payload.get('message', '') for e in stdout_events])

    assert "TRAINLENS_START_OK" in stdout_text, "Missing START marker"
    assert "TRAINLENS_FINISH_OK" in stdout_text, "Missing FINISH marker"

    # Check JobManager released job
    current_job = job_manager.get_current_job()
    assert current_job is None, "Job should be released after completion"


def test_full_chain_nonzero_exit(qapp, job_manager, tmp_path):
    """Test complete chain with non-zero exit code"""
    workspace = tmp_path
    script_path = workspace / "fail.py"
    script_path.write_text("""
import sys
print("TRAINLENS_START_OK", flush=True)
print("ERROR: Simulated failure", flush=True)
sys.exit(1)
""")

    adapter = CustomScriptAdapter()

    job = TrainingJob(
        job_id="test-e2e-002",
        mode=TrainingMode.CUSTOM_SCRIPT,
        status=TrainingStatus.PREPARING,
        created_at=datetime.now(),
        workspace=workspace,
        display_name="Failure Test",
        framework="custom",
        python_executable=Path(sys.executable),
        command=[str(script_path)],
        metadata={}
    )

    events = []
    adapter.subscribe(lambda e: events.append(e))

    config = {'script_path': str(script_path), 'arguments': []}

    success, message = job_manager.request_start(job=job, adapter=adapter, config=config)
    assert success is True, f"Start should succeed: {message}"

    # Wait for completion with signal propagation
    wait_for_completion(qapp, adapter)

    # Check failed event
    event_types = [e.event_type.value for e in events]

    assert 'failed' in event_types, "Should have failed event for non-zero exit"

    failed_events = [e for e in events if e.event_type.value == 'failed']
    if failed_events:
        assert failed_events[0].payload.get('exit_code') == 1


def test_full_chain_invalid_python(job_manager, tmp_path):
    """Test complete chain with invalid Python executable"""
    workspace = tmp_path
    script_path = workspace / "train.py"
    script_path.write_text("print('hello')")

    adapter = CustomScriptAdapter()

    job = TrainingJob(
        job_id="test-e2e-003",
        mode=TrainingMode.CUSTOM_SCRIPT,
        status=TrainingStatus.PREPARING,
        created_at=datetime.now(),
        workspace=workspace,
        display_name="Invalid Python Test",
        framework="custom",
        python_executable=Path("/nonexistent/python.exe"),
        command=[str(script_path)],
        metadata={}
    )

    config = {'script_path': str(script_path), 'arguments': []}

    success, message = job_manager.request_start(job=job, adapter=adapter, config=config)

    assert success is False, "Should fail for invalid python"
    assert "Python executable not found" in message, "Should have diagnostic message"
    assert "nonexistent" in message and "python" in message, "Should include python path"
    assert str(script_path) in message, "Should include script path"
    assert str(workspace) in message, "Should include workspace"


def test_full_chain_invalid_script(job_manager, tmp_path):
    """Test complete chain with invalid script path"""
    workspace = tmp_path
    nonexistent_script = workspace / "nonexistent.py"

    adapter = CustomScriptAdapter()

    job = TrainingJob(
        job_id="test-e2e-004",
        mode=TrainingMode.CUSTOM_SCRIPT,
        status=TrainingStatus.PREPARING,
        created_at=datetime.now(),
        workspace=workspace,
        display_name="Invalid Script Test",
        framework="custom",
        python_executable=Path(sys.executable),
        command=[str(nonexistent_script)],
        metadata={}
    )

    config = {'script_path': str(nonexistent_script), 'arguments': ['--test']}

    success, message = job_manager.request_start(job=job, adapter=adapter, config=config)

    assert success is False, "Should fail for invalid script"
    assert "Script file not found" in message, "Should have diagnostic message"
    assert str(nonexistent_script) in message, "Should include script path"
    assert "--test" in message, "Should include arguments"
