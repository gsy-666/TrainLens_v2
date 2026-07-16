"""Integration test using QSignalSpy for signal verification

Uses PyQt6's built-in QSignalSpy to properly wait for signals from ProcessManager.
This avoids the QEventLoop timing issues seen in other tests.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QSignalSpy

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
    yield jm
    # Force cleanup after test
    if jm.get_current_job() is not None:
        try:
            jm.request_stop()
            import time
            time.sleep(0.5)
        except:
            pass


@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace with test script"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    script_path = workspace / "train.py"
    script_path.write_text("""
import os
import sys
import time

print("TRAINLENS_START_OK", flush=True)
print("PYTHON=" + sys.executable, flush=True)
print("CWD=" + os.getcwd(), flush=True)
print("ARGS=" + repr(sys.argv[1:]), flush=True)

for i in range(3):
    print(f"STEP={i + 1}/3", flush=True)
    time.sleep(0.2)

print("TRAINLENS_FINISH_OK", flush=True)
""")

    return workspace, script_path


def test_qsignalspy_normal_completion(qapp, job_manager, temp_workspace):
    """Test normal completion using QSignalSpy"""
    workspace, script_path = temp_workspace

    print(f"\n[TEST] QSignalSpy normal completion")
    print(f"  Workspace: {workspace}")
    print(f"  Script: {script_path}")
    print(f"  Python: {sys.executable}")

    # Create adapter
    adapter = CustomScriptAdapter()

    # Create job
    job = TrainingJob(
        job_id="qspy-test-001",
        mode=TrainingMode.CUSTOM_SCRIPT,
        status=TrainingStatus.PREPARING,
        created_at=datetime.now(),
        workspace=workspace,
        display_name="QSignalSpy Test",
        framework="custom",
        python_executable=Path(sys.executable),
        command=[str(script_path), "--test-arg", "value"],
        metadata={}
    )

    # Capture events
    events = []
    stdout_lines = []

    def capture_event(event):
        events.append(event)
        event_type = event.event_type.value
        print(f"  Event: {event_type}")

        if event_type == 'console_output' and event.payload.get('stream') == 'stdout':
            stdout_lines.append(event.payload.get('message', ''))

    adapter.subscribe(capture_event)

    # Config
    config = {
        'script_path': str(script_path),
        'arguments': ['--test-arg', 'value'],
    }

    # Start via JobManager
    success, message = job_manager.request_start(job=job, adapter=adapter, config=config)

    print(f"  Start result: success={success}")
    if not success:
        print(f"  Error: {message}")

    assert success is True, f"Failed to start: {message}"

    # Wait for process to complete by polling events
    print(f"  Waiting for process to complete...")
    import time
    timeout = 15
    elapsed = 0
    while adapter.is_running() and elapsed < timeout:
        qapp.processEvents()
        time.sleep(0.1)
        elapsed += 0.1

    # Process remaining events
    for _ in range(20):
        qapp.processEvents()
        time.sleep(0.05)

    # Verify process completed
    assert not adapter.is_running(), "Process should have completed"

    # Verify events
    event_types = [e.event_type.value for e in events]
    print(f"  Event types: {event_types}")

    assert 'process_started' in event_types, "Missing process_started"
    assert 'completed' in event_types or 'failed' in event_types, "Missing completion event"

    # Verify stdout
    stdout_text = '\n'.join(stdout_lines)
    print(f"  Stdout lines: {len(stdout_lines)}")
    print(f"  Stdout sample: {stdout_text[:200]}")

    assert "TRAINLENS_START_OK" in stdout_text, "Missing START marker"
    assert "TRAINLENS_FINISH_OK" in stdout_text, "Missing FINISH marker"
    assert sys.executable in stdout_text, "Missing python path"
    assert "['--test-arg', 'value']" in stdout_text, "Missing arguments"

    # Verify completion event
    completed_events = [e for e in events if e.event_type.value == 'completed']
    assert len(completed_events) > 0, "Should have completed event"
    assert completed_events[0].payload.get('exit_code') == 0, "Exit code should be 0"

    print("[PASS] QSignalSpy normal completion test PASSED!")


def test_qsignalspy_nonzero_exit(qapp, job_manager, tmp_path):
    """Test non-zero exit using QSignalSpy"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    script_path = workspace / "fail.py"
    script_path.write_text("""
import sys
print("TRAINLENS_START_OK", flush=True)
print("ERROR: Simulated failure", flush=True)
sys.exit(1)
""")

    print(f"\n[TEST] QSignalSpy non-zero exit")

    adapter = CustomScriptAdapter()

    job = TrainingJob(
        job_id="qspy-test-002",
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

    # Wait for completion by polling
    print(f"  Waiting for completion...")
    import time
    timeout = 10
    elapsed = 0
    while adapter.is_running() and elapsed < timeout:
        qapp.processEvents()
        time.sleep(0.1)
        elapsed += 0.1

    # Process remaining events
    for _ in range(20):
        qapp.processEvents()
        time.sleep(0.05)

    # Verify failed event
    event_types = [e.event_type.value for e in events]
    print(f"  Event types: {event_types}")

    assert 'failed' in event_types, "Should have failed event"

    failed_events = [e for e in events if e.event_type.value == 'failed']
    assert failed_events[0].payload.get('exit_code') == 1, "Exit code should be 1"

    print("[PASS] QSignalSpy non-zero exit test PASSED!")


def test_qsignalspy_invalid_python_path(qapp, job_manager, tmp_path):
    """Test invalid Python path detection"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    script_path = workspace / "train.py"
    script_path.write_text("print('hello')")

    print(f"\n[TEST] QSignalSpy invalid Python path")

    adapter = CustomScriptAdapter()

    job = TrainingJob(
        job_id="qspy-test-003",
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

    print(f"  Result: success={success}")
    print(f"  Message: {message[:200]}")

    assert success is False, "Should fail for invalid python"
    assert "Python executable not found" in message, "Should have diagnostic"
    assert "python" in message.lower(), "Should mention python"
    assert str(script_path) in message, "Should include script path"

    print("[PASS] QSignalSpy invalid Python path test PASSED!")


def test_qsignalspy_with_arguments(qapp, job_manager, temp_workspace):
    """Test script with various argument types"""
    workspace, script_path = temp_workspace

    print(f"\n[TEST] QSignalSpy with arguments")

    adapter = CustomScriptAdapter()

    job = TrainingJob(
        job_id="qspy-test-004",
        mode=TrainingMode.CUSTOM_SCRIPT,
        status=TrainingStatus.PREPARING,
        created_at=datetime.now(),
        workspace=workspace,
        display_name="Arguments Test",
        framework="custom",
        python_executable=Path(sys.executable),
        command=[str(script_path), "--epochs", "10", "--name", "test model"],
        metadata={}
    )

    stdout_lines = []

    def capture_stdout(event):
        if event.event_type.value == 'console_output' and event.payload.get('stream') == 'stdout':
            stdout_lines.append(event.payload.get('message', ''))

    adapter.subscribe(capture_stdout)

    config = {
        'script_path': str(script_path),
        'arguments': ['--epochs', '10', '--name', 'test model'],
    }

    success, message = job_manager.request_start(job=job, adapter=adapter, config=config)
    assert success is True, f"Start should succeed: {message}"

    # Wait for completion by polling
    import time
    timeout = 15
    elapsed = 0
    while adapter.is_running() and elapsed < timeout:
        qapp.processEvents()
        time.sleep(0.1)
        elapsed += 0.1

    # Process remaining events
    for _ in range(20):
        qapp.processEvents()
        time.sleep(0.05)

    stdout_text = '\n'.join(stdout_lines)
    print(f"  Stdout: {stdout_text[:300]}")

    assert "['--epochs', '10', '--name', 'test model']" in stdout_text, "Arguments not passed correctly"

    print("[PASS] QSignalSpy arguments test PASSED!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
