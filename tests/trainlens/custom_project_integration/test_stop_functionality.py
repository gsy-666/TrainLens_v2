"""Integration test for Stop functionality

Verifies:
- Stop transitions to STOPPED terminal state
- Stop is idempotent (second call returns False)
- Only one 'stopped' event is emitted
- JobManager clears current_job after STOPPED
"""

import sys
import os
import time
import uuid
from pathlib import Path
from typing import List

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QSignalSpy

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.services.training_center.job_manager import JobManager
from anylabeling.services.training_center.models import TrainingJob, TrainingMode, TrainingStatus
from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter
from anylabeling.services.training_center.event_protocol import TrainingEvent, TrainingEventType


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
            time.sleep(0.5)
        except:
            pass


@pytest.fixture
def custom_adapter():
    """Create CustomScriptAdapter instance"""
    adapter = CustomScriptAdapter()
    yield adapter

    # Cleanup
    if adapter.is_running():
        adapter.stop()
        # Wait for process to stop
        timeout = 10
        start = time.time()
        while adapter.is_running() and (time.time() - start) < timeout:
            time.sleep(0.1)


def test_stop_transitions_to_stopped(qapp, job_manager, custom_adapter, tmp_path):
    """Test that stopping a running job transitions to STOPPED state"""

    # Capture events
    captured_events: List[TrainingEvent] = []

    def capture_event(event: TrainingEvent):
        captured_events.append(event)

    job_manager.subscribe_events(capture_event)

    # Create long-running job (60 seconds)
    test_script = tmp_path / "test_script_long.py"
    test_script.write_text("""
import time
print("TRAINLENS_START_OK", flush=True)
for i in range(60):
    print(f"STEP={i+1}/60", flush=True)
    time.sleep(1)
print("TRAINLENS_FINISH_OK", flush=True)
""")

    job_id = str(uuid.uuid4())
    job = TrainingJob(
        job_id=job_id,
        mode=TrainingMode.CUSTOM_SCRIPT,
        workspace=test_script.parent,
        python_executable=Path(sys.executable),
        display_name="Long Test",
    )

    config = {
        'script_path': str(test_script),
        'arguments': [],
    }

    # Start job
    success, msg = job_manager.request_start(job, custom_adapter, config)
    assert success, f"Failed to start: {msg}"

    # Wait for process to actually start (process events)
    for _ in range(30):  # 3 seconds
        qapp.processEvents()
        time.sleep(0.1)

    # Verify job is running
    current_job = job_manager.get_current_job()
    assert current_job is not None
    assert current_job.status == TrainingStatus.RUNNING

    # Request stop
    stop_success = job_manager.request_stop()
    assert stop_success, "First stop request should succeed"

    # Status should be STOPPING or STOPPED (stop can complete very fast)
    assert current_job.status in (TrainingStatus.STOPPING, TrainingStatus.STOPPED), \
        f"Expected STOPPING or STOPPED, got {current_job.status}"

    # Wait for stop to process (process events for QTimer to fire)
    for _ in range(20):  # 2 seconds
        qapp.processEvents()
        time.sleep(0.1)
        if not custom_adapter.is_running():
            break

    # Second stop should be idempotent (return False)
    stop_again = job_manager.request_stop()
    assert not stop_again, "Second stop request should return False (idempotent)"

    assert not custom_adapter.is_running(), "Process should have stopped"

    # Wait a bit more for signal propagation (process events)
    for _ in range(5):  # 0.5 seconds
        qapp.processEvents()
        time.sleep(0.1)

    # Verify STOPPED event was emitted
    stopped_events = [e for e in captured_events if e.event_type == TrainingEventType.STOPPED]
    assert len(stopped_events) == 1, f"Expected exactly 1 STOPPED event, got {len(stopped_events)}"

    # Verify NO completed or failed events
    completed_events = [e for e in captured_events if e.event_type == TrainingEventType.COMPLETED]
    failed_events = [e for e in captured_events if e.event_type == TrainingEventType.FAILED]
    assert len(completed_events) == 0, "Should not emit COMPLETED for stopped job"
    assert len(failed_events) == 0, "Should not emit FAILED for stopped job"

    # Verify JobManager cleared current_job
    final_job = job_manager.get_current_job()
    assert final_job is None, "JobManager should clear current_job after STOPPED"

    # Cleanup
    job_manager.unsubscribe_events(capture_event)


def test_stop_before_process_starts(qapp, job_manager, custom_adapter):
    """Test stopping a job that hasn't started subprocess yet"""

    # Create job with invalid script (won't start)
    job_id = str(uuid.uuid4())
    job = TrainingJob(
        job_id=job_id,
        mode=TrainingMode.CUSTOM_SCRIPT,
        workspace=Path.cwd(),
        python_executable=Path(sys.executable),
        display_name="Invalid Test",
    )

    config = {
        'script_path': 'nonexistent_script.py',
        'arguments': [],
    }

    # Try to start (will fail)
    success, msg = job_manager.request_start(job, custom_adapter, config)
    assert not success, "Should fail to start with nonexistent script"

    # Verify no job is running
    current_job = job_manager.get_current_job()
    assert current_job is None

    # Try to stop (should return False)
    stop_result = job_manager.request_stop()
    assert not stop_result, "Stop should return False when no job is running"
