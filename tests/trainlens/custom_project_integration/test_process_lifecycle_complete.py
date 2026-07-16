"""Complete process lifecycle tests with ProcessWatcher architecture

Tests all 8 scenarios required for verification:
1. Natural completion (exit code 0)
2. Non-zero exit (process failure)
3. User Stop (long-running task)
4. Process already exited when Stop clicked
5. Empty stderr (stdout only)
6. Empty stdout (stderr only)
7. No output at all
8. Repeated Stop (idempotency)
"""

import sys
import os
import time
import uuid
from pathlib import Path
from typing import List

import pytest
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.services.training_center.job_manager import JobManager
from anylabeling.services.training_center.models import TrainingJob, TrainingMode, TrainingStatus
from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter
from anylabeling.services.training_center.event_protocol import TrainingEvent, TrainingEventType


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


@pytest.fixture
def custom_adapter():
    """Create CustomScriptAdapter instance"""
    adapter = CustomScriptAdapter()
    yield adapter
    if adapter.is_running():
        adapter.stop()
        timeout = 10
        start = time.time()
        while adapter.is_running() and (time.time() - start) < timeout:
            time.sleep(0.1)


def test_1_natural_completion(qapp, job_manager, custom_adapter, tmp_path):
    """Test 1: Natural completion with exit code 0"""

    captured_events: List[TrainingEvent] = []
    def capture_event(event: TrainingEvent):
        captured_events.append(event)
    job_manager.subscribe_events(capture_event)

    # Script with normal output and exit 0
    test_script = tmp_path / "test_normal.py"
    test_script.write_text("""
import time
print("START", flush=True)
for i in range(3):
    print(f"STEP={i+1}/3", flush=True)
    time.sleep(0.2)
print("FINISH", flush=True)
""")

    job_id = str(uuid.uuid4())
    job = TrainingJob(
        job_id=job_id,
        mode=TrainingMode.CUSTOM_SCRIPT,
        workspace=test_script.parent,
        python_executable=Path(sys.executable),
        display_name="Normal Test",
    )

    success, msg = job_manager.request_start(job, custom_adapter, {'script_path': str(test_script), 'arguments': []})
    assert success, f"Failed to start: {msg}"

    # Wait for completion with signal propagation
    wait_for_completion(qapp, custom_adapter)

    # Verify
    assert not custom_adapter.is_running()

    started = [e for e in captured_events if e.event_type == TrainingEventType.PROCESS_STARTED]
    completed = [e for e in captured_events if e.event_type == TrainingEventType.COMPLETED]
    failed = [e for e in captured_events if e.event_type == TrainingEventType.FAILED]
    stopped = [e for e in captured_events if e.event_type == TrainingEventType.STOPPED]
    console = [e for e in captured_events if e.event_type == TrainingEventType.CONSOLE_OUTPUT]

    assert len(started) == 1, "Should have exactly 1 PROCESS_STARTED"
    assert len(completed) == 1, "Should have exactly 1 COMPLETED"
    assert len(failed) == 0, "Should have no FAILED"
    assert len(stopped) == 0, "Should have no STOPPED"
    assert len(console) >= 5, "Should have console output (START, STEP=1/3, 2/3, 3/3, FINISH)"

    assert job_manager.get_current_job() is None, "Job should be cleared"

    job_manager.unsubscribe_events(capture_event)


def test_2_nonzero_exit(qapp, job_manager, custom_adapter, tmp_path):
    """Test 2: Non-zero exit (process failure)"""

    captured_events: List[TrainingEvent] = []
    def capture_event(event: TrainingEvent):
        captured_events.append(event)
    job_manager.subscribe_events(capture_event)

    test_script = tmp_path / "test_fail.py"
    test_script.write_text("""
import sys
print("START", flush=True)
print("ERROR: Something went wrong", flush=True)
sys.exit(1)
""")

    job_id = str(uuid.uuid4())
    job = TrainingJob(
        job_id=job_id,
        mode=TrainingMode.CUSTOM_SCRIPT,
        workspace=test_script.parent,
        python_executable=Path(sys.executable),
        display_name="Fail Test",
    )

    success, msg = job_manager.request_start(job, custom_adapter, {'script_path': str(test_script), 'arguments': []})
    assert success

    # Wait for completion with signal propagation
    wait_for_completion(qapp, custom_adapter)

    started = [e for e in captured_events if e.event_type == TrainingEventType.PROCESS_STARTED]
    completed = [e for e in captured_events if e.event_type == TrainingEventType.COMPLETED]
    failed = [e for e in captured_events if e.event_type == TrainingEventType.FAILED]

    assert len(started) == 1
    assert len(completed) == 0
    assert len(failed) == 1, "Should have exactly 1 FAILED"
    assert failed[0].payload.get('exit_code') == 1

    job_manager.unsubscribe_events(capture_event)


def test_3_user_stop(qapp, job_manager, custom_adapter, tmp_path):
    """Test 3: User Stop during long-running task"""

    captured_events: List[TrainingEvent] = []
    def capture_event(event: TrainingEvent):
        captured_events.append(event)
    job_manager.subscribe_events(capture_event)

    test_script = tmp_path / "test_long.py"
    test_script.write_text("""
import time
print("START", flush=True)
for i in range(60):
    print(f"STEP={i+1}/60", flush=True)
    time.sleep(1)
print("FINISH", flush=True)
""")

    job_id = str(uuid.uuid4())
    job = TrainingJob(
        job_id=job_id,
        mode=TrainingMode.CUSTOM_SCRIPT,
        workspace=test_script.parent,
        python_executable=Path(sys.executable),
        display_name="Long Test",
    )

    success, msg = job_manager.request_start(job, custom_adapter, {'script_path': str(test_script), 'arguments': []})
    assert success

    # Wait for start
    for _ in range(20):
        qapp.processEvents()
        time.sleep(0.1)

    # Stop
    stop_success = job_manager.request_stop()
    assert stop_success

    # Wait for stop
    for _ in range(30):
        qapp.processEvents()
        time.sleep(0.1)
        if not custom_adapter.is_running():
            break

    assert not custom_adapter.is_running()

    stopped = [e for e in captured_events if e.event_type == TrainingEventType.STOPPED]
    completed = [e for e in captured_events if e.event_type == TrainingEventType.COMPLETED]
    failed = [e for e in captured_events if e.event_type == TrainingEventType.FAILED]

    assert len(stopped) == 1, "Should have exactly 1 STOPPED"
    assert len(completed) == 0
    assert len(failed) == 0

    job_manager.unsubscribe_events(capture_event)


def test_4_stop_after_exit(qapp, job_manager, custom_adapter, tmp_path):
    """Test 4: Click Stop after process already exited naturally"""

    captured_events: List[TrainingEvent] = []
    def capture_event(event: TrainingEvent):
        captured_events.append(event)
    job_manager.subscribe_events(capture_event)

    test_script = tmp_path / "test_quick.py"
    test_script.write_text("""
print("DONE", flush=True)
""")

    job_id = str(uuid.uuid4())
    job = TrainingJob(
        job_id=job_id,
        mode=TrainingMode.CUSTOM_SCRIPT,
        workspace=test_script.parent,
        python_executable=Path(sys.executable),
        display_name="Quick Test",
    )

    success, msg = job_manager.request_start(job, custom_adapter, {'script_path': str(test_script), 'arguments': []})
    assert success

    # Wait for natural completion with signal propagation
    wait_for_completion(qapp, custom_adapter)

    # Process should have completed naturally
    assert not custom_adapter.is_running()

    # Try to stop (should fail or be no-op since no active job)
    stop_result = job_manager.request_stop()
    assert not stop_result, "Stop should return False when no job running"

    # Should still be COMPLETED, not changed to STOPPED
    completed = [e for e in captured_events if e.event_type == TrainingEventType.COMPLETED]
    stopped = [e for e in captured_events if e.event_type == TrainingEventType.STOPPED]

    assert len(completed) == 1
    assert len(stopped) == 0, "Should not emit STOPPED for already completed job"

    job_manager.unsubscribe_events(capture_event)


def test_5_empty_stderr(qapp, job_manager, custom_adapter, tmp_path):
    """Test 5: Script with only stdout (no stderr)"""

    captured_events: List[TrainingEvent] = []
    def capture_event(event: TrainingEvent):
        captured_events.append(event)
    job_manager.subscribe_events(capture_event)

    test_script = tmp_path / "test_stdout_only.py"
    test_script.write_text("""
print("STDOUT ONLY", flush=True)
""")

    job_id = str(uuid.uuid4())
    job = TrainingJob(
        job_id=job_id,
        mode=TrainingMode.CUSTOM_SCRIPT,
        workspace=test_script.parent,
        python_executable=Path(sys.executable),
        display_name="Stdout Only",
    )

    success, msg = job_manager.request_start(job, custom_adapter, {'script_path': str(test_script), 'arguments': []})
    assert success

    # Wait for completion with signal propagation
    wait_for_completion(qapp, custom_adapter)

    completed = [e for e in captured_events if e.event_type == TrainingEventType.COMPLETED]
    assert len(completed) == 1, "Should complete even with empty stderr"

    job_manager.unsubscribe_events(capture_event)


def test_6_empty_stdout(qapp, job_manager, custom_adapter, tmp_path):
    """Test 6: Script with only stderr (no stdout)"""

    captured_events: List[TrainingEvent] = []
    def capture_event(event: TrainingEvent):
        captured_events.append(event)
    job_manager.subscribe_events(capture_event)

    test_script = tmp_path / "test_stderr_only.py"
    test_script.write_text("""
import sys
print("STDERR ONLY", file=sys.stderr, flush=True)
""")

    job_id = str(uuid.uuid4())
    job = TrainingJob(
        job_id=job_id,
        mode=TrainingMode.CUSTOM_SCRIPT,
        workspace=test_script.parent,
        python_executable=Path(sys.executable),
        display_name="Stderr Only",
    )

    success, msg = job_manager.request_start(job, custom_adapter, {'script_path': str(test_script), 'arguments': []})
    assert success

    # Wait for completion with signal propagation
    wait_for_completion(qapp, custom_adapter)

    completed = [e for e in captured_events if e.event_type == TrainingEventType.COMPLETED]
    assert len(completed) == 1, "Should complete even with empty stdout"

    job_manager.unsubscribe_events(capture_event)


def test_7_no_output(qapp, job_manager, custom_adapter, tmp_path):
    """Test 7: Script with no output at all"""

    captured_events: List[TrainingEvent] = []
    def capture_event(event: TrainingEvent):
        captured_events.append(event)
    job_manager.subscribe_events(capture_event)

    test_script = tmp_path / "test_silent.py"
    test_script.write_text("""
import time
time.sleep(0.5)
""")

    job_id = str(uuid.uuid4())
    job = TrainingJob(
        job_id=job_id,
        mode=TrainingMode.CUSTOM_SCRIPT,
        workspace=test_script.parent,
        python_executable=Path(sys.executable),
        display_name="Silent",
    )

    success, msg = job_manager.request_start(job, custom_adapter, {'script_path': str(test_script), 'arguments': []})
    assert success

    # Wait for completion with signal propagation
    wait_for_completion(qapp, custom_adapter)

    completed = [e for e in captured_events if e.event_type == TrainingEventType.COMPLETED]
    assert len(completed) == 1, "Should complete even with no output"

    job_manager.unsubscribe_events(capture_event)


def test_8_repeated_stop(qapp, job_manager, custom_adapter, tmp_path):
    """Test 8: Repeated Stop calls (idempotency)"""

    captured_events: List[TrainingEvent] = []
    def capture_event(event: TrainingEvent):
        captured_events.append(event)
    job_manager.subscribe_events(capture_event)

    test_script = tmp_path / "test_repeat_stop.py"
    test_script.write_text("""
import time
for i in range(60):
    time.sleep(1)
""")

    job_id = str(uuid.uuid4())
    job = TrainingJob(
        job_id=job_id,
        mode=TrainingMode.CUSTOM_SCRIPT,
        workspace=test_script.parent,
        python_executable=Path(sys.executable),
        display_name="Repeat Stop",
    )

    success, msg = job_manager.request_start(job, custom_adapter, {'script_path': str(test_script), 'arguments': []})
    assert success

    for _ in range(20):
        qapp.processEvents()
        time.sleep(0.1)

    # First stop
    stop1 = job_manager.request_stop()
    assert stop1 == True, "First stop should succeed"

    # Second stop (should be idempotent)
    stop2 = job_manager.request_stop()
    assert stop2 == False, "Second stop should return False (idempotent)"

    # Third stop
    stop3 = job_manager.request_stop()
    assert stop3 == False, "Third stop should also return False"

    for _ in range(30):
        qapp.processEvents()
        time.sleep(0.1)
        if not custom_adapter.is_running():
            break

    # Should have exactly one STOPPED event
    stopped = [e for e in captured_events if e.event_type == TrainingEventType.STOPPED]
    assert len(stopped) == 1, "Should have exactly 1 STOPPED event despite multiple stop calls"

    job_manager.unsubscribe_events(capture_event)
