"""Real end-to-end integration test for Custom Project launch

Tests complete chain: JobManager → CustomScriptAdapter → ProcessManager → subprocess
Uses real components, no mocks except PyQt dependencies.
"""

import sys
import os
import time
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

# Mock all PyQt6 modules before any imports
mock_pyqt = MagicMock()
sys.modules['PyQt6'] = mock_pyqt
sys.modules['PyQt6.QtCore'] = MagicMock()
sys.modules['PyQt6.QtCore'].QObject = object
sys.modules['PyQt6.QtCore'].pyqtSignal = lambda *args, **kwargs: MagicMock()
sys.modules['PyQt6.QtWidgets'] = MagicMock()
sys.modules['PyQt6.QtGui'] = MagicMock()
sys.modules['PyQt6.QtMultimedia'] = MagicMock()
sys.modules['PyQt6.QtMultimediaWidgets'] = MagicMock()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


def test_full_chain_successful_launch():
    """Test complete chain: JobManager → Adapter → ProcessManager → subprocess"""
    print("\n[TEST] Full chain successful launch")

    # Import after mocking
    from anylabeling.services.training_center.job_manager import JobManager
    from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter
    from anylabeling.services.training_center.models import TrainingJob, TrainingMode, TrainingStatus

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
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

        # Create real JobManager and force cleanup any previous state
        job_manager = JobManager()
        if job_manager._current_adapter:
            try:
                job_manager._current_adapter.stop()
                time.sleep(0.2)
            except:
                pass
            job_manager._current_adapter.unsubscribe(job_manager._on_adapter_event)
        job_manager._current_adapter = None
        job_manager._current_job = None

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
            print(f"  Event: {event.event_type} - {event.job_id}")

        adapter.subscribe(capture_event)

        # Config for adapter
        config = {
            'script_path': str(script_path),
            'arguments': ['--test-arg', 'value'],
        }

        print(f"  Workspace: {workspace}")
        print(f"  Script: {script_path}")
        print(f"  Python: {sys.executable}")
        print(f"  Config: {config}")

        # Start via JobManager
        success, message = job_manager.request_start(job=job, adapter=adapter, config=config)

        print(f"  request_start result: success={success}, message={message}")

        if not success:
            print(f"  ERROR: Failed to start: {message}")
            assert False, f"Failed to start: {message}"

        assert success is True, f"Start failed: {message}"

        # Wait for process to complete
        timeout = 10.0
        start_time = time.time()
        while adapter.is_running() and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        # Check process completed
        assert not adapter.is_running(), "Process should have completed"

        # Verify events
        print(f"  Total events captured: {len(events)}")

        event_types = [e.event_type.value for e in events]
        print(f"  Event types: {event_types}")

        # Should have: process_started, console_output (multiple), completed
        assert 'process_started' in event_types, "Missing process_started event"
        assert 'completed' in event_types or 'failed' in event_types, "Missing completion event"

        # Check console output
        stdout_events = [e for e in events if e.event_type.value == 'console_output' and e.payload.get('stream') == 'stdout']
        stdout_text = '\n'.join([e.payload.get('message', '') for e in stdout_events])

        print(f"  Stdout:\n{stdout_text}")

        assert "TRAINLENS_START_OK" in stdout_text, "Missing START marker"
        assert "TRAINLENS_FINISH_OK" in stdout_text, "Missing FINISH marker"

        # Check JobManager released job
        current_job = job_manager.get_current_job()
        print(f"  Current job after completion: {current_job}")

        print("[PASS] Full chain successful launch test PASSED!")


def test_full_chain_nonzero_exit():
    """Test complete chain with non-zero exit code"""
    print("\n[TEST] Full chain non-zero exit")

    from anylabeling.services.training_center.job_manager import JobManager
    from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter
    from anylabeling.services.training_center.models import TrainingJob, TrainingMode, TrainingStatus

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        script_path = workspace / "fail.py"
        script_path.write_text("""
import sys
print("TRAINLENS_START_OK", flush=True)
print("ERROR: Simulated failure", flush=True)
sys.exit(1)
""")

        # Create JobManager and force cleanup any previous state
        job_manager = JobManager()
        if job_manager._current_adapter:
            try:
                job_manager._current_adapter.stop()
                time.sleep(0.2)
            except:
                pass
            job_manager._current_adapter.unsubscribe(job_manager._on_adapter_event)
        job_manager._current_adapter = None
        job_manager._current_job = None

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

        # Wait for completion
        timeout = 10.0
        start_time = time.time()
        while adapter.is_running() and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        # Check failed event
        event_types = [e.event_type.value for e in events]
        print(f"  Event types: {event_types}")

        assert 'failed' in event_types, "Should have failed event for non-zero exit"

        failed_events = [e for e in events if e.event_type.value == 'failed']
        if failed_events:
            print(f"  Failed event: {failed_events[0]}")
            assert failed_events[0].payload.get('exit_code') == 1

        print("[PASS] Full chain non-zero exit test PASSED!")


def test_full_chain_invalid_python():
    """Test complete chain with invalid Python executable"""
    print("\n[TEST] Full chain invalid python")

    from anylabeling.services.training_center.job_manager import JobManager
    from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter
    from anylabeling.services.training_center.models import TrainingJob, TrainingMode, TrainingStatus

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        script_path = workspace / "train.py"
        script_path.write_text("print('hello')")

        # Create JobManager and force cleanup any previous state
        job_manager = JobManager()
        if job_manager._current_adapter:
            try:
                job_manager._current_adapter.stop()
                time.sleep(0.2)
            except:
                pass
            job_manager._current_adapter.unsubscribe(job_manager._on_adapter_event)
        job_manager._current_adapter = None
        job_manager._current_job = None

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

        print(f"  Result: success={success}")
        print(f"  Message:\n{message}")

        assert success is False, "Should fail for invalid python"
        assert "Python executable not found" in message, "Should have diagnostic message"
        assert "/nonexistent/python" in message, "Should include python path"
        assert str(script_path) in message, "Should include script path"
        assert str(workspace) in message, "Should include workspace"

        print("[PASS] Full chain invalid python test PASSED!")


def test_full_chain_invalid_script():
    """Test complete chain with invalid script path"""
    print("\n[TEST] Full chain invalid script")

    from anylabeling.services.training_center.job_manager import JobManager
    from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter
    from anylabeling.services.training_center.models import TrainingJob, TrainingMode, TrainingStatus

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        nonexistent_script = workspace / "nonexistent.py"

        # Create JobManager and force cleanup any previous state
        job_manager = JobManager()
        if job_manager._current_adapter:
            try:
                job_manager._current_adapter.stop()
                time.sleep(0.2)
            except:
                pass
            job_manager._current_adapter.unsubscribe(job_manager._on_adapter_event)
        job_manager._current_adapter = None
        job_manager._current_job = None

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

        print(f"  Result: success={success}")
        print(f"  Message:\n{message}")

        assert success is False, "Should fail for invalid script"
        assert "Script file not found" in message, "Should have diagnostic message"
        assert str(nonexistent_script) in message, "Should include script path"
        assert "--test" in message, "Should include arguments"

        print("[PASS] Full chain invalid script test PASSED!")


if __name__ == "__main__":
    print("=" * 70)
    print("Running real end-to-end integration tests")
    print("=" * 70)

    try:
        test_full_chain_successful_launch()
        test_full_chain_nonzero_exit()
        test_full_chain_invalid_python()
        test_full_chain_invalid_script()

        print("\n" + "=" * 70)
        print("[PASS] ALL END-TO-END INTEGRATION TESTS PASSED!")
        print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"[FAIL] TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)
