"""Real subprocess integration test - direct import

Tests actual process execution by directly importing adapter module.
Bypasses anylabeling.__init__ to avoid GUI dependencies.
"""

import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Mock PyQt6 BEFORE any imports
mock_pyqt = MagicMock()
mock_pyqt_core = MagicMock()
mock_pyqt_core.QObject = object
mock_pyqt_core.pyqtSignal = lambda *args, **kwargs: Mock()

sys.modules['PyQt6'] = mock_pyqt
sys.modules['PyQt6.QtCore'] = mock_pyqt_core
sys.modules['PyQt6.QtWidgets'] = MagicMock()
sys.modules['PyQt6.QtGui'] = MagicMock()

# Add project root to path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


def test_real_subprocess_execution():
    """Test adapter with real subprocess execution"""
    # Direct import to bypass anylabeling.__init__
    from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter

    # Create temporary workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create test script
        script_path = workspace / "test_train.py"
        script_path.write_text("""
import sys
import time
print("TRAINLENS_START_OK", flush=True)
print(f"PYTHON={sys.executable}", flush=True)
print(f"ARGS={sys.argv[1:]}", flush=True)
time.sleep(0.5)
print("TRAINLENS_FINISH_OK", flush=True)
""")

        # Create adapter
        adapter = CustomScriptAdapter()

        # Create mock job
        job = Mock()
        job.job_id = "real-test-001"
        job.workspace = workspace
        job.python_executable = Path(sys.executable)
        job.command = [str(script_path), "--epochs", "10"]

        config = {
            'script_path': str(script_path),
            'arguments': ['--epochs', '10'],
        }

        # Capture console output
        stdout_lines = []
        stderr_lines = []

        def on_event(event):
            if event.get('type') == 'console_output':
                if event.get('stream') == 'stdout':
                    stdout_lines.append(event.get('message'))
                elif event.get('stream') == 'stderr':
                    stderr_lines.append(event.get('message'))

        adapter.subscribe(on_event)

        # Start process
        success, message = adapter.start(job, config)

        print(f"\n[PASS] Start result: success={success}, message={message}")

        if not success:
            print(f"\n[FAIL] ERROR: Failed to start process")
            print(f"Message:\n{message}")
            raise AssertionError(f"Process failed to start: {message}")

        assert success is True
        assert "Process started" in message

        # Wait for process to complete
        timeout = 5.0
        start_time = time.time()
        while adapter.is_running() and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        # Verify process completed
        if adapter.is_running():
            raise AssertionError("Process should have completed within timeout")

        # Verify output
        stdout_text = '\n'.join(stdout_lines)
        print(f"\n[PASS] Stdout captured ({len(stdout_lines)} lines):")
        for line in stdout_lines[:10]:  # Show first 10 lines
            print(f"  {line}")

        assert "TRAINLENS_START_OK" in stdout_text
        assert "TRAINLENS_FINISH_OK" in stdout_text
        assert sys.executable in stdout_text
        assert "--epochs" in stdout_text or "10" in stdout_text

        print("\n[PASS] Real subprocess execution test PASSED!")
        return True


def test_real_subprocess_with_invalid_python():
    """Test error handling with invalid python path"""
    from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        script_path = workspace / "test.py"
        script_path.write_text("print('hello')")

        adapter = CustomScriptAdapter()

        job = Mock()
        job.job_id = "real-test-002"
        job.workspace = workspace
        job.python_executable = Path("/nonexistent/python.exe")

        config = {
            'script_path': str(script_path),
            'arguments': [],
        }

        success, message = adapter.start(job, config)

        print(f"\n[PASS] Invalid python test result: success={success}")
        print(f"Message (first 500 chars):\n{message[:500]}")

        assert success is False
        assert "Python executable not found" in message
        assert "/nonexistent/python" in message
        assert str(script_path) in message
        assert str(workspace) in message

        print("\n[PASS] Invalid python error handling test PASSED!")
        return True


def test_real_subprocess_with_invalid_script():
    """Test error handling with invalid script path"""
    from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        nonexistent_script = workspace / "nonexistent.py"

        adapter = CustomScriptAdapter()

        job = Mock()
        job.job_id = "real-test-003"
        job.workspace = workspace
        job.python_executable = Path(sys.executable)

        config = {
            'script_path': str(nonexistent_script),
            'arguments': ['--test'],
        }

        success, message = adapter.start(job, config)

        print(f"\n[PASS] Invalid script test result: success={success}")
        print(f"Message (first 500 chars):\n{message[:500]}")

        assert success is False
        assert "Script file not found" in message
        assert str(nonexistent_script) in message
        assert "--test" in message

        print("\n[PASS] Invalid script error handling test PASSED!")
        return True


if __name__ == "__main__":
    import sys
    import os

    # Add project root to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

    print("=" * 70)
    print("Running real subprocess integration tests")
    print("=" * 70)

    try:
        test_real_subprocess_execution()
        test_real_subprocess_with_invalid_python()
        test_real_subprocess_with_invalid_script()

        print("\n" + "=" * 70)
        print("[PASS] ALL REAL SUBPROCESS TESTS PASSED!")
        print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"[FAIL] TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)
