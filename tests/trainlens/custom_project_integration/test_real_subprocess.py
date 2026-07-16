"""Real subprocess integration test for CustomScriptAdapter

Tests actual process execution without GUI dependencies.
"""

import sys
import time
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock


def test_real_subprocess_execution():
    """Test adapter with real subprocess execution"""
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

        # Mock PyQt6 to avoid import errors
        from unittest.mock import MagicMock
        sys.modules['PyQt6'] = MagicMock()
        sys.modules['PyQt6.QtCore'] = MagicMock()
        sys.modules['PyQt6.QtCore'].QObject = object
        sys.modules['PyQt6.QtCore'].pyqtSignal = lambda *args, **kwargs: Mock()

        # Import after mocking
        from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter

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

        print(f"\nStart result: success={success}, message={message}")

        if not success:
            print(f"\nERROR: Failed to start process")
            print(f"Message: {message}")
            assert False, f"Process failed to start: {message}"

        assert success is True
        assert "Process started" in message

        # Wait for process to complete
        timeout = 5.0
        start_time = time.time()
        while adapter.is_running() and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        # Verify process completed
        assert not adapter.is_running(), "Process should have completed"

        # Verify output
        stdout_text = '\n'.join(stdout_lines)
        print(f"\nStdout captured ({len(stdout_lines)} lines):")
        print(stdout_text)

        assert "TRAINLENS_START_OK" in stdout_text
        assert "TRAINLENS_FINISH_OK" in stdout_text
        assert sys.executable in stdout_text
        assert "--epochs" in stdout_text or "10" in stdout_text

        print("\n✓ Real subprocess test passed!")


def test_real_subprocess_with_invalid_python():
    """Test error handling with invalid python path"""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        script_path = workspace / "test.py"
        script_path.write_text("print('hello')")

        # Mock PyQt6
        from unittest.mock import MagicMock
        sys.modules['PyQt6'] = MagicMock()
        sys.modules['PyQt6.QtCore'] = MagicMock()
        sys.modules['PyQt6.QtCore'].QObject = object
        sys.modules['PyQt6.QtCore'].pyqtSignal = lambda *args, **kwargs: Mock()

        from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter

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

        print(f"\nInvalid python test result: success={success}")
        print(f"Message:\n{message}")

        assert success is False
        assert "Python executable not found" in message
        assert "/nonexistent/python" in message
        assert str(script_path) in message
        assert str(workspace) in message

        print("\n✓ Invalid python error handling test passed!")


def test_real_subprocess_with_invalid_script():
    """Test error handling with invalid script path"""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        nonexistent_script = workspace / "nonexistent.py"

        # Mock PyQt6
        from unittest.mock import MagicMock
        sys.modules['PyQt6'] = MagicMock()
        sys.modules['PyQt6.QtCore'] = MagicMock()
        sys.modules['PyQt6.QtCore'].QObject = object
        sys.modules['PyQt6.QtCore'].pyqtSignal = lambda *args, **kwargs: Mock()

        from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter

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

        print(f"\nInvalid script test result: success={success}")
        print(f"Message:\n{message}")

        assert success is False
        assert "Script file not found" in message
        assert str(nonexistent_script) in message
        assert "--test" in message

        print("\n✓ Invalid script error handling test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Running real subprocess integration tests")
    print("=" * 60)

    test_real_subprocess_execution()
    test_real_subprocess_with_invalid_python()
    test_real_subprocess_with_invalid_script()

    print("\n" + "=" * 60)
    print("All real subprocess tests passed!")
    print("=" * 60)
