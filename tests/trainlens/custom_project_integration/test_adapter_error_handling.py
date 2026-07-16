"""Tests for CustomScriptAdapter error handling and diagnostics

Verifies that adapter provides detailed error messages with full diagnostic information.
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from anylabeling.services.training_center.models import TrainingMode, TrainingStatus, TrainingJob
from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter


@pytest.fixture
def adapter():
    """Create CustomScriptAdapter instance"""
    return CustomScriptAdapter()


@pytest.fixture
def sample_job(tmp_path):
    """Create sample TrainingJob"""
    return TrainingJob(
        job_id="test-001",
        mode=TrainingMode.CUSTOM_SCRIPT,
        status=TrainingStatus.IDLE,
        created_at=datetime.now(),
        workspace=tmp_path,
        display_name="Test Job",
        framework="custom",
        python_executable=Path("python"),
        command=["train.py"],
        metadata={},
    )


class TestAdapterErrorDiagnostics:
    """Test error message diagnostics"""

    def test_missing_python_executable_shows_diagnostics(self, adapter, sample_job, tmp_path):
        """Error includes python path when executable not found"""
        script_path = tmp_path / "train.py"
        script_path.write_text("print('test')")

        sample_job.python_executable = Path("/nonexistent/python")

        config = {
            'script_path': str(script_path),
            'arguments': ['--epochs', '10'],
        }

        success, message = adapter.start(sample_job, config)

        assert success is False
        assert "Python executable not found" in message
        assert "nonexistent" in message and "python" in message  # Platform-agnostic path check
        assert str(script_path) in message
        assert str(tmp_path) in message
        assert "--epochs" in message

    def test_missing_script_shows_diagnostics(self, adapter, sample_job, tmp_path):
        """Error includes script path when script not found"""
        import sys
        sample_job.python_executable = Path(sys.executable)

        nonexistent_script = tmp_path / "nonexistent.py"

        config = {
            'script_path': str(nonexistent_script),
            'arguments': ['--batch-size', '32'],
        }

        success, message = adapter.start(sample_job, config)

        assert success is False
        assert "Script file not found" in message
        assert str(nonexistent_script) in message
        assert str(tmp_path) in message
        assert "--batch-size" in message

    def test_missing_workspace_shows_diagnostics(self, adapter, sample_job, tmp_path):
        """Error includes workspace path when workspace not found"""
        import sys
        sample_job.python_executable = Path(sys.executable)
        sample_job.workspace = Path("/nonexistent/workspace")

        script_path = tmp_path / "train.py"
        script_path.write_text("print('test')")

        config = {
            'script_path': str(script_path),
            'arguments': [],
        }

        success, message = adapter.start(sample_job, config)

        assert success is False
        assert "Workspace directory not found" in message
        assert "nonexistent" in message and "workspace" in message  # Platform-agnostic path check

    def test_process_manager_error_captured(self, adapter, sample_job, tmp_path):
        """ProcessManager errors are captured and included in message"""
        import sys
        script_path = tmp_path / "train.py"
        script_path.write_text("print('test')")

        sample_job.python_executable = Path(sys.executable)

        config = {
            'script_path': str(script_path),
            'arguments': ['--test'],
        }

        # Mock ProcessManager.start to fail and emit error during start
        def mock_start_with_error(run):
            # Simulate stderr emission during start attempt
            adapter._on_stderr_buffered("Error: Permission denied")
            return False

        with patch.object(adapter.manager, 'start', side_effect=mock_start_with_error):
            success, message = adapter.start(sample_job, config)

        assert success is False
        assert "Failed to start process" in message
        assert "Error: Permission denied" in message
        assert str(script_path) in message
        assert str(sys.executable) in message

    def test_successful_start_clears_error_buffer(self, adapter, sample_job, tmp_path):
        """Successful start clears previous error buffer"""
        import sys
        script_path = tmp_path / "train.py"
        script_path.write_text("print('test')")

        sample_job.python_executable = Path(sys.executable)

        # Add stale errors to buffer
        adapter._error_buffer.append("Old error")

        config = {
            'script_path': str(script_path),
            'arguments': [],
        }

        with patch.object(adapter.manager, 'start', return_value=True):
            success, message = adapter.start(sample_job, config)

        assert success is True
        assert "Old error" not in message
        # Buffer should have been cleared at start
        assert len(adapter._error_buffer) == 0


class TestAdapterValidation:
    """Test pre-validation before ProcessManager call"""

    def test_validation_prevents_invalid_subprocess_call(self, adapter, sample_job, tmp_path):
        """Validation catches errors before subprocess.Popen"""
        sample_job.python_executable = Path("/invalid/python")

        config = {
            'script_path': str(tmp_path / "nonexistent.py"),
            'arguments': [],
        }

        # ProcessManager.start should NOT be called
        with patch.object(adapter.manager, 'start') as mock_start:
            success, message = adapter.start(sample_job, config)

        assert success is False
        assert not mock_start.called
        assert "Python executable not found" in message
