"""Isolated tests for CustomScriptAdapter error handling

Tests adapter logic without importing PyQt6-dependent modules.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock


class MockTrainingMode:
    CUSTOM_SCRIPT = "custom_script"


class MockTrainingStatus:
    IDLE = "idle"


class MockTrainingJob:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_adapter_missing_python_executable():
    """Error includes python path when executable not found"""
    with patch.dict('sys.modules', {
        'PyQt6': MagicMock(),
        'PyQt6.QtCore': MagicMock(),
    }):
        from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter

        adapter = CustomScriptAdapter()

        tmp_path = Path(__file__).parent
        script_path = Path(__file__)  # Use this file as dummy script

        job = MockTrainingJob(
            job_id="test-001",
            mode=MockTrainingMode.CUSTOM_SCRIPT,
            status=MockTrainingStatus.IDLE,
            created_at=datetime.now(),
            workspace=tmp_path,
            display_name="Test Job",
            framework="custom",
            python_executable=Path("/nonexistent/python"),
            command=["train.py"],
            metadata={},
        )

        config = {
            'script_path': str(script_path),
            'arguments': ['--epochs', '10'],
        }

        success, message = adapter.start(job, config)

        assert success is False
        assert "Python executable not found" in message
        assert "nonexistent" in message and "python" in message  # Platform-agnostic path check
        assert str(script_path) in message
        assert str(tmp_path) in message
        assert "--epochs" in message


def test_adapter_missing_script():
    """Error includes script path when script not found"""
    with patch.dict('sys.modules', {
        'PyQt6': MagicMock(),
        'PyQt6.QtCore': MagicMock(),
    }):
        from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter

        adapter = CustomScriptAdapter()

        tmp_path = Path(__file__).parent
        nonexistent_script = tmp_path / "nonexistent_test_script.py"

        job = MockTrainingJob(
            job_id="test-002",
            workspace=tmp_path,
            python_executable=Path(sys.executable),
        )

        config = {
            'script_path': str(nonexistent_script),
            'arguments': ['--batch-size', '32'],
        }

        success, message = adapter.start(job, config)

        assert success is False
        assert "Script file not found" in message
        assert str(nonexistent_script) in message
        assert str(tmp_path) in message
        assert "--batch-size" in message


def test_adapter_missing_workspace():
    """Error includes workspace path when workspace not found"""
    with patch.dict('sys.modules', {
        'PyQt6': MagicMock(),
        'PyQt6.QtCore': MagicMock(),
    }):
        from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter

        adapter = CustomScriptAdapter()

        script_path = Path(__file__)

        job = MockTrainingJob(
            job_id="test-003",
            workspace=Path("/nonexistent/workspace"),
            python_executable=Path(sys.executable),
        )

        config = {
            'script_path': str(script_path),
            'arguments': [],
        }

        success, message = adapter.start(job, config)

        assert success is False
        assert "Workspace directory not found" in message
        assert "nonexistent" in message and "workspace" in message  # Platform-agnostic path check


def test_adapter_validation_prevents_subprocess_call():
    """Validation catches errors before ProcessManager.start"""
    with patch.dict('sys.modules', {
        'PyQt6': MagicMock(),
        'PyQt6.QtCore': MagicMock(),
    }):
        from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter

        adapter = CustomScriptAdapter()

        tmp_path = Path(__file__).parent

        job = MockTrainingJob(
            job_id="test-004",
            workspace=tmp_path,
            python_executable=Path("/invalid/python"),
        )

        config = {
            'script_path': str(tmp_path / "nonexistent.py"),
            'arguments': [],
        }

        # ProcessManager.start should NOT be called
        with patch.object(adapter.manager, 'start') as mock_start:
            success, message = adapter.start(job, config)

        assert success is False
        assert not mock_start.called
        assert "Python executable not found" in message


def test_adapter_error_buffer_captured():
    """ProcessManager errors are captured via stderr buffer"""
    with patch.dict('sys.modules', {
        'PyQt6': MagicMock(),
        'PyQt6.QtCore': MagicMock(),
    }):
        from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter

        adapter = CustomScriptAdapter()

        tmp_path = Path(__file__).parent
        script_path = Path(__file__)

        job = MockTrainingJob(
            job_id="test-005",
            workspace=tmp_path,
            python_executable=Path(sys.executable),
        )

        config = {
            'script_path': str(script_path),
            'arguments': ['--test'],
        }

        # Mock ProcessManager.start to fail and emit error during start
        def mock_start_with_error(run):
            # Simulate stderr emission during start attempt
            adapter._on_stderr_buffered("Error: Permission denied: test.py")
            return False

        with patch.object(adapter.manager, 'start', side_effect=mock_start_with_error):
            success, message = adapter.start(job, config)

        assert success is False
        assert "Failed to start process" in message
        assert "Error: Permission denied: test.py" in message
        assert str(script_path) in message
        assert str(sys.executable) in message


if __name__ == "__main__":
    test_adapter_missing_python_executable()
    test_adapter_missing_script()
    test_adapter_missing_workspace()
    test_adapter_validation_prevents_subprocess_call()
    test_adapter_error_buffer_captured()
    print("All isolated adapter tests passed!")
