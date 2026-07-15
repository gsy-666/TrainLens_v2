"""Integration Tests for Custom Project + JobManager

Simplified tests focusing on core integration functionality.
"""

import pytest
from unittest.mock import patch
from pathlib import Path
from datetime import datetime

from anylabeling.services.training_center.models import TrainingMode, TrainingStatus, TrainingJob
from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter


@pytest.fixture
def temp_history_dir(tmp_path):
    """Temporary history directory"""
    return tmp_path / "history"


@pytest.fixture
def history_store(temp_history_dir):
    """Fresh HistoryStore instance"""
    from anylabeling.services.training_center.history import HistoryStore
    store = HistoryStore(history_dir=temp_history_dir)
    yield store
    store.clear()


class TestCustomProjectIntegration:
    """Test custom project integration basics"""

    def test_training_job_creation(self):
        """Can create TrainingJob for custom script"""
        job = TrainingJob(
            job_id="test-custom-001",
            mode=TrainingMode.CUSTOM_SCRIPT,
            status=TrainingStatus.IDLE,
            created_at=datetime.now(),
            workspace=Path("/tmp/workspace"),
            display_name="Test Script",
            framework="custom",
            python_executable=Path("/usr/bin/python3"),
            command=["train.py", "--epochs", "10"],
            metadata={"test": "value"},
        )

        assert job.job_id == "test-custom-001"
        assert job.mode == TrainingMode.CUSTOM_SCRIPT
        assert job.status == TrainingStatus.IDLE
        assert job.command == ["train.py", "--epochs", "10"]

    def test_adapter_creation(self):
        """Can create CustomScriptAdapter"""
        adapter = CustomScriptAdapter()

        assert adapter is not None
        can_start, reason = adapter.can_start()
        assert can_start is True

    def test_adapter_config_validation(self):
        """Adapter validates config correctly"""
        adapter = CustomScriptAdapter()
        job = TrainingJob(
            job_id="test-001",
            mode=TrainingMode.CUSTOM_SCRIPT,
            status=TrainingStatus.IDLE,
            created_at=datetime.now(),
            workspace=Path("/tmp"),
            display_name="Test",
            framework="custom",
            python_executable=Path("/usr/bin/python3"),
            command=["train.py"],
            metadata={},
        )

        # Valid config
        valid_config = {'script_path': '/tmp/train.py'}
        with patch('anylabeling.services.run_monitor.process_manager.ProcessManager.start', return_value=True):
            success, _ = adapter.start(job, valid_config)
        assert success is True

        # Invalid config (missing script_path)
        invalid_config = {'arguments': ['--epochs', '10']}
        success, message = adapter.start(job, invalid_config)
        assert success is False
        assert "script_path" in message.lower()


class TestHistoryStoreIntegration:
    """Test history persistence"""

    def test_job_written_to_history(self, history_store):
        """Can write job to history"""
        from anylabeling.services.training_center.history import JobHistoryRecord

        record = JobHistoryRecord(
            job_id="history-test-001",
            mode=TrainingMode.CUSTOM_SCRIPT.value,
            status=TrainingStatus.IDLE.value,
            created_at=datetime.now().isoformat(),
            workspace="/tmp/workspace",
            display_name="History Test",
            framework="custom",
            python_executable="/usr/bin/python3",
            command=["train.py", "--epochs", "100"],
            metadata={"test_key": "test_value"},
        )
        history_store.append_job(record)

        retrieved = history_store.get_job("history-test-001")
        assert retrieved is not None
        assert retrieved.job_id == "history-test-001"
        assert retrieved.mode == TrainingMode.CUSTOM_SCRIPT.value
        assert retrieved.command == ["train.py", "--epochs", "100"]

    def test_history_updated_on_completion(self, history_store):
        """Can update history when job completes"""
        from anylabeling.services.training_center.history import JobHistoryRecord

        record = JobHistoryRecord(
            job_id="complete-history-001",
            mode=TrainingMode.CUSTOM_SCRIPT.value,
            status=TrainingStatus.RUNNING.value,
            created_at=datetime.now().isoformat(),
            started_at=datetime.now().isoformat(),
            workspace="/tmp",
            display_name="Complete History",
            framework="custom",
            metadata={},
        )
        history_store.append_job(record)

        # Finalize
        history_store.finalize_job(
            "complete-history-001",
            status=TrainingStatus.COMPLETED,
            ended_at=datetime.now(),
            final_epoch=100,
            duration_seconds=3600.0,
        )

        # Verify
        retrieved = history_store.get_job("complete-history-001")
        assert retrieved.status == TrainingStatus.COMPLETED.value
        assert retrieved.final_epoch == 100
        assert retrieved.duration_seconds == 3600.0

    def test_history_updated_on_failure(self, history_store):
        """Can update history when job fails"""
        from anylabeling.services.training_center.history import JobHistoryRecord

        record = JobHistoryRecord(
            job_id="fail-history-001",
            mode=TrainingMode.CUSTOM_SCRIPT.value,
            status=TrainingStatus.RUNNING.value,
            created_at=datetime.now().isoformat(),
            workspace="/tmp",
            display_name="Fail History",
            framework="custom",
            metadata={},
        )
        history_store.append_job(record)

        # Finalize with error
        history_store.finalize_job(
            "fail-history-001",
            status=TrainingStatus.FAILED,
            ended_at=datetime.now(),
            error_message="Process exited with code 1",
        )

        # Verify
        retrieved = history_store.get_job("fail-history-001")
        assert retrieved.status == TrainingStatus.FAILED.value
        assert retrieved.error_message == "Process exited with code 1"

    def test_history_query_by_mode(self, history_store):
        """Can query history by training mode"""
        from anylabeling.services.training_center.history import JobHistoryRecord

        # Add custom script job
        custom_job = JobHistoryRecord(
            job_id="custom-001",
            mode=TrainingMode.CUSTOM_SCRIPT.value,
            status=TrainingStatus.COMPLETED.value,
            created_at=datetime.now().isoformat(),
            workspace="/tmp",
            display_name="Custom Job",
            framework="custom",
            metadata={},
        )
        history_store.append_job(custom_job)

        # Query by mode
        results = history_store.list_jobs(mode=TrainingMode.CUSTOM_SCRIPT)
        assert len(results) == 1
        assert results[0].job_id == "custom-001"
