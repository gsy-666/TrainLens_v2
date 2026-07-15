"""Tests for Training Center Models

Verifies TrainingStatus state transitions and TrainingJob data integrity.
"""

import pytest
from datetime import datetime
from pathlib import Path

from anylabeling.services.training_center.models import (
    TrainingMode,
    TrainingStatus,
    TrainingJob,
)


class TestTrainingStatus:
    """Test TrainingStatus enum and state checks"""

    def test_terminal_states(self):
        """Terminal states: COMPLETED, FAILED, STOPPED"""
        assert TrainingStatus.COMPLETED.is_terminal()
        assert TrainingStatus.FAILED.is_terminal()
        assert TrainingStatus.STOPPED.is_terminal()

        assert not TrainingStatus.IDLE.is_terminal()
        assert not TrainingStatus.PREPARING.is_terminal()
        assert not TrainingStatus.RUNNING.is_terminal()
        assert not TrainingStatus.STOPPING.is_terminal()

    def test_active_states(self):
        """Active states: PREPARING, RUNNING, STOPPING"""
        assert TrainingStatus.PREPARING.is_active()
        assert TrainingStatus.RUNNING.is_active()
        assert TrainingStatus.STOPPING.is_active()

        assert not TrainingStatus.IDLE.is_active()
        assert not TrainingStatus.COMPLETED.is_active()
        assert not TrainingStatus.FAILED.is_active()
        assert not TrainingStatus.STOPPED.is_active()

    def test_legal_transitions(self):
        """Verify expected state transition paths"""
        # Normal flow: IDLE -> PREPARING -> RUNNING -> COMPLETED
        start_state = TrainingStatus.IDLE
        assert not start_state.is_active()

        preparing = TrainingStatus.PREPARING
        assert preparing.is_active()
        assert not preparing.is_terminal()

        running = TrainingStatus.RUNNING
        assert running.is_active()
        assert not running.is_terminal()

        completed = TrainingStatus.COMPLETED
        assert not completed.is_active()
        assert completed.is_terminal()

    def test_stop_flow(self):
        """Stop flow: RUNNING -> STOPPING -> STOPPED"""
        running = TrainingStatus.RUNNING
        assert running.is_active()

        stopping = TrainingStatus.STOPPING
        assert stopping.is_active()
        assert not stopping.is_terminal()

        stopped = TrainingStatus.STOPPED
        assert not stopped.is_active()
        assert stopped.is_terminal()

    def test_failure_flow(self):
        """Failure can occur from any active state"""
        for status in [TrainingStatus.PREPARING, TrainingStatus.RUNNING, TrainingStatus.STOPPING]:
            assert status.is_active()

        failed = TrainingStatus.FAILED
        assert not failed.is_active()
        assert failed.is_terminal()


class TestTrainingJob:
    """Test TrainingJob dataclass"""

    def test_job_creation_minimal(self):
        """Create job with minimal required fields"""
        job = TrainingJob(
            job_id="test-001",
            mode=TrainingMode.CUSTOM_SCRIPT,
            status=TrainingStatus.IDLE,
            created_at=datetime.now(),
            started_at=None,
            ended_at=None,
            workspace=None,
            output_directory=None,
            display_name=None,
            framework=None,
            python_executable=None,
            command=None,
            metadata={},
            error_message=None,
        )

        assert job.job_id == "test-001"
        assert job.mode == TrainingMode.CUSTOM_SCRIPT
        assert job.status == TrainingStatus.IDLE
        assert isinstance(job.created_at, datetime)
        assert job.metadata == {}

    def test_job_creation_full(self):
        """Create job with all fields populated"""
        now = datetime.now()
        workspace = Path("/tmp/workspace")
        output_dir = Path("/tmp/output")
        python_exe = Path("/usr/bin/python3")

        job = TrainingJob(
            job_id="test-002",
            mode=TrainingMode.GUIDED_ULTRALYTICS,
            status=TrainingStatus.RUNNING,
            created_at=now,
            started_at=now,
            ended_at=None,
            workspace=workspace,
            output_directory=output_dir,
            display_name="YOLOv8 Detection",
            framework="ultralytics",
            python_executable=python_exe,
            command=["python", "train.py"],
            metadata={"model": "yolov8n", "epochs": 100},
            error_message=None,
        )

        assert job.job_id == "test-002"
        assert job.mode == TrainingMode.GUIDED_ULTRALYTICS
        assert job.status == TrainingStatus.RUNNING
        assert job.workspace == workspace
        assert job.output_directory == output_dir
        assert job.display_name == "YOLOv8 Detection"
        assert job.framework == "ultralytics"
        assert job.python_executable == python_exe
        assert job.command == ["python", "train.py"]
        assert job.metadata["model"] == "yolov8n"
        assert job.metadata["epochs"] == 100

    def test_job_status_mutation(self):
        """Job status can be updated"""
        job = TrainingJob(
            job_id="test-003",
            mode=TrainingMode.CUSTOM_SCRIPT,
            status=TrainingStatus.IDLE,
            created_at=datetime.now(),
            started_at=None,
            ended_at=None,
            workspace=None,
            output_directory=None,
            display_name=None,
            framework=None,
            python_executable=None,
            command=None,
            metadata={},
            error_message=None,
        )

        assert job.status == TrainingStatus.IDLE

        job.status = TrainingStatus.PREPARING
        assert job.status == TrainingStatus.PREPARING
        assert job.status.is_active()

        job.status = TrainingStatus.RUNNING
        assert job.status == TrainingStatus.RUNNING

        job.status = TrainingStatus.COMPLETED
        assert job.status == TrainingStatus.COMPLETED
        assert job.status.is_terminal()

    def test_job_metadata_mutable(self):
        """Job metadata dictionary is mutable"""
        job = TrainingJob(
            job_id="test-004",
            mode=TrainingMode.CUSTOM_SCRIPT,
            status=TrainingStatus.IDLE,
            created_at=datetime.now(),
            started_at=None,
            ended_at=None,
            workspace=None,
            output_directory=None,
            display_name=None,
            framework=None,
            python_executable=None,
            command=None,
            metadata={},
            error_message=None,
        )

        assert len(job.metadata) == 0

        job.metadata["key1"] = "value1"
        assert job.metadata["key1"] == "value1"

        job.metadata.update({"key2": "value2", "key3": 123})
        assert len(job.metadata) == 3
        assert job.metadata["key3"] == 123
