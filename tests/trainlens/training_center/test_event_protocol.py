"""Tests for Training Center Event Protocol

Verifies unified event creation and schema validation.
"""

import pytest
import time

from anylabeling.services.training_center.event_protocol import (
    TrainingEventType,
    TrainingEvent,
    create_job_created_event,
    create_preparing_event,
    create_process_started_event,
    create_console_output_event,
    create_resource_sample_event,
    create_epoch_metrics_event,
    create_checkpoint_saved_event,
    create_completed_event,
    create_failed_event,
    create_stopped_event,
)


class TestTrainingEventType:
    """Test TrainingEventType enum"""

    def test_all_event_types_defined(self):
        """All expected event types exist"""
        expected = [
            "JOB_CREATED",
            "PREPARING",
            "PROCESS_STARTED",
            "CONSOLE_OUTPUT",
            "RESOURCE_SAMPLE",
            "EPOCH_METRICS",
            "CHECKPOINT_SAVED",
            "COMPLETED",
            "FAILED",
            "STOPPED",
        ]

        for name in expected:
            assert hasattr(TrainingEventType, name)

    def test_event_type_values(self):
        """Event type values match expected format"""
        assert TrainingEventType.JOB_CREATED.value == "job_created"
        assert TrainingEventType.PROCESS_STARTED.value == "process_started"
        assert TrainingEventType.CONSOLE_OUTPUT.value == "console_output"
        assert TrainingEventType.COMPLETED.value == "completed"
        assert TrainingEventType.FAILED.value == "failed"
        assert TrainingEventType.STOPPED.value == "stopped"


class TestTrainingEvent:
    """Test TrainingEvent dataclass"""

    def test_event_creation_minimal(self):
        """Create event with minimal fields"""
        event = TrainingEvent(
            schema_version=1,
            job_id="job-001",
            event_type=TrainingEventType.CONSOLE_OUTPUT,
            timestamp=time.time(),
            payload={"message": "test"},
            source=None,
        )

        assert event.schema_version == 1
        assert event.job_id == "job-001"
        assert event.event_type == TrainingEventType.CONSOLE_OUTPUT
        assert isinstance(event.timestamp, float)
        assert event.payload["message"] == "test"
        assert event.source is None

    def test_event_with_source(self):
        """Event includes source identifier"""
        event = TrainingEvent(
            schema_version=1,
            job_id="job-002",
            event_type=TrainingEventType.PROCESS_STARTED,
            timestamp=time.time(),
            payload={"pid": 1234},
            source="ultralytics",
        )

        assert event.source == "ultralytics"


class TestEventCreators:
    """Test event creator functions"""

    def test_create_job_created_event(self):
        """job_created event constructor"""
        ts = time.time()
        event = create_job_created_event(
            job_id="job-001",
            timestamp=ts,
            mode="custom_script",
            display_name="Test Job",
        )

        assert event.schema_version == 1
        assert event.job_id == "job-001"
        assert event.event_type == TrainingEventType.JOB_CREATED
        assert event.timestamp == ts
        assert event.payload["mode"] == "custom_script"
        assert event.payload["display_name"] == "Test Job"

    def test_create_preparing_event(self):
        """preparing event constructor"""
        event = create_preparing_event(
            job_id="job-002",
            timestamp=time.time(),
            source="ultralytics",
        )

        assert event.event_type == TrainingEventType.PREPARING
        assert event.job_id == "job-002"
        assert event.source == "ultralytics"

    def test_create_process_started_event(self):
        """process_started event constructor"""
        event = create_process_started_event(
            job_id="job-003",
            timestamp=time.time(),
            pid=5678,
            source="custom_script",
        )

        assert event.event_type == TrainingEventType.PROCESS_STARTED
        assert event.payload["pid"] == 5678
        assert event.source == "custom_script"

    def test_create_console_output_event(self):
        """console_output event constructor"""
        event = create_console_output_event(
            job_id="job-004",
            timestamp=time.time(),
            message="Training started",
            stream="stdout",
            source="ultralytics",
        )

        assert event.event_type == TrainingEventType.CONSOLE_OUTPUT
        assert event.payload["message"] == "Training started"
        assert event.payload["stream"] == "stdout"

    def test_create_resource_sample_event(self):
        """resource_sample event constructor"""
        event = create_resource_sample_event(
            job_id="job-005",
            timestamp=time.time(),
            cpu_percent=45.2,
            memory_mb=1024,
            gpu_percent=80.5,
        )

        assert event.event_type == TrainingEventType.RESOURCE_SAMPLE
        assert event.payload["cpu_percent"] == 45.2
        assert event.payload["memory_mb"] == 1024
        assert event.payload["gpu_percent"] == 80.5

    def test_create_epoch_metrics_event(self):
        """epoch_metrics event constructor"""
        event = create_epoch_metrics_event(
            job_id="job-006",
            timestamp=time.time(),
            epoch=10,
            metrics={"loss": 0.5, "accuracy": 0.95},
        )

        assert event.event_type == TrainingEventType.EPOCH_METRICS
        assert event.payload["epoch"] == 10
        assert event.payload["metrics"]["loss"] == 0.5
        assert event.payload["metrics"]["accuracy"] == 0.95

    def test_create_checkpoint_saved_event(self):
        """checkpoint_saved event constructor"""
        event = create_checkpoint_saved_event(
            job_id="job-007",
            timestamp=time.time(),
            checkpoint_path="/tmp/checkpoint_10.pt",
            epoch=10,
        )

        assert event.event_type == TrainingEventType.CHECKPOINT_SAVED
        assert event.payload["checkpoint_path"] == "/tmp/checkpoint_10.pt"
        assert event.payload["epoch"] == 10

    def test_create_completed_event(self):
        """completed event constructor"""
        event = create_completed_event(
            job_id="job-008",
            timestamp=time.time(),
            source="ultralytics",
            exit_code=0,
        )

        assert event.event_type == TrainingEventType.COMPLETED
        assert event.payload["exit_code"] == 0
        assert event.source == "ultralytics"

    def test_create_failed_event(self):
        """failed event constructor"""
        event = create_failed_event(
            job_id="job-009",
            timestamp=time.time(),
            error="Out of memory",
            source="custom_script",
            exit_code=1,
        )

        assert event.event_type == TrainingEventType.FAILED
        assert event.payload["error"] == "Out of memory"
        assert event.payload["exit_code"] == 1

    def test_create_stopped_event(self):
        """stopped event constructor"""
        event = create_stopped_event(
            job_id="job-010",
            timestamp=time.time(),
            source="custom_script",
        )

        assert event.event_type == TrainingEventType.STOPPED
        assert event.source == "custom_script"

    def test_event_payload_extensibility(self):
        """Event payloads accept arbitrary kwargs"""
        event = create_process_started_event(
            job_id="job-011",
            timestamp=time.time(),
            pid=9999,
            source="test",
            custom_field="custom_value",
            another_field=123,
        )

        assert event.payload["pid"] == 9999
        assert event.payload["custom_field"] == "custom_value"
        assert event.payload["another_field"] == 123
