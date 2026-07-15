"""Event protocol for Run Monitor

Defines structured event format for training communication.
"""

import json
from typing import Dict, Optional

from .models import TrainingEvent


class EventProtocol:
    """Handles structured training event parsing and creation"""

    SCHEMA_VERSION = 1

    # Event types
    EVENT_RUN_CREATED = "run_created"
    EVENT_ENVIRONMENT_DETECTED = "environment_detected"
    EVENT_PROCESS_STARTED = "process_started"
    EVENT_CONSOLE_OUTPUT = "console_output"
    EVENT_BATCH_PROGRESS = "batch_progress"
    EVENT_EPOCH_METRICS = "epoch_metrics"
    EVENT_VALIDATION_STARTED = "validation_started"
    EVENT_VALIDATION_COMPLETED = "validation_completed"
    EVENT_CHECKPOINT_SAVED = "checkpoint_saved"
    EVENT_RESOURCE_SAMPLE = "resource_sample"
    EVENT_PROCESS_COMPLETED = "process_completed"
    EVENT_PROCESS_FAILED = "process_failed"
    EVENT_PROCESS_STOPPED = "process_stopped"

    @staticmethod
    def parse_line(line: str) -> Optional[TrainingEvent]:
        """
        Parse a line from stdout to check if it's a structured event.

        Args:
            line: Line from stdout

        Returns:
            TrainingEvent if line is a valid event, None otherwise
        """
        line = line.strip()
        if not line.startswith("{"):
            return None

        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                return None

            # Check for schema_version to identify as event
            if "schema_version" not in data:
                return None

            return TrainingEvent.from_dict(data)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def create_event(
        run_id: str, event_type: str, payload: Dict, timestamp: float
    ) -> TrainingEvent:
        """
        Create a training event.

        Args:
            run_id: Run identifier
            event_type: Event type
            payload: Event-specific data
            timestamp: Unix timestamp

        Returns:
            TrainingEvent object
        """
        return TrainingEvent(
            schema_version=EventProtocol.SCHEMA_VERSION,
            run_id=run_id,
            event=event_type,
            timestamp=timestamp,
            payload=payload,
        )

    @staticmethod
    def format_event_for_logging(event: TrainingEvent) -> str:
        """
        Format event as JSON line for logging.

        Args:
            event: TrainingEvent to format

        Returns:
            JSON string
        """
        return json.dumps(event.to_dict())


# Helper functions for common event creation

def create_run_created_event(run_id: str, timestamp: float, **kwargs) -> TrainingEvent:
    """Create run_created event"""
    return EventProtocol.create_event(
        run_id=run_id,
        event_type=EventProtocol.EVENT_RUN_CREATED,
        payload=kwargs,
        timestamp=timestamp,
    )


def create_process_started_event(
    run_id: str, timestamp: float, pid: int, **kwargs
) -> TrainingEvent:
    """Create process_started event"""
    payload = {"pid": pid}
    payload.update(kwargs)
    return EventProtocol.create_event(
        run_id=run_id,
        event_type=EventProtocol.EVENT_PROCESS_STARTED,
        payload=payload,
        timestamp=timestamp,
    )


def create_epoch_metrics_event(
    run_id: str, timestamp: float, epoch: int, metrics: Dict
) -> TrainingEvent:
    """Create epoch_metrics event"""
    payload = {"epoch": epoch}
    payload.update(metrics)
    return EventProtocol.create_event(
        run_id=run_id,
        event_type=EventProtocol.EVENT_EPOCH_METRICS,
        payload=payload,
        timestamp=timestamp,
    )


def create_process_completed_event(
    run_id: str, timestamp: float, exit_code: int, duration: float
) -> TrainingEvent:
    """Create process_completed event"""
    return EventProtocol.create_event(
        run_id=run_id,
        event_type=EventProtocol.EVENT_PROCESS_COMPLETED,
        payload={"exit_code": exit_code, "duration": duration},
        timestamp=timestamp,
    )


def create_process_failed_event(
    run_id: str, timestamp: float, exit_code: int, error: str
) -> TrainingEvent:
    """Create process_failed event"""
    return EventProtocol.create_event(
        run_id=run_id,
        event_type=EventProtocol.EVENT_PROCESS_FAILED,
        payload={"exit_code": exit_code, "error": error},
        timestamp=timestamp,
    )
