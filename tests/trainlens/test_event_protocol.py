"""Test event protocol"""

import json
import time
from anylabeling.services.run_monitor.event_protocol import (
    EventProtocol,
    create_run_created_event,
    create_epoch_metrics_event,
)


def test_parse_valid_event():
    """Test parsing valid structured event"""
    line = json.dumps({
        "schema_version": 1,
        "run_id": "run_123",
        "event": "epoch_metrics",
        "timestamp": 1234567890.0,
        "payload": {"epoch": 1, "loss": 0.5}
    })

    event = EventProtocol.parse_line(line)
    assert event is not None
    assert event.run_id == "run_123"
    assert event.event == "epoch_metrics"
    assert event.payload["loss"] == 0.5


def test_parse_regular_log_line():
    """Test that regular log lines return None"""
    line = "Training started..."

    event = EventProtocol.parse_line(line)
    assert event is None


def test_parse_invalid_json():
    """Test that invalid JSON returns None"""
    line = '{"invalid": json}'

    event = EventProtocol.parse_line(line)
    assert event is None


def test_parse_json_without_schema():
    """Test that JSON without schema_version returns None"""
    line = json.dumps({"some": "data"})

    event = EventProtocol.parse_line(line)
    assert event is None


def test_create_run_created_event():
    """Test creating run_created event"""
    timestamp = time.time()
    event = create_run_created_event(
        run_id="run_123",
        timestamp=timestamp,
        script="train.py",
        python="/usr/bin/python3"
    )

    assert event.run_id == "run_123"
    assert event.event == EventProtocol.EVENT_RUN_CREATED
    assert event.timestamp == timestamp
    assert event.payload["script"] == "train.py"


def test_create_epoch_metrics_event():
    """Test creating epoch_metrics event"""
    timestamp = time.time()
    metrics = {"loss": 0.5, "accuracy": 0.95}

    event = create_epoch_metrics_event(
        run_id="run_123",
        timestamp=timestamp,
        epoch=10,
        metrics=metrics
    )

    assert event.event == EventProtocol.EVENT_EPOCH_METRICS
    assert event.payload["epoch"] == 10
    assert event.payload["loss"] == 0.5
    assert event.payload["accuracy"] == 0.95


def test_format_event_for_logging():
    """Test formatting event as JSON string"""
    timestamp = time.time()
    event = create_run_created_event(
        run_id="run_123",
        timestamp=timestamp,
        script="train.py"
    )

    json_str = EventProtocol.format_event_for_logging(event)
    assert isinstance(json_str, str)

    # Should be valid JSON
    parsed = json.loads(json_str)
    assert parsed["schema_version"] == 1
    assert parsed["run_id"] == "run_123"
