"""Tests for metrics parsers (results.csv and metrics.jsonl)."""

import csv
import json
import tempfile
from pathlib import Path

from anylabeling.services.training_center.metrics.parser import (
    parse_metrics_jsonl,
    parse_results_csv,
)


class TestParseResultsCSV:
    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("")
        try:
            data = parse_results_csv(Path(f.name), job_id="test")
            assert data.job_id == "test"
            assert data.source == "results.csv"
            assert data.samples == []
        finally:
            Path(f.name).unlink(missing_ok=True)

    def test_single_epoch(self):
        content = """epoch,train/box_loss,train/cls_loss,metrics/mAP50(B),metrics/mAP50-95(B)
1,1.5,0.8,0.65,0.45
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
        try:
            data = parse_results_csv(Path(f.name))
            assert len(data.samples) == 1
            s = data.samples[0]
            assert s.epoch == 1
            assert s.values["train/box_loss"] == 1.5
            assert s.values["metrics/mAP50(B)"] == 0.65
        finally:
            Path(f.name).unlink(missing_ok=True)

    def test_multiple_epochs(self):
        content = """epoch,train/box_loss,metrics/mAP50(B)
1,1.5,0.5
2,1.2,0.6
3,0.9,0.7
4,0.7,0.75
5,0.5,0.8
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
        try:
            data = parse_results_csv(Path(f.name))
            assert len(data.samples) == 5
            assert [s.epoch for s in data.samples] == [1, 2, 3, 4, 5]
            # Verify losss descending
            losses = [s.values["train/box_loss"] for s in data.samples]
            assert losses == [1.5, 1.2, 0.9, 0.7, 0.5]
        finally:
            Path(f.name).unlink(missing_ok=True)

    def test_dedup_same_epoch(self):
        content = """epoch,train/box_loss
1,1.5
1,1.4
2,1.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
        try:
            data = parse_results_csv(Path(f.name))
            # Should only keep first occurrence per epoch
            assert len(data.samples) == 2
            assert data.samples[0].epoch == 1
            assert data.samples[1].epoch == 2
        finally:
            Path(f.name).unlink(missing_ok=True)

    def test_skip_nan_inf(self):
        content = """epoch,train/box_loss,metrics/mAP50(B)
1,NaN,0.5
2,Inf,-Inf
3,0.9,NaN
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
        try:
            data = parse_results_csv(Path(f.name))
            # Row 2 (Inf,-Inf) has NO valid values → skipped entirely
            # Rows 1 and 3 each have 1 valid value → kept
            assert len(data.samples) == 2
            # Row 1: NaN, 0.5 → kept
            assert data.samples[0].epoch == 1
            assert "train/box_loss" not in data.samples[0].values  # NaN
            assert data.samples[0].values["metrics/mAP50(B)"] == 0.5
            # Row 3: 0.9, NaN → kept
            assert data.samples[1].epoch == 3
            assert data.samples[1].values["train/box_loss"] == 0.9
            assert "metrics/mAP50(B)" not in data.samples[1].values  # NaN
        finally:
            Path(f.name).unlink(missing_ok=True)

    def test_nonexistent_file(self):
        data = parse_results_csv(Path("/nonexistent/results.csv"), job_id="test")
        assert data.job_id == "test"
        assert data.samples == []

    def test_usv_columns(self):
        """Columns with spaces: (B) suffix, x/y, lr, val/, lr/pg etc."""
        content = """epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
1,1.2,0.8,0.9,0.7,0.6,0.5,0.4,1.3,0.9,1.0,0.001,0.001,0.001
10,0.5,0.3,0.4,0.9,0.85,0.8,0.7,0.6,0.4,0.5,0.0005,0.0005,0.0005
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
        try:
            data = parse_results_csv(Path(f.name))
            assert len(data.samples) == 2

            s1 = data.samples[0]
            assert s1.values["train/box_loss"] == 1.2
            assert s1.values["metrics/mAP50(B)"] == 0.5
            assert s1.values["lr/pg0"] == 0.001

            s2 = data.samples[1]
            assert s2.values["metrics/mAP50-95(B)"] == 0.7
            assert s2.values["val/box_loss"] == 0.6
            assert s2.values["lr/pg2"] == 0.0005
        finally:
            Path(f.name).unlink(missing_ok=True)


class TestParseMetricsJSONL:
    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
        try:
            data = parse_metrics_jsonl(Path(f.name), job_id="test")
            assert data.job_id == "test"
            assert data.source == "metrics.jsonl"
            assert data.samples == []
        finally:
            Path(f.name).unlink(missing_ok=True)

    def test_valid_jsonl(self):
        lines = [
            json.dumps({"epoch": 1, "metrics": {"loss": 1.5, "mAP50": 0.5}}),
            json.dumps({"epoch": 2, "metrics": {"loss": 1.2, "mAP50": 0.6}}),
            json.dumps({"epoch": 3, "metrics": {"loss": 0.9, "mAP50": 0.7}}),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(lines))
        try:
            data = parse_metrics_jsonl(Path(f.name))
            assert len(data.samples) == 3
            assert [s.epoch for s in data.samples] == [1, 2, 3]
            assert data.samples[0].values["loss"] == 1.5
        finally:
            Path(f.name).unlink(missing_ok=True)

    def test_malformed_lines_skipped(self):
        lines = [
            json.dumps({"epoch": 1, "metrics": {"loss": 1.0}}),
            "not valid json {{{",
            json.dumps({"epoch": 2, "metrics": {"loss": 0.5}}),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(lines))
        try:
            data = parse_metrics_jsonl(Path(f.name))
            assert len(data.samples) == 2
            assert data.samples[0].epoch == 1
            assert data.samples[1].epoch == 2
        finally:
            Path(f.name).unlink(missing_ok=True)

    def test_nonexistent_file(self):
        data = parse_metrics_jsonl(Path("/nonexistent/metrics.jsonl"), job_id="test")
        assert data.samples == []
