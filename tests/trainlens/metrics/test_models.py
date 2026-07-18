"""Tests for training metrics models."""

import time

from anylabeling.services.training_center.metrics.models import (
    MetricRunData,
    MetricSample,
    MetricSeries,
)


class TestMetricSample:
    def test_create_sample_minimal(self):
        s = MetricSample(job_id="job1", epoch=1, values={"loss": 0.5})
        assert s.job_id == "job1"
        assert s.epoch == 1
        assert s.values == {"loss": 0.5}
        assert s.step is None
        assert s.total_epochs is None
        # timestamp defaults to None (set externally at capture time)
        assert s.timestamp is None

    def test_create_sample_full(self):
        s = MetricSample(
            job_id="job1",
            epoch=5,
            step=100,
            total_epochs=50,
            values={"mAP50": 0.85, "loss": 0.32},
            timestamp=1234567890.0,
        )
        assert s.total_epochs == 50
        assert s.step == 100

    def test_sample_repr(self):
        s = MetricSample(job_id="j", epoch=3, values={"a": 1.0})
        r = repr(s)
        assert "j" in r and "3" in r


class TestMetricSeries:
    def test_series_fields(self):
        s = MetricSeries(name="loss", group="loss", display_name="Box Loss", points=[(1, 0.5)])
        assert s.name == "loss"
        assert s.group == "loss"
        assert s.display_name == "Box Loss"
        assert s.points == [(1, 0.5)]


class TestMetricRunData:
    def test_empty_run(self):
        r = MetricRunData(job_id="j", source="test")
        assert r.job_id == "j"
        assert r.samples == []
        assert r.total_epochs is None

    def test_add_samples(self):
        r = MetricRunData(job_id="j", source="test")
        r.samples.append(MetricSample(job_id="j", epoch=1, values={"loss": 1.0}))
        r.samples.append(MetricSample(job_id="j", epoch=2, values={"loss": 0.5}))
        assert len(r.samples) == 2

    def test_to_series_groups(self):
        r = MetricRunData(job_id="j", source="test", total_epochs=10)
        r.samples = [
            MetricSample(job_id="j", epoch=1, values={"train/box_loss": 1.2, "metrics/mAP50(B)": 0.5, "lr/pg0": 0.001}),
            MetricSample(job_id="j", epoch=2, values={"train/box_loss": 0.8, "metrics/mAP50(B)": 0.7, "lr/pg0": 0.0005}),
        ]
        series = r.to_series()
        assert len(series) >= 3

        # Verify grouping
        groups = {s.group for s in series}
        assert "loss" in groups
        assert "quality" in groups
        assert "learning_rate" in groups

        # Verify points
        loss_s = [s for s in series if s.name == "train/box_loss"][0]
        assert loss_s.points == [(1, 1.2), (2, 0.8)]

    def test_to_series_other_group(self):
        r = MetricRunData(job_id="j", source="test")
        r.samples = [MetricSample(job_id="j", epoch=1, values={"custom_metric": 0.9})]
        series = r.to_series()
        groups = {s.group for s in series}
        assert "other" in groups

    def test_to_series_empty(self):
        r = MetricRunData(job_id="j", source="test")
        assert r.to_series() == []

    def test_repr(self):
        r = MetricRunData(job_id="job1", source="results.csv", total_epochs=100)
        r.samples = [MetricSample(job_id="job1", epoch=1, values={"loss": 1.0})]
        rep = repr(r)
        assert "job1" in rep and "100" in rep and "1" in rep
