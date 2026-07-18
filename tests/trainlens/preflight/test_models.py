"""Tests for preflight models (severity, result, fingerprint)."""

from anylabeling.services.training_center.preflight.models import (
    PreflightSeverity, PreflightIssue, PreflightResult,
    GuidedPreflightContext, CustomPreflightContext,
)


class TestPreflightSeverity:
    def test_pass_warning_error_values(self):
        assert PreflightSeverity.PASS.value == "pass"
        assert PreflightSeverity.WARNING.value == "warning"
        assert PreflightSeverity.ERROR.value == "error"


class TestPreflightResult:
    def test_empty_result(self):
        r = PreflightResult(mode="test")
        assert r.pass_count == 0
        assert r.warning_count == 0
        assert r.error_count == 0
        assert r.can_start is True  # No errors → can start
        assert r.has_errors is False
        assert r.has_warnings is False

    def test_error_blocks_start(self):
        r = PreflightResult(mode="test")
        r.add(PreflightIssue(code="E1", severity=PreflightSeverity.ERROR, title="err", message="x"))
        assert r.has_errors is True
        assert r.can_start is False

    def test_warning_allows_start(self):
        r = PreflightResult(mode="test")
        r.add(PreflightIssue(code="W1", severity=PreflightSeverity.WARNING, title="warn", message="x"))
        assert r.has_warnings is True
        assert r.has_errors is False
        assert r.can_start is True

    def test_cancelled_blocks_start(self):
        r = PreflightResult(mode="test", cancelled=True)
        assert r.can_start is False

    def test_summary_counts(self):
        r = PreflightResult(mode="test")
        r.add(PreflightIssue(code="P1", severity=PreflightSeverity.PASS, title="p", message="x"))
        r.add(PreflightIssue(code="W1", severity=PreflightSeverity.WARNING, title="w", message="x"))
        r.add(PreflightIssue(code="E1", severity=PreflightSeverity.ERROR, title="e", message="x"))
        assert r.pass_count == 1
        assert r.warning_count == 1
        assert r.error_count == 1

    def test_can_start_with_multiple_errors(self):
        r = PreflightResult(mode="test")
        for i in range(3):
            r.add(PreflightIssue(code=f"E{i}", severity=PreflightSeverity.ERROR, title="e", message="x"))
        assert r.error_count == 3
        assert r.can_start is False

    def test_errors_and_warnings_filtered(self):
        r = PreflightResult(mode="test")
        r.add(PreflightIssue(code="E1", severity=PreflightSeverity.ERROR, title="e1", message="x"))
        r.add(PreflightIssue(code="W1", severity=PreflightSeverity.WARNING, title="w1", message="x"))
        r.add(PreflightIssue(code="P1", severity=PreflightSeverity.PASS, title="p1", message="x"))
        assert len(r.errors()) == 1
        assert len(r.warnings()) == 1


class TestFingerprint:
    def test_stable_fingerprint(self):
        ctx = GuidedPreflightContext(
            task_type="detect", model_path="model.pt", dataset_yaml="data.yaml",
            epochs=100, batch=16, imgsz=640, device="cpu", output_dir="/tmp/out",
        )
        fp1 = ctx.fingerprint()
        fp2 = ctx.fingerprint()
        assert fp1 == fp2

    def test_config_change_changes_fingerprint(self):
        ctx1 = GuidedPreflightContext(epochs=100)
        ctx2 = GuidedPreflightContext(epochs=200)
        assert ctx1.fingerprint() != ctx2.fingerprint()

    def test_custom_fingerprint(self):
        ctx = CustomPreflightContext(
            project_dir="/tmp", python_path="/usr/bin/python3",
            script_path="/tmp/train.py",
        )
        fp = ctx.fingerprint()
        assert len(fp) == 16
