"""Regression tests for Run Monitor window lifecycle (AttributeError fix).

These tests verify the source code of open_run_monitor() in label_widget.py
to confirm the fix from commit 18efa05d is applied:
  - No vqa_window references
  - Uses run_monitor_window (not _run_monitor_window)
  - Has RuntimeError catch for destroyed C++ Qt objects
  - Calls show(), raise_(), activateWindow()
"""

import pytest
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Source reference for the correct implementation (from commit 18efa05d)
EXPECTED_PATTERNS = [
    "hasattr(self, 'run_monitor_window')",
    "run_monitor_window is None",
    "self.run_monitor_window",
    "RuntimeError",
    "isVisible()",
    ".show()",
    ".raise_()",
    ".activateWindow()",
]

FORBIDDEN_PATTERNS = [
    "vqa_window",
    "_run_monitor_window",
]


def get_open_run_monitor_source():
    """Extract the source code of LabelingWidget.open_run_monitor."""
    import anylabeling.views.labeling.label_widget as lw_module
    return inspect.getsource(lw_module.LabelingWidget.open_run_monitor)


class TestOpenRunMonitorSource:

    def test_no_vqa_window_access(self):
        """Source must NOT reference vqa_window"""
        source = get_open_run_monitor_source()
        assert "vqa_window" not in source, (
            f"vqa_window reference still present:\n{source}"
        )

    def test_uses_correct_attribute_name(self):
        """Source must use run_monitor_window, not _run_monitor_window"""
        source = get_open_run_monitor_source()
        assert "self.run_monitor_window" in source, (
            f"Missing correct attribute name:\n{source}"
        )
        assert "self._run_monitor_window" not in source, (
            f"Legacy _run_monitor_window still in use:\n{source}"
        )

    def test_hasattr_guard_present(self):
        """Source must guard with hasattr check"""
        source = get_open_run_monitor_source()
        assert "hasattr(self, 'run_monitor_window')" in source, (
            f"Missing hasattr guard:\n{source}"
        )

    def test_runtime_error_caught_for_destroyed_window(self):
        """Source must catch RuntimeError for destroyed C++ Qt objects"""
        source = get_open_run_monitor_source()
        assert "RuntimeError" in source, (
            f"Missing RuntimeError catch:\n{source}"
        )

    def test_is_visible_check_present(self):
        """Source must call isVisible() to detect destroyed windows"""
        source = get_open_run_monitor_source()
        assert "isVisible()" in source, (
            f"Missing isVisible() call:\n{source}"
        )

    def test_recreates_on_destroy(self):
        """Source must recreate window when RuntimeError caught"""
        source = get_open_run_monitor_source()
        # Inside the except RuntimeError block, it should reassign
        assert "RunMonitorWindow(parent=self)" in source, (
            f"Missing window recreation:\n{source}"
        )

    def test_shows_window(self):
        """Source must call show()"""
        source = get_open_run_monitor_source()
        assert ".show()" in source

    def test_raises_window(self):
        """Source must call raise_()"""
        source = get_open_run_monitor_source()
        assert ".raise_()" in source

    def test_activates_window(self):
        """Source must call activateWindow()"""
        source = get_open_run_monitor_source()
        assert ".activateWindow()" in source

    def test_local_import_pattern(self):
        """Source uses local 'from import' for RunMonitorWindow"""
        source = get_open_run_monitor_source()
        assert "from anylabeling.views.run_monitor import RunMonitorWindow" in source, (
            f"Missing local import of RunMonitorWindow:\n{source}"
        )

    def test_no_duplicate_vqa_references(self):
        """Source must NOT have any vqa_window references anywhere"""
        source = get_open_run_monitor_source()
        for line in source.splitlines():
            if "vqa_window" in line and not line.strip().startswith('"""'):
                assert False, f"vqa_window reference found in line: {line.strip()}"


class TestOpenRunMonitorIntegration:
    """Light integration: verify the _source_ of the method matches the expected
    patterns from the original fix in commit 18efa05d."""

    def test_all_expected_patterns_present(self):
        """Every expected code pattern exists in the source"""
        source = get_open_run_monitor_source()
        for pattern in EXPECTED_PATTERNS:
            assert pattern in source, (
                f"Expected pattern '{pattern}' not found in:\n{source}"
            )

    def test_no_forbidden_patterns(self):
        """No forbidden patterns exist in the source"""
        source = get_open_run_monitor_source()
        for pattern in FORBIDDEN_PATTERNS:
            assert pattern not in source, (
                f"Forbidden pattern '{pattern}' found in:\n{source}"
            )

    def test_single_definition(self):
        """Only one open_run_monitor method in LabelingWidget"""
        import anylabeling.views.labeling.label_widget as lw_module
        count = 0
        for name in dir(lw_module.LabelingWidget):
            if name == "open_run_monitor":
                count += 1
        assert count == 1, f"Found {count} open_run_monitor definitions, expected 1"
