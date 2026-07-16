"""Regression tests for open_run_monitor delegation.

open_run_monitor() now delegates to open_training_center("custom").
"""

import pytest
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

FORBIDDEN_PATTERNS = [
    "vqa_window",
    "_run_monitor_window",
    "run_monitor_window",
    "RuntimeError",
    "RunMonitorWindow(",
]


def get_open_run_monitor_source():
    import anylabeling.views.labeling.label_widget as lw_module
    return inspect.getsource(lw_module.LabelingWidget.open_run_monitor)


class TestOpenRunMonitorDelegates:

    def test_no_vqa_window_access(self):
        source = get_open_run_monitor_source()
        assert "vqa_window" not in source

    def test_no_legacy_run_monitor_window_attr(self):
        source = get_open_run_monitor_source()
        assert "run_monitor_window" not in source
        assert "_run_monitor_window" not in source

    def test_no_legacy_runtime_error_catch(self):
        source = get_open_run_monitor_source()
        assert "RuntimeError" not in source

    def test_delegates_to_open_training_center_custom(self):
        source = get_open_run_monitor_source()
        assert "open_training_center" in source
        assert '"custom"' in source or "'custom'" in source

    def test_no_legacy_runmonitorwindow_construction(self):
        source = get_open_run_monitor_source()
        assert "RunMonitorWindow(" not in source

    def test_single_definition(self):
        import anylabeling.views.labeling.label_widget as lw_module
        count = sum(1 for n in dir(lw_module.LabelingWidget) if n == "open_run_monitor")
        assert count == 1

    def test_no_forbidden_patterns(self):
        source = get_open_run_monitor_source()
        for pattern in FORBIDDEN_PATTERNS:
            assert pattern not in source, f"Forbidden '{pattern}' found"
