"""Tests: open_run_monitor removed — Run menu deleted, Training Center unifies entry."""

import pytest
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

FORBIDDEN_PATTERNS = [
    "open_run_monitor",
    "self.menus.run",
]


class TestOpenRunMonitorRemoved:
    """open_run_monitor was removed alongside the Run menu.
    Training Center is accessed via Train > Ultralytics menu.
    """

    def test_method_does_not_exist(self):
        """open_run_monitor method no longer exists on LabelingWidget."""
        import anylabeling.views.labeling.label_widget as lw_module
        assert not hasattr(lw_module.LabelingWidget, 'open_run_monitor'), (
            "open_run_monitor should be removed — Run menu is deleted"
        )

    def test_no_forbidden_legacy_patterns_in_widget(self):
        """Entire label_widget.py must not contain legacy Run menu patterns."""
        import anylabeling.views.labeling.label_widget as lw_module
        source = inspect.getsource(lw_module)
        for pattern in FORBIDDEN_PATTERNS:
            assert pattern not in source, f"Forbidden '{pattern}' found in label_widget"
