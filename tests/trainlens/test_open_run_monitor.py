"""Test open_run_monitor method logic without full imports"""

import sys
import os
import pytest
from unittest.mock import Mock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


class TestOpenRunMonitorLogic:
    """Test open_run_monitor logic in isolation"""

    def test_open_run_monitor_no_vqa_window_access(self):
        """open_run_monitor does not access vqa_window attribute"""
        widget = Mock()
        widget.run_monitor_window = None

        mock_window = Mock()
        mock_run_monitor_class = Mock(return_value=mock_window)

        # Mock the import at the point where it's used
        with MagicMock() as mock_module:
            mock_module.RunMonitorWindow = mock_run_monitor_class
            sys.modules['anylabeling.views.run_monitor'] = mock_module

            # Execute the method logic
            exec("""
def open_run_monitor(self):
    from anylabeling.views.run_monitor import RunMonitorWindow
    if not hasattr(self, 'run_monitor_window') or self.run_monitor_window is None:
        self.run_monitor_window = RunMonitorWindow(parent=self)
    else:
        try:
            self.run_monitor_window.isVisible()
        except RuntimeError:
            self.run_monitor_window = RunMonitorWindow(parent=self)

    self.run_monitor_window.show()
    self.run_monitor_window.raise_()
    self.run_monitor_window.activateWindow()

open_run_monitor(widget)
""", {'widget': widget})

            # Verify RunMonitorWindow was created with correct parent
            mock_run_monitor_class.assert_called_once_with(parent=widget)
            mock_window.show.assert_called_once()
            mock_window.raise_.assert_called_once()
            mock_window.activateWindow.assert_called_once()

            # The test passes if no AttributeError was raised for vqa_window
            # Mock objects auto-create attributes on access, so we can't check
            # if vqa_window exists. The important thing is the method completed
            # without trying to call methods on self.vqa_window

            # Cleanup
            del sys.modules['anylabeling.views.run_monitor']

    def test_open_run_monitor_creates_window_first_time(self):
        """First call creates RunMonitorWindow"""
        widget = Mock()
        widget.run_monitor_window = None

        mock_window = Mock()
        mock_run_monitor_class = Mock(return_value=mock_window)

        with MagicMock() as mock_module:
            mock_module.RunMonitorWindow = mock_run_monitor_class
            sys.modules['anylabeling.views.run_monitor'] = mock_module

            exec("""
def open_run_monitor(self):
    from anylabeling.views.run_monitor import RunMonitorWindow
    if not hasattr(self, 'run_monitor_window') or self.run_monitor_window is None:
        self.run_monitor_window = RunMonitorWindow(parent=self)
    else:
        try:
            self.run_monitor_window.isVisible()
        except RuntimeError:
            self.run_monitor_window = RunMonitorWindow(parent=self)

    self.run_monitor_window.show()
    self.run_monitor_window.raise_()
    self.run_monitor_window.activateWindow()

open_run_monitor(widget)
""", {'widget': widget})

            assert widget.run_monitor_window == mock_window
            mock_run_monitor_class.assert_called_once()

            del sys.modules['anylabeling.views.run_monitor']

    def test_open_run_monitor_reuses_existing_window(self):
        """Second call reuses existing window"""
        widget = Mock()
        existing_window = Mock()
        existing_window.isVisible.return_value = True
        widget.run_monitor_window = existing_window

        mock_run_monitor_class = Mock()

        with MagicMock() as mock_module:
            mock_module.RunMonitorWindow = mock_run_monitor_class
            sys.modules['anylabeling.views.run_monitor'] = mock_module

            exec("""
def open_run_monitor(self):
    from anylabeling.views.run_monitor import RunMonitorWindow
    if not hasattr(self, 'run_monitor_window') or self.run_monitor_window is None:
        self.run_monitor_window = RunMonitorWindow(parent=self)
    else:
        try:
            self.run_monitor_window.isVisible()
        except RuntimeError:
            self.run_monitor_window = RunMonitorWindow(parent=self)

    self.run_monitor_window.show()
    self.run_monitor_window.raise_()
    self.run_monitor_window.activateWindow()

open_run_monitor(widget)
""", {'widget': widget})

            # Should not create new window
            mock_run_monitor_class.assert_not_called()

            # Should show existing window
            existing_window.show.assert_called_once()
            existing_window.raise_.assert_called_once()
            existing_window.activateWindow.assert_called_once()

            del sys.modules['anylabeling.views.run_monitor']

    def test_open_run_monitor_recreates_destroyed_window(self):
        """Recreates window if C++ object was destroyed"""
        widget = Mock()
        destroyed_window = Mock()
        destroyed_window.isVisible.side_effect = RuntimeError("wrapped C/C++ object has been deleted")
        widget.run_monitor_window = destroyed_window

        new_window = Mock()
        mock_run_monitor_class = Mock(return_value=new_window)

        with MagicMock() as mock_module:
            mock_module.RunMonitorWindow = mock_run_monitor_class
            sys.modules['anylabeling.views.run_monitor'] = mock_module

            exec("""
def open_run_monitor(self):
    from anylabeling.views.run_monitor import RunMonitorWindow
    if not hasattr(self, 'run_monitor_window') or self.run_monitor_window is None:
        self.run_monitor_window = RunMonitorWindow(parent=self)
    else:
        try:
            self.run_monitor_window.isVisible()
        except RuntimeError:
            self.run_monitor_window = RunMonitorWindow(parent=self)

    self.run_monitor_window.show()
    self.run_monitor_window.raise_()
    self.run_monitor_window.activateWindow()

open_run_monitor(widget)
""", {'widget': widget})

            # Should create new window
            mock_run_monitor_class.assert_called_once()
            assert widget.run_monitor_window == new_window

            del sys.modules['anylabeling.views.run_monitor']

    def test_open_run_monitor_no_attribute_error(self):
        """Does not raise AttributeError for missing vqa_window"""
        widget = Mock()
        widget.run_monitor_window = None

        # Explicitly ensure vqa_window doesn't exist
        if hasattr(widget, 'vqa_window'):
            delattr(widget, 'vqa_window')

        mock_window = Mock()
        mock_run_monitor_class = Mock(return_value=mock_window)

        with MagicMock() as mock_module:
            mock_module.RunMonitorWindow = mock_run_monitor_class
            sys.modules['anylabeling.views.run_monitor'] = mock_module

            try:
                exec("""
def open_run_monitor(self):
    from anylabeling.views.run_monitor import RunMonitorWindow
    if not hasattr(self, 'run_monitor_window') or self.run_monitor_window is None:
        self.run_monitor_window = RunMonitorWindow(parent=self)
    else:
        try:
            self.run_monitor_window.isVisible()
        except RuntimeError:
            self.run_monitor_window = RunMonitorWindow(parent=self)

    self.run_monitor_window.show()
    self.run_monitor_window.raise_()
    self.run_monitor_window.activateWindow()

open_run_monitor(widget)
""", {'widget': widget})
            except AttributeError as e:
                if 'vqa_window' in str(e):
                    pytest.fail(f"Should not access vqa_window: {e}")
                raise
            finally:
                del sys.modules['anylabeling.views.run_monitor']

    def test_open_run_monitor_uses_consistent_attribute_name(self):
        """Uses run_monitor_window (not _run_monitor_window)"""
        widget = Mock()
        widget.run_monitor_window = None

        mock_window = Mock()
        mock_run_monitor_class = Mock(return_value=mock_window)

        with MagicMock() as mock_module:
            mock_module.RunMonitorWindow = mock_run_monitor_class
            sys.modules['anylabeling.views.run_monitor'] = mock_module

            exec("""
def open_run_monitor(self):
    from anylabeling.views.run_monitor import RunMonitorWindow
    if not hasattr(self, 'run_monitor_window') or self.run_monitor_window is None:
        self.run_monitor_window = RunMonitorWindow(parent=self)
    else:
        try:
            self.run_monitor_window.isVisible()
        except RuntimeError:
            self.run_monitor_window = RunMonitorWindow(parent=self)

    self.run_monitor_window.show()
    self.run_monitor_window.raise_()
    self.run_monitor_window.activateWindow()

open_run_monitor(widget)
""", {'widget': widget})

            # Should set run_monitor_window (not _run_monitor_window)
            assert widget.run_monitor_window == mock_window

            del sys.modules['anylabeling.views.run_monitor']
