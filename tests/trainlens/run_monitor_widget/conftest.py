"""Pytest configuration for run_monitor_widget tests

These tests need actual PyQt6 for QApplication and widgets, not mocks.
Override parent conftest to prevent PyQt6 mocking.
"""

import sys

# Remove PyQt6 mocks if they were added by parent conftest
for module in list(sys.modules.keys()):
    if module.startswith('PyQt6'):
        del sys.modules[module]

# These tests will import real PyQt6 - if not available, they should be skipped
pytest_plugins = []
