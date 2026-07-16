"""Global pytest configuration for all tests

Mocks UI dependencies to allow service/logic tests to run without full GUI stack.
"""

import os
import sys
from unittest.mock import MagicMock

# Skip view imports in anylabeling package to avoid UI dependencies
os.environ['ANYLABELING_SKIP_VIEW_IMPORTS'] = '1'

# Mock all UI and external dependencies before any other imports
sys.modules['PyQt6'] = MagicMock()
sys.modules['PyQt6.QtCore'] = MagicMock()
sys.modules['PyQt6.QtWidgets'] = MagicMock()
sys.modules['PyQt6.QtGui'] = MagicMock()
sys.modules['natsort'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['anthropic'] = MagicMock()
sys.modules['opencv-python'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['jsonlines'] = MagicMock()
sys.modules['onnxruntime'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
