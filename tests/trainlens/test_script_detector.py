"""Test script detector"""

import pytest
from pathlib import Path

from anylabeling.services.run_monitor.script_detector import ScriptDetector


def test_script_detector_filename_match(tmp_path):
    """Test detection by filename with entry point"""
    detector = ScriptDetector()

    # Create a training script with entry point to reach threshold
    train_script = tmp_path / "train.py"
    train_script.write_text("""
import argparse

if __name__ == "__main__":
    print('Training started')
""")

    result = detector.detect(train_script)
    assert result is not None
    assert "filename_match" in result.reasons


def test_script_detector_ultralytics(tmp_path):
    """Test Ultralytics framework detection"""
    detector = ScriptDetector()

    script = tmp_path / "train_yolo.py"
    script.write_text("""
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolov8n.pt')
    model.train(data='coco.yaml', epochs=100)
""")

    result = detector.detect(script)
    assert result is not None
    assert result.framework == "ultralytics"
    assert "imports_ultralytics" in result.reasons


def test_script_detector_pytorch(tmp_path):
    """Test PyTorch framework detection"""
    detector = ScriptDetector()

    script = tmp_path / "train.py"
    script.write_text("""
import torch
from torch.utils.data import DataLoader

model = torch.nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(100):
    loss.backward()
    optimizer.step()
""")

    result = detector.detect(script)
    assert result is not None
    assert result.framework == "pytorch"
    assert "contains_training_code" in result.reasons


def test_script_detector_non_training_script(tmp_path):
    """Test that non-training scripts are not detected"""
    detector = ScriptDetector()

    script = tmp_path / "utils.py"
    script.write_text("""
def helper_function():
    return 42
""")

    result = detector.detect(script)
    assert result is None


def test_script_detector_invalid_file():
    """Test handling of invalid files"""
    detector = ScriptDetector()

    # Non-existent file
    result = detector.detect(Path("/nonexistent/file.py"))
    assert result is None

    # Non-Python file
    result = detector.detect(Path("test.txt"))
    assert result is None
