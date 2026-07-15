"""Test workspace scanner"""

import pytest
from pathlib import Path

from anylabeling.services.run_monitor.workspace_scanner import WorkspaceScanner


def test_workspace_scanner_basic(tmp_path):
    """Test basic workspace scanning"""
    scanner = WorkspaceScanner()

    # Create some Python files
    (tmp_path / "train.py").write_text("from ultralytics import YOLO")
    (tmp_path / "utils.py").write_text("def helper(): pass")

    workspace = scanner.scan(tmp_path)

    assert workspace is not None
    assert workspace.path == tmp_path
    assert len(workspace.detected_scripts) > 0
    assert len(workspace.detected_environments) > 0


def test_workspace_scanner_excludes_dirs(tmp_path):
    """Test that excluded directories are skipped"""
    scanner = WorkspaceScanner()

    # Create files in excluded directories
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "train.py").write_text("from ultralytics import YOLO")

    pycache_dir = tmp_path / "__pycache__"
    pycache_dir.mkdir()
    (pycache_dir / "train.py").write_text("from ultralytics import YOLO")

    # Create valid training script
    (tmp_path / "train.py").write_text("from ultralytics import YOLO")

    workspace = scanner.scan(tmp_path)

    # Should only find the root train.py, not the ones in excluded dirs
    script_paths = [s.path for s in workspace.detected_scripts]
    assert tmp_path / "train.py" in script_paths
    assert git_dir / "train.py" not in script_paths
    assert pycache_dir / "train.py" not in script_paths


def test_workspace_scanner_cancel(tmp_path):
    """Test scanner cancellation"""
    scanner = WorkspaceScanner()

    # Create many files
    for i in range(10):
        (tmp_path / f"file{i}.py").write_text("print('hello')")

    # Start scan and immediately cancel
    scanner.cancel()
    workspace = scanner.scan(tmp_path)

    # Should still return a workspace object
    assert workspace is not None


def test_workspace_scanner_invalid_path():
    """Test handling of invalid workspace path"""
    scanner = WorkspaceScanner()

    with pytest.raises(ValueError):
        scanner.scan(Path("/nonexistent/path"))
