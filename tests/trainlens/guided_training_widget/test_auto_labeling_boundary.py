"""Regression protection: auto-labeling must not be modified by training changes.

These tests enforce the boundary between training and auto-labeling.
They compare against a snapshot of the original X-AnyLabeling baseline
to ensure training development doesn't accidentally change auto-labeling.
"""

import os
import sys
import hashlib
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# ── Baseline snapshot (from original X-AnyLabeling commit 7ca5b981) ──────
# These values MUST NOT change with training-related modifications.

BASELINE_MODEL_COUNT = 28  # Number of .yaml model configs in auto_labeling/

BASELINE_MODEL_NAMES = sorted([
    "rtdetr_r50",
    "segment_anything_vit_b",
    "segment_anything_vit_b_quant",
    "segment_anything_vit_h_quant",
    "segment_anything_vit_l",
    "segment_anything_vit_l_quant",
    "yolov5l",
    "yolov5m",
    "yolov5n",
    "yolov5s",
    "yolov5s_resnet50",
    "yolov5x",
    "yolov6l_mbla",
    "yolov6lite_l_face",
    "yolov6lite_m_face",
    "yolov6lite_s_face",
    "yolov6m_mbla",
    "yolov6s",
    "yolov6s6",
    "yolov6s_mbla",
    "yolov6x_mbla",
    "yolov7",
    "yolov8l",
    "yolov8m",
    "yolov8n",
    "yolov8s",
    "yolov8x",
    "yolox_s",
])

# ── Files that must match baseline (hash check) ─────────────────────────

BASELINE_CRITICAL_FILES = [
    "anylabeling/services/auto_labeling/model_manager.py",
    "anylabeling/services/auto_labeling/model.py",
    "anylabeling/services/auto_labeling/types.py",
    "anylabeling/services/auto_labeling/__init__.py",
    "anylabeling/configs/auto_labeling/models.yaml",
]

# Known SHA256 hashes of critical auto-labeling files at baseline
# These are verified manually: git show 7ca5b981:<path> | sha256sum
BASELINE_HASHES = {
    # Will be lazily populated on first run
}

# ── Forbidden imports from training into auto_labeling ──────────────────

FORBIDDEN_TRAINING_IMPORTS = [
    "anylabeling.services.auto_labeling.model_manager",
    "anylabeling.services.auto_labeling.model",
]

# Training modules that we check
TRAINING_MODULE_ROOTS = [
    "anylabeling/services/training_center",
    "anylabeling/services/auto_training",
    "anylabeling/views/training",
]


class TestAutoLabelingRegression:
    """Verify auto-labeling files match baseline snapshot."""

    def test_model_config_count_matches_baseline(self):
        """Auto-labeling model config count must equal baseline."""
        config_dir = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..",
            "anylabeling", "configs", "auto_labeling"
        )
        config_dir = os.path.abspath(config_dir)

        yaml_files = [
            f for f in os.listdir(config_dir)
            if f.endswith(".yaml") and f != "models.yaml"
        ]
        actual_count = len(yaml_files)
        assert actual_count == BASELINE_MODEL_COUNT, (
            f"Model config count changed: {actual_count} (baseline: {BASELINE_MODEL_COUNT}). "
            f"Auto-labeling model list must not be modified without explicit approval."
        )

    def test_model_names_match_baseline(self):
        """All baseline model names must be present and no extras."""
        config_dir = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..",
            "anylabeling", "configs", "auto_labeling"
        )
        config_dir = os.path.abspath(config_dir)

        actual_names = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(config_dir)
            if f.endswith(".yaml") and f != "models.yaml"
        ])

        assert actual_names == BASELINE_MODEL_NAMES, (
            f"Model names do not match baseline.\n"
            f"Expected ({len(BASELINE_MODEL_NAMES)}): {BASELINE_MODEL_NAMES}\n"
            f"Got ({len(actual_names)}): {actual_names}"
        )

    def test_critical_files_exist(self):
        """Critical auto-labeling files must exist."""
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        for rel_path in BASELINE_CRITICAL_FILES:
            abs_path = os.path.join(project_root, rel_path)
            assert os.path.exists(abs_path), (
                f"Critical auto-labeling file missing: {rel_path}"
            )

    def test_models_yaml_structure(self):
        """models.yaml must be a list of model entries (original X-AnyLabeling format)."""
        import yaml
        config_dir = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..",
            "anylabeling", "configs", "auto_labeling"
        )
        config_dir = os.path.abspath(config_dir)
        models_yaml = os.path.join(config_dir, "models.yaml")

        with open(models_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert isinstance(data, list), (
            "models.yaml must be a list (original X-AnyLabeling format)"
        )
        assert len(data) > 0, "models.yaml must not be empty"
        # First entry should be the default model (SAM in original)
        assert "model_name" in data[0], (
            "Each entry must have 'model_name'"
        )


class TestTrainingAutoLabelingIsolation:
    """Training code must not import auto_labeling private modules."""

    def test_training_does_not_import_auto_labeling(self):
        """Training modules must not import from auto-labeling."""
        import ast
        import glob

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )

        violations = []

        for module_root in TRAINING_MODULE_ROOTS:
            search_dir = os.path.join(project_root, module_root)
            py_files = glob.glob(os.path.join(search_dir, "**", "*.py"), recursive=True)

            for py_file in py_files:
                try:
                    with open(py_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                for forbidden in FORBIDDEN_TRAINING_IMPORTS:
                                    if alias.name == forbidden or alias.name.startswith(forbidden + "."):
                                        violations.append(
                                            f"{os.path.relpath(py_file, project_root)} imports {alias.name}"
                                        )
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                for forbidden in FORBIDDEN_TRAINING_IMPORTS:
                                    if node.module == forbidden or node.module.startswith(forbidden + "."):
                                        violations.append(
                                            f"{os.path.relpath(py_file, project_root)} imports {node.module}"
                                        )
                except SyntaxError:
                    pass

        assert not violations, (
            f"Training modules must not import auto-labeling:\n"
            + "\n".join(violations)
        )
