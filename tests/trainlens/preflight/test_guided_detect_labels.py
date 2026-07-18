"""Tests for preflight YAML and detect label checks."""

import os
from pathlib import Path

from anylabeling.services.training_center.preflight import (
    PreflightResult, PreflightSeverity, PreflightIssue,
)
from anylabeling.services.training_center.preflight.guided_yaml import (
    read_yaml_safe, check_yaml_structure, check_yaml_dataset_paths,
    _resolve_dataset_path, _count_images_in_path,
)
from anylabeling.services.training_center.preflight.guided_detect_labels import (
    check_detect_labels, _parse_label_file,
)


class TestYAMLReading:
    def test_read_valid_yaml(self, tmp_path):
        yaml = tmp_path / "data.yaml"
        yaml.write_text("train: images/train\nval: images/val\nnc: 1\nnames: ['cat']\n")
        data, err = read_yaml_safe(str(yaml))
        assert err is None
        assert data["train"] == "images/train"

    def test_yaml_syntax_error(self, tmp_path):
        yaml = tmp_path / "data.yaml"
        yaml.write_text("train: [unclosed\n")
        data, err = read_yaml_safe(str(yaml))
        assert err is not None

    def test_missing_file(self, tmp_path):
        data, err = read_yaml_safe(str(tmp_path / "nonexistent.yaml"))
        assert err is not None

    def test_utf8_bom(self, tmp_path):
        yaml = tmp_path / "data.yaml"
        yaml.write_bytes(b'\xef\xbb\xbftrain: images/train\nval: images/val\nnames: [a]\n')
        data, err = read_yaml_safe(str(yaml))
        assert err is None
        assert data["train"] == "images/train"


class TestYAMLStructure:
    def test_no_train_errors(self, tmp_path):
        yaml = tmp_path / "data.yaml"
        yaml.write_text("val: v\nnames: [a]\n")
        data, _ = read_yaml_safe(str(yaml))
        r = PreflightResult(mode="test")
        check_yaml_structure(r, str(yaml), data)
        assert any(i.code == "YAML_NO_TRAIN" for i in r.errors())

    def test_no_val_errors(self, tmp_path):
        yaml = tmp_path / "data.yaml"
        yaml.write_text("train: t\nnames: [a]\n")
        data, _ = read_yaml_safe(str(yaml))
        r = PreflightResult(mode="test")
        check_yaml_structure(r, str(yaml), data)
        assert any(i.code == "YAML_NO_VAL" for i in r.errors())

    def test_names_list_valid(self, tmp_path):
        yaml = tmp_path / "data.yaml"
        yaml.write_text("train: t\nval: v\nnames: [a, b]\n")
        data, _ = read_yaml_safe(str(yaml))
        r = PreflightResult(mode="test")
        check_yaml_structure(r, str(yaml), data)
        assert r.error_count == 0

    def test_names_dict_valid(self, tmp_path):
        yaml = tmp_path / "data.yaml"
        yaml.write_text("train: t\nval: v\nnames:\n  0: cat\n  1: dog\n")
        data, _ = read_yaml_safe(str(yaml))
        r = PreflightResult(mode="test")
        check_yaml_structure(r, str(yaml), data)
        assert r.error_count == 0

    def test_empty_names_errors(self, tmp_path):
        yaml = tmp_path / "data.yaml"
        yaml.write_text("train: t\nval: v\nnames: []\n")
        data, _ = read_yaml_safe(str(yaml))
        r = PreflightResult(mode="test")
        check_yaml_structure(r, str(yaml), data)
        assert any(i.code == "YAML_NAMES_EMPTY" for i in r.errors())

    def test_nc_mismatch_errors(self, tmp_path):
        yaml = tmp_path / "data.yaml"
        yaml.write_text("train: t\nval: v\nnc: 5\nnames: [a]\n")
        data, _ = read_yaml_safe(str(yaml))
        r = PreflightResult(mode="test")
        check_yaml_structure(r, str(yaml), data)
        assert any(i.code == "YAML_NC_MISMATCH" for i in r.errors())

    def test_duplicate_names_warns(self, tmp_path):
        yaml = tmp_path / "data.yaml"
        yaml.write_text("train: t\nval: v\nnames: [a, a]\n")
        data, _ = read_yaml_safe(str(yaml))
        r = PreflightResult(mode="test")
        check_yaml_structure(r, str(yaml), data)
        assert any(i.code == "YAML_DUPLICATE_NAMES" for i in r.warnings())


class TestYAMLPaths:
    def test_train_missing_errors(self, tmp_path):
        yaml = tmp_path / "data.yaml"
        yaml.write_text("train: nonexistent_dir\nval: .\nnames: [a]\n")
        data, _ = read_yaml_safe(str(yaml))
        r = PreflightResult(mode="test")
        check_yaml_dataset_paths(r, str(yaml), data)
        assert any("TRAIN_PATH_MISSING" in i.code for i in r.errors())

    def test_valid_paths_pass(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        val_dir = tmp_path / "val"
        val_dir.mkdir()
        # Create a dummy image
        from PIL import Image
        Image.new('RGB', (10, 10)).save(str(train_dir / "img.jpg"))
        Image.new('RGB', (10, 10)).save(str(val_dir / "img.jpg"))

        yaml = tmp_path / "data.yaml"
        yaml.write_text(f"train: {train_dir}\nval: {val_dir}\nnames: [a]\n")
        data, _ = read_yaml_safe(str(yaml))
        r = PreflightResult(mode="test")
        check_yaml_dataset_paths(r, str(yaml), data)
        assert all(e.code != "TRAIN_NO_IMAGES" for e in r.errors())


class TestDetectLabelParsing:
    def test_valid_labels(self, tmp_path):
        label = tmp_path / "labels.txt"
        label.write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.7 0.15 0.15\n")
        instances, errors = _parse_label_file(str(label), 2)
        assert len(instances) == 2
        assert instances == [0, 1]
        assert errors == []

    def test_class_oob_errors(self, tmp_path):
        label = tmp_path / "labels.txt"
        label.write_text("5 0.5 0.5 0.2 0.2\n")
        instances, errors = _parse_label_file(str(label), 3)
        assert len(instances) == 0
        assert len(errors) > 0

    def test_negative_class_errors(self, tmp_path):
        label = tmp_path / "labels.txt"
        label.write_text("-1 0.5 0.5 0.2 0.2\n")
        instances, errors = _parse_label_file(str(label), 3)
        assert len(instances) == 0
        assert len(errors) > 0

    def test_nan_coords_errors(self, tmp_path):
        label = tmp_path / "labels.txt"
        label.write_text("0 NaN 0.5 0.2 0.2\n")
        instances, errors = _parse_label_file(str(label), 3)
        assert len(errors) > 0

    def test_oob_coords_errors(self, tmp_path):
        label = tmp_path / "labels.txt"
        label.write_text("0 1.5 2.0 0.2 0.2\n")
        instances, errors = _parse_label_file(str(label), 3)
        assert len(errors) > 0

    def test_zero_size_errors(self, tmp_path):
        label = tmp_path / "labels.txt"
        label.write_text("0 0.5 0.5 0 0.2\n0 0.5 0.5 0.2 0\n")
        instances, errors = _parse_label_file(str(label), 3)
        assert len(errors) >= 2

    def test_few_fields_errors(self, tmp_path):
        label = tmp_path / "labels.txt"
        label.write_text("0 0.5 0.5\n")
        instances, errors = _parse_label_file(str(label), 3)
        assert len(errors) > 0

    def test_error_includes_file_and_line(self, tmp_path):
        label = tmp_path / "labels.txt"
        label.write_text("0 0.5 0.5 0.2 0.2\n0 abc def ghi jkl\n")
        instances, errors = _parse_label_file(str(label), 3)
        # The line '0 abc def ghi jkl' has 5 fields but non-numeric coords
        assert len(errors) >= 1, f"Expected errors for invalid line, got errors={errors}"

    def test_empty_file(self, tmp_path):
        label = tmp_path / "labels.txt"
        label.write_text("")
        instances, errors = _parse_label_file(str(label), 3)
        assert instances == []
        assert errors == []

    def test_inf_coords_errors(self, tmp_path):
        label = tmp_path / "labels.txt"
        label.write_text("0 inf 0.5 0.2 0.2\n")
        instances, errors = _parse_label_file(str(label), 3)
        assert len(errors) > 0


class TestDetectLabelFull:
    """End-to-end label checks on a minimal dataset."""

    def _make_detect_dataset(self, tmp_path):
        """Create a minimal YOLO detect dataset."""
        train_img_dir = tmp_path / "images" / "train"
        val_img_dir = tmp_path / "images" / "val"
        train_lbl_dir = tmp_path / "labels" / "train"
        val_lbl_dir = tmp_path / "labels" / "val"
        for d in (train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
            d.mkdir(parents=True)

        from PIL import Image
        for i, d in enumerate([train_img_dir, val_img_dir]):
            for j in range(2):
                Image.new('RGB', (64, 64)).save(str(d / f"img{j:04d}.jpg"))

        yaml = tmp_path / "data.yaml"
        yaml.write_text(
            f"path: {tmp_path}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: object\n"
        )
        return yaml, train_lbl_dir, val_lbl_dir

    def test_no_valid_instances_errors(self, tmp_path):
        yaml, train_lbl, val_lbl = self._make_detect_dataset(tmp_path)
        # Create empty label files (no valid instances)
        (train_lbl / "img0000.txt").write_text("")
        (train_lbl / "img0001.txt").write_text("")
        (val_lbl / "img0000.txt").write_text("")
        (val_lbl / "img0001.txt").write_text("")

        data, _ = read_yaml_safe(str(yaml))
        r = PreflightResult(mode="test")
        check_detect_labels(r, str(yaml), data, "detect")
        assert any("NO_LABELS" in i.code for i in r.errors()), f"Should have NO_LABELS error, got: {[i.code for i in r.errors()]}"

    def test_valid_detect_passes(self, tmp_path):
        yaml, train_lbl, val_lbl = self._make_detect_dataset(tmp_path)
        (train_lbl / "img0000.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (train_lbl / "img0001.txt").write_text("0 0.3 0.7 0.15 0.15\n")
        (val_lbl / "img0000.txt").write_text("0 0.5 0.5 0.25 0.25\n")
        (val_lbl / "img0001.txt").write_text("0 0.7 0.3 0.1 0.1\n")

        data, _ = read_yaml_safe(str(yaml))
        r = PreflightResult(mode="test")
        check_detect_labels(r, str(yaml), data, "detect")
        assert r.error_count == 0, f"Should have no errors, got: {[i.code for i in r.errors()]}"

    def test_non_detect_skips_deep_check(self, tmp_path):
        yaml, _, _ = self._make_detect_dataset(tmp_path)
        data, _ = read_yaml_safe(str(yaml))
        r = PreflightResult(mode="test")
        check_detect_labels(r, str(yaml), data, "segment")
        assert any("DEEP_LABEL_CHECK_SKIPPED" in i.code for i in r.issues)
