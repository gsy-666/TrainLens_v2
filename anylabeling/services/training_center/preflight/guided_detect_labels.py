"""TrainLens Preflight — YOLO Detect label validation.

Validates label files for the Ultralytics YOLO Detection task.
Each label line: class_id center_x center_y width height
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from . import PreflightIssue, PreflightResult, PreflightSeverity
from .guided_yaml import _resolve_dataset_path, _count_images_in_path


def check_detect_labels(
    result: PreflightResult,
    yaml_path: str,
    yaml_data: Optional[dict],
    task_type: str,
):
    """Validate YOLO Detect labels for train and val splits.

    For non-Detect tasks, emits a WARNING that deep label validation is skipped.
    """
    if yaml_data is None:
        return

    if task_type.lower() != "detect":
        result.add(PreflightIssue(
            code="DEEP_LABEL_CHECK_SKIPPED", severity=PreflightSeverity.WARNING,
            title="Deep label validation not available for this task type",
            message=f"Deep label validation for task '{task_type}' is not available in the current version. "
                    f"Only basic dataset checks were performed.",
            suggestion="Manually verify your label format is correct for this task.",
        ))
        return

    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    nc = _get_class_count(yaml_data)

    for split in ("train", "val"):
        resolved = _resolve_dataset_path(yaml_dir, yaml_data, split)
        if resolved is None or not os.path.exists(resolved):
            continue

        images = _get_images_in_path(resolved)
        if not images:
            continue

        _validate_split_labels(result, split, yaml_dir, resolved, images, nc)


def _get_images_in_path(path: str) -> List[str]:
    """Get sorted image file paths from a dataset path."""
    from . import SUPPORTED_IMAGE_EXTS

    images = []
    try:
        for root, dirs, files in os.walk(path):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTS:
                    images.append(os.path.join(root, f))
    except OSError:
        pass
    return images


def _get_class_count(yaml_data: dict) -> int:
    """Get number of classes from YAML data."""
    names = yaml_data.get("names", {})
    if isinstance(names, dict):
        return len(names)
    elif isinstance(names, list):
        return len(names)
    return 0


def _find_label_path(image_path: str, images_root: str) -> Optional[str]:
    """Find corresponding label file for an image.

    Uses the Ultralytics convention: .../images/train/a.jpg → .../labels/train/a.txt
    """
    image_path_norm = os.path.normpath(image_path)
    images_root_norm = os.path.normpath(images_root)

    # Try standard Ultralytics layout: images → labels
    parts = image_path_norm.split(os.sep)
    replaced = False
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            replaced = True
            break

    if replaced:
        label_path = os.sep.join(parts)
        label_path = os.path.splitext(label_path)[0] + ".txt"
        if os.path.isfile(label_path):
            return label_path

    # Fallback: same directory as image, same name .txt
    label_path = os.path.splitext(image_path)[0] + ".txt"
    if os.path.isfile(label_path):
        return label_path

    return None


def _validate_split_labels(
    result: PreflightResult,
    split: str,
    yaml_dir: str,
    dataset_root: str,
    images: List[str],
    nc: int,
):
    """Validate all label files for a dataset split."""
    total_images = len(images)
    labeled_images = 0
    unlabeled_images = 0
    empty_label_files = 0
    total_valid_instances = 0
    class_instance_counts: Dict[int, int] = {}
    corrupted_files: List[Tuple[str, str]] = []

    for img_path in images:
        label_path = _find_label_path(img_path, dataset_root)

        if label_path is None:
            unlabeled_images += 1
            continue

        instances, errors = _parse_label_file(label_path, nc)
        if errors:
            corrupted_files.extend([(label_path, e) for e in errors])
            if not instances:
                continue

        if instances:
            labeled_images += 1
            total_valid_instances += len(instances)
            for cls_id in instances:
                class_instance_counts[cls_id] = class_instance_counts.get(cls_id, 0) + 1
        else:
            empty_label_files += 1

    # ── Report results ──

    # Effective instances
    if total_valid_instances == 0:
        result.add(PreflightIssue(
            code=f"{split.upper()}_NO_LABELS",
            severity=PreflightSeverity.ERROR,
            title=f"{split.capitalize()} set contains no valid labeled instances",
            message=f"No valid labeled instances found in {split} set "
                    f"({total_images} images, {empty_label_files} empty label files, "
                    f"{labeled_images} labeled).",
        ))
    else:
        result.add(PreflightIssue(
            code=f"{split.upper()}_LABELS_OK",
            severity=PreflightSeverity.PASS,
            title=f"{split.capitalize()}: {total_valid_instances} valid instances",
            message=f"{split}: {total_images} images, {labeled_images} labeled, "
                    f"{total_valid_instances} instances across {len(class_instance_counts)} classes.",
        ))

    # Unlabeled images
    if unlabeled_images > 0:
        empty_ratio = (unlabeled_images + empty_label_files) / total_images if total_images > 0 else 0
        sev = PreflightSeverity.WARNING if empty_ratio > 0.5 else PreflightSeverity.WARNING
        result.add(PreflightIssue(
            code=f"{split.upper()}_UNLABELED",
            severity=sev,
            title=f"{unlabeled_images} images in {split} have no labels",
            message=f"Label files missing for {unlabeled_images}/{total_images} images. "
                    f"{empty_label_files} label files are empty.",
            suggestion="Background images without labels are allowed, but verify this is intentional.",
        ))

    # Corrupted label files
    if corrupted_files:
        for path, error in corrupted_files[:10]:
            result.add(PreflightIssue(
                code=f"{split.upper()}_CORRUPTED_LABEL",
                severity=PreflightSeverity.ERROR,
                title="Corrupted label file",
                message=error,
                path=path,
            ))
        if len(corrupted_files) > 10:
            result.add(PreflightIssue(
                code=f"{split.upper()}_CORRUPTED_LABEL",
                severity=PreflightSeverity.ERROR,
                title=f"{len(corrupted_files)} corrupted label files",
                message=f"{len(corrupted_files)} label files have errors (showing first 10).",
            ))

    # Per-class stats
    if nc > 0 and class_instance_counts:
        missing_classes = [i for i in range(nc) if i not in class_instance_counts]
        for cls_id in missing_classes:
            result.add(PreflightIssue(
                code="CLASS_ZERO_INSTANCES",
                severity=PreflightSeverity.WARNING,
                title=f"Class {cls_id} has 0 instances",
                message=f"No instances found for class index {cls_id} in {split} set.",
            ))

        few_classes = [i for i, c in class_instance_counts.items() if c < 5]
        for cls_id in few_classes:
            result.add(PreflightIssue(
                code="CLASS_FEW_INSTANCES",
                severity=PreflightSeverity.WARNING,
                title=f"Class {cls_id} has only {class_instance_counts[cls_id]} instances",
                message=f"Class index {cls_id} has very few instances ({class_instance_counts[cls_id]}) "
                        f"in {split} set.",
                suggestion="Add more annotated examples for this class.",
            ))

        # Class imbalance
        non_zero = {k: v for k, v in class_instance_counts.items() if v > 0}
        if len(non_zero) >= 2:
            max_count = max(non_zero.values())
            min_count = min(non_zero.values())
            if max_count / min_count > 20:
                from . import CLASS_IMBALANCE_WARNING_RATIO
                result.add(PreflightIssue(
                    code="CLASS_IMBALANCE",
                    severity=PreflightSeverity.WARNING,
                    title="Significant class imbalance detected",
                    message=f"Largest class has {max_count} instances, smallest has {min_count}. "
                            f"Ratio: {max_count / min_count:.1f}:1.",
                    suggestion="Consider collecting more data for underrepresented classes.",
                ))


def _parse_label_file(
    label_path: str, nc: int
) -> Tuple[List[int], List[str]]:
    """Parse a YOLO label file. Returns (valid_class_ids, error_messages)."""
    instances = []
    errors = []

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError) as e:
        return [], [f"Cannot read {label_path}: {e}"]

    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            errors.append(f"{label_path}:{line_no}: expected >=5 fields, got {len(parts)}")
            continue

        # Parse class_id
        try:
            class_id = int(float(parts[0]))
        except (ValueError, TypeError):
            errors.append(f"{label_path}:{line_no}: invalid class_id '{parts[0]}'")
            continue

        if class_id < 0:
            errors.append(f"{label_path}:{line_no}: class_id {class_id} is negative")
            continue

        if nc > 0 and class_id >= nc:
            errors.append(f"{label_path}:{line_no}: class_id {class_id} >= {nc} classes")
            continue

        # Parse coordinates
        try:
            x, y, w, h = [float(p) for p in parts[1:5]]
        except (ValueError, TypeError):
            errors.append(f"{label_path}:{line_no}: invalid coordinate values")
            continue

        # Check for NaN/Inf
        import math
        for name, val in [("x", x), ("y", y), ("w", w), ("h", h)]:
            if math.isnan(val) or math.isinf(val):
                errors.append(f"{label_path}:{line_no}: {name}={val} (NaN or Inf)")
                break
        else:
            # Check bounds
            if not (-0.01 <= x <= 1.01 and -0.01 <= y <= 1.01):
                errors.append(f"{label_path}:{line_no}: center ({x},{y}) out of [0,1] range")
            if not (0 < w <= 1.01 and 0 < h <= 1.01):
                errors.append(f"{label_path}:{line_no}: size ({w},{h}) invalid (must be >0 and ≤1)")
            else:
                instances.append(class_id)

    return instances, errors
