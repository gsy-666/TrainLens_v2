"""TrainLens Preflight — YAML parsing and dataset path validation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from . import PreflightIssue, PreflightResult, PreflightSeverity


# ── YAML reading ────────────────────────────────────────────────────

def read_yaml_safe(yaml_path: str) -> Tuple[Optional[dict], Optional[str]]:
    """Safely read a YAML file. Returns (data_dict, error_message)."""
    try:
        import yaml
    except ImportError:
        return None, "PyYAML is not installed. Install with: pip install pyyaml"

    path = Path(yaml_path)
    if not path.exists():
        return None, f"File not found: {yaml_path}"

    try:
        # Try UTF-8-SIG first (handles BOM), then UTF-8
        content = None
        for encoding in ("utf-8-sig", "utf-8"):
            try:
                content = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if content is None:
            return None, f"Cannot decode file as UTF-8: {yaml_path}"

        data = yaml.safe_load(content)
        if data is None:
            return {}, None
        if not isinstance(data, dict):
            return None, f"YAML root must be a mapping, got {type(data).__name__}"
        return data, None
    except yaml.YAMLError as e:
        return None, f"YAML syntax error: {e}"


# ── YAML content checks ─────────────────────────────────────────────

def check_yaml_structure(result: PreflightResult, yaml_path: str, yaml_data: Optional[dict]):
    """Check YAML has required fields (train, val, names)."""
    if yaml_data is None:
        return  # Error already reported

    # Check train
    if "train" not in yaml_data:
        result.add(PreflightIssue(
            code="YAML_NO_TRAIN", severity=PreflightSeverity.ERROR,
            title="YAML missing 'train' field",
            message=f"The dataset YAML does not specify a 'train' path.",
            path=yaml_path,
        ))

    # Check val
    if "val" not in yaml_data:
        result.add(PreflightIssue(
            code="YAML_NO_VAL", severity=PreflightSeverity.ERROR,
            title="YAML missing 'val' field",
            message=f"The dataset YAML does not specify a 'val' path.",
            path=yaml_path,
        ))

    # Check names
    if "names" not in yaml_data:
        result.add(PreflightIssue(
            code="YAML_NO_NAMES", severity=PreflightSeverity.ERROR,
            title="YAML missing 'names' field",
            message="The dataset YAML does not specify class names.",
            path=yaml_path,
        ))
    else:
        names = yaml_data["names"]
        if isinstance(names, dict):
            nc = len(names)
            name_list = sorted(names.values())
        elif isinstance(names, list):
            nc = len(names)
            name_list = names
        else:
            result.add(PreflightIssue(
                code="YAML_NAMES_INVALID", severity=PreflightSeverity.ERROR,
                title="YAML 'names' has invalid type",
                message=f"'names' must be a list or dict, got {type(names).__name__}",
                path=yaml_path,
            ))
            name_list = []
            nc = 0

        if nc == 0:
            result.add(PreflightIssue(
                code="YAML_NAMES_EMPTY", severity=PreflightSeverity.ERROR,
                title="No class names defined",
                message="The 'names' field is empty. At least one class is required.",
                path=yaml_path,
            ))

        # Check for duplicate class names
        if name_list:
            seen = set()
            dupes = set()
            for n in name_list:
                n_str = str(n)
                if n_str in seen:
                    dupes.add(n_str)
                seen.add(n_str)
            if dupes:
                result.add(PreflightIssue(
                    code="YAML_DUPLICATE_NAMES", severity=PreflightSeverity.WARNING,
                    title="Duplicate class names",
                    message=f"Duplicate class names found: {', '.join(sorted(dupes))}",
                    path=yaml_path,
                ))

    # Check nc consistency
    nc_field = yaml_data.get("nc")
    names = yaml_data.get("names", {})
    if isinstance(names, dict):
        actual_nc = len(names)
    elif isinstance(names, list):
        actual_nc = len(names)
    else:
        actual_nc = 0

    if nc_field is not None and actual_nc > 0 and int(nc_field) != actual_nc:
        result.add(PreflightIssue(
            code="YAML_NC_MISMATCH", severity=PreflightSeverity.ERROR,
            title="nc does not match names count",
            message=f"YAML 'nc'={nc_field} but 'names' has {actual_nc} entries.",
            path=yaml_path,
            suggestion="Update 'nc' to match the number of class names.",
        ))


# ── Path resolution ─────────────────────────────────────────────────

def _resolve_dataset_path(yaml_dir: str, yaml_data: dict, key: str) -> Optional[str]:
    """Resolve a dataset path relative to YAML directory (with 'path' prefix support)."""
    raw = yaml_data.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None

    base = yaml_data.get("path", "")
    if base:
        candidate = os.path.join(base, raw)
    else:
        candidate = raw

    # Resolve relative to YAML dir
    if not os.path.isabs(candidate):
        candidate = os.path.normpath(os.path.join(yaml_dir, candidate))
    return candidate


def _count_images_in_path(path: str) -> Tuple[int, List[str]]:
    """Count images in a dataset path. Supports dirs, txt lists, single images."""
    from . import SUPPORTED_IMAGE_EXTS

    images = []
    if not os.path.exists(path):
        return 0, []

    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            # Image list file
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and os.path.isfile(line):
                            images.append(line)
            except (OSError, UnicodeDecodeError):
                return 0, []
        elif ext in SUPPORTED_IMAGE_EXTS:
            images.append(path)
        return len(images), images

    # Directory: find all images
    try:
        for root, dirs, files in os.walk(path):
            for f in files:
                if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTS:
                    images.append(os.path.join(root, f))
    except OSError:
        return 0, []
    return len(images), images


def check_yaml_dataset_paths(result: PreflightResult, yaml_path: str, yaml_data: Optional[dict]):
    """Check train/val paths exist and validate image counts."""
    if yaml_data is None:
        return

    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))

    for split in ("train", "val"):
        resolved = _resolve_dataset_path(yaml_dir, yaml_data, split)
        if resolved is None:
            continue  # Error already reported

        if not os.path.exists(resolved):
            result.add(PreflightIssue(
                code=f"{split.upper()}_PATH_MISSING",
                severity=PreflightSeverity.ERROR,
                title=f"Dataset {split} path does not exist",
                message=f"The {split} path does not exist: {resolved}",
                path=resolved,
            ))
            continue

        count, images = _count_images_in_path(resolved)
        if count == 0:
            result.add(PreflightIssue(
                code=f"{split.upper()}_NO_IMAGES",
                severity=PreflightSeverity.ERROR,
                title=f"No images found in {split} set",
                message=f"The {split} path contains no supported images: {resolved}",
                path=resolved,
            ))
        else:
            result.add(PreflightIssue(
                code=f"{split.upper()}_COUNT",
                severity=PreflightSeverity.PASS,
                title=f"{split.capitalize()}: {count} images",
                message=f"{split} set: {count} supported images",
                path=resolved,
            ))

    # Check train/val overlap
    train_path = _resolve_dataset_path(yaml_dir, yaml_data, "train")
    val_path = _resolve_dataset_path(yaml_dir, yaml_data, "val")
    if train_path and val_path:
        if os.path.normpath(train_path) == os.path.normpath(val_path):
            result.add(PreflightIssue(
                code="TRAIN_VAL_SAME_PATH",
                severity=PreflightSeverity.ERROR,
                title="Train and validation paths are identical",
                message=f"train and val both point to: {train_path}",
                path=train_path,
                suggestion="Use separate directories for training and validation data.",
            ))
        else:
            # Check for overlapping images
            _, train_imgs = _count_images_in_path(train_path)
            _, val_imgs = _count_images_in_path(val_path)
            train_set = set(train_imgs)
            val_set = set(val_imgs)
            overlap = train_set & val_set
            if overlap:
                result.add(PreflightIssue(
                    code="TRAIN_VAL_OVERLAP", severity=PreflightSeverity.WARNING,
                    title="Train and validation sets have overlapping images",
                    message=f"{len(overlap)} images appear in both train and val sets.",
                    details=f"Overlapping images: {', '.join(sorted(overlap)[:5])}...",
                    path=train_path,
                    suggestion="Ensure train and validation sets are disjoint.",
                ))
