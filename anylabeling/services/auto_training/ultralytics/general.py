import glob
import json
import os
import re
import shutil
import random
from datetime import datetime
from typing import List

from ._io import load_yaml_config, save_yaml_config
from .config import (
    get_dataset_path,
    TASK_LABEL_MAPPINGS,
    TASK_SHAPE_MAPPINGS,
)


def create_yolo_dataset(
    image_list: List[str],
    task_type: str,
    dataset_ratio: float,
    data_file: str,
    output_dir: str = None,
    pose_cfg_file: str = None,
    skip_empty_files: bool = False,
    only_checked_files: bool = False,
) -> str:
    """Create YOLO dataset from image list and annotations.

    Args:
        image_list: List of image paths
        task_type: Type of detection task
        dataset_ratio: Ratio to split train/val data
        data_file: Path to data config file
        output_dir: Optional output directory for labels
        pose_cfg_file: Optional pose config file for pose detection
        skip_empty_files: Whether to skip empty label files
        only_checked_files: Whether to use only checked files

    Returns:
        Path to created dataset directory
    """
    from anylabeling.views.labeling.label_converter import LabelConverter

    def _process_images_batch(
        image_label_pairs, images_dir, labels_dir, converter, mode, skip_empty
    ):
        for image_file, label_file in image_label_pairs:
            filename = os.path.basename(image_file)
            dst_image_path = os.path.join(images_dir, filename)

            if os.name == "nt":  # Windows
                shutil.copy2(image_file, dst_image_path)
            else:
                os.symlink(image_file, dst_image_path)

            if label_file and os.path.exists(label_file):
                dst_label_path = os.path.join(
                    labels_dir, os.path.splitext(filename)[0] + ".txt"
                )
                converter.custom_to_yolo(
                    label_file,
                    dst_label_path,
                    mode,
                    skip_empty_files=skip_empty,
                )

    def _process_classify_images_batch(image_label_pairs, base_dir):
        for image_file, label_file in image_label_pairs:
            filename = os.path.basename(image_file)

            if not label_file or not os.path.exists(label_file):
                continue

            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                flags = data.get("flags", {})

                for flag_name, flag_value in flags.items():
                    if flag_value:
                        class_dir = os.path.join(base_dir, flag_name)
                        os.makedirs(class_dir, exist_ok=True)
                        dst_image_path = os.path.join(class_dir, filename)

                        if os.name == "nt":  # Windows
                            shutil.copy2(image_file, dst_image_path)
                        else:
                            os.symlink(image_file, dst_image_path)
                        break
            except (json.JSONDecodeError, IOError):
                continue

    if task_type == "Classify":
        data = {"names": {}, "nc": 0}
        converter = None
        data_file_name = "classification"
    else:
        data = load_yaml_config(data_file)
        if task_type.lower() == "pose":
            if not pose_cfg_file:
                return (
                    None,
                    "Pose configuration file is required for pose detection tasks",
                )
            converter = LabelConverter(pose_cfg_file=pose_cfg_file)
        else:
            converter = LabelConverter()
        # Extract class names: prefer YAML, fall back to JSON annotations
        classes_from_yaml = None
        if data and isinstance(data, dict) and "names" in data:
            names_dict = data["names"]
            if isinstance(names_dict, dict):
                classes_from_yaml = [names_dict[i] for i in sorted(names_dict.keys())]
            elif isinstance(names_dict, list):
                classes_from_yaml = list(names_dict)
        # Auto-extract from JSON annotations first
        all_labels = set()
        skipped_labels = []
        for image_file in image_list:
            label_json = os.path.join(
                os.path.dirname(image_file) if not output_dir else output_dir,
                os.path.splitext(os.path.basename(image_file))[0] + ".json",
            )
            if os.path.exists(label_json):
                try:
                    with open(label_json, "r", encoding="utf-8") as f:
                        jd = json.load(f)
                    for s in jd.get("shapes", []):
                        lbl = s.get("label")
                        if lbl is None:
                            continue
                        stripped = str(lbl).strip()
                        if not stripped:
                            skipped_labels.append(
                                f"{label_json}: empty/whitespace label in shape"
                            )
                            continue
                        all_labels.add(stripped)
                except Exception:
                    pass
        if all_labels:
            converter.classes = sorted(all_labels)  # prefer auto-extracted from source JSON
        elif classes_from_yaml:
            converter.classes = classes_from_yaml  # fallback to YAML when no JSON annotations
        else:
            converter.classes = []
        data_file_name = os.path.splitext(os.path.basename(data_file))[0] if data_file else "auto_dataset"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = os.path.join(
        get_dataset_path(), task_type.lower(), f"{data_file_name}_{timestamp}"
    )

    if task_type == "Classify":
        train_dir = os.path.join(temp_dir, "train")
        val_dir = os.path.join(temp_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
    else:
        train_images_dir = os.path.join(temp_dir, "images", "train")
        val_images_dir = os.path.join(temp_dir, "images", "val")
        train_labels_dir = os.path.join(temp_dir, "labels", "train")
        val_labels_dir = os.path.join(temp_dir, "labels", "val")
        for dir_path in [
            train_images_dir,
            val_images_dir,
            train_labels_dir,
            val_labels_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

    background_images = []
    valid_images = []
    valid_shapes = TASK_SHAPE_MAPPINGS.get(task_type, [])

    for image_file in image_list:
        label_dir, filename = os.path.split(image_file)
        if output_dir:
            label_dir = output_dir
        label_file = os.path.join(
            label_dir, os.path.splitext(filename)[0] + ".json"
        )

        if not os.path.exists(label_file):
            if only_checked_files:
                continue
            background_images.append(image_file)
            continue

        try:
            with open(label_file, "r", encoding="utf-8") as f:
                label_info = json.load(f)

            if (
                only_checked_files
                and label_info.get("checked", False) is not True
            ):
                continue

            if task_type == "Classify":
                flags = label_info.get("flags", {})
                has_valid_flag = any(
                    flag_value for flag_value in flags.values()
                )
                if has_valid_flag:
                    valid_images.append((image_file, label_file))
                else:
                    background_images.append(image_file)
            else:
                shapes = label_info.get("shapes", [])
                has_valid_shape = any(
                    shape.get("shape_type") in valid_shapes
                    for shape in shapes
                    if "shape_type" in shape
                )
                if has_valid_shape:
                    valid_images.append((image_file, label_file))
                else:
                    background_images.append(image_file)
        except Exception:
            if only_checked_files:
                continue
            background_images.append(image_file)
            continue

    # ensure train/val split is randomized
    valid_images = random.sample(valid_images, k=len(valid_images))

    # ── Class-aware stratified split ──────────────────────────────────
    # Build class→images index for single-instance / rare-class coverage
    class_to_images: dict[int, list] = {}
    for idx, (_, label_file) in enumerate(valid_images):
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                jd = json.load(f)
            for s in jd.get("shapes", []):
                lbl = s.get("label")
                if lbl and str(lbl).strip():
                    # Map label to converter class ID (already contiguous)
                    try:
                        cid = converter.classes.index(str(lbl).strip())
                    except ValueError:
                        continue
                    class_to_images.setdefault(cid, []).append(idx)
        except Exception:
            pass

    # Remove duplicates (same image may appear under multiple classes)
    for cid in list(class_to_images.keys()):
        class_to_images[cid] = list(set(class_to_images[cid]))

    total_valid = len(valid_images)
    target_train = round(total_valid * dataset_ratio)
    target_val = total_valid - target_train

    # Safety: when ratio > 0.5, train must be > val
    if dataset_ratio > 0.5 and target_train <= target_val:
        target_train = target_val + 1
        target_val = total_valid - target_train

    # Phase 1: guarantee single-instance classes go to train
    train_indices: set = set()
    for cid, img_indices in class_to_images.items():
        if len(img_indices) == 1:
            train_indices.add(img_indices[0])

    # Phase 2: ensure every class has at least one image in train
    for cid, img_indices in class_to_images.items():
        if not any(i in train_indices for i in img_indices):
            for i in img_indices:
                if i not in train_indices:
                    train_indices.add(i)
                    break

    # Phase 3: fill remaining slots from unassigned images
    remaining = [i for i in range(total_valid) if i not in train_indices]
    needed = target_train - len(train_indices)
    if needed > 0:
        extra = remaining[:needed]
        train_indices.update(extra)
        remaining = [i for i in remaining if i not in train_indices]

    train_valid_images = [valid_images[i] for i in sorted(train_indices)]
    val_valid_images = [valid_images[i] for i in sorted(remaining)]

    # Log split stats
    import logging
    _logger = logging.getLogger(__name__)
    _logger.info(
        f"Dataset split: train={len(train_valid_images)}, val={len(val_valid_images)}, "
        f"total={total_valid}, target_train={target_train}, dataset_ratio={dataset_ratio}"
    )

    if task_type == "Classify":
        _process_classify_images_batch(train_valid_images, train_dir)
        _process_classify_images_batch(val_valid_images, val_dir)
    else:
        if skip_empty_files:
            all_train_images = train_valid_images
        else:
            all_train_images = [
                (img, None) for img in background_images
            ] + train_valid_images

        mode = TASK_LABEL_MAPPINGS.get(task_type, "hbb")
        _process_images_batch(
            all_train_images,
            train_images_dir,
            train_labels_dir,
            converter,
            mode,
            skip_empty_files,
        )
        _process_images_batch(
            val_valid_images,
            val_images_dir,
            val_labels_dir,
            converter,
            mode,
            skip_empty_files,
        )

    info_file = os.path.join(temp_dir, "dataset_info.txt")
    with open(info_file, "w", encoding="utf-8") as f:
        f.write(
            f"Dataset created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Task type: {task_type}\n")
        f.write(f"Total images: {len(image_list)}\n")
        if task_type == "Classify":
            f.write(f"Train images: {len(train_valid_images)}\n")
            f.write(f"Val images: {len(val_valid_images)}\n")
        else:
            f.write(f"Train images: {len(all_train_images)}\n")
            f.write(f"Val images: {len(val_valid_images)}\n")
            f.write(f"Background images: {len(background_images)}\n")
            f.write(f"Skip empty files: {skip_empty_files}\n")
            f.write(f"Only checked files: {only_checked_files}\n")
        f.write(f"Valid labeled images: {len(valid_images)}\n")
        f.write(f"Dataset ratio: {dataset_ratio}\n")

    yaml_file = os.path.join(temp_dir, "data.yaml")

    if task_type == "Classify":
        class_names = {}
        train_dir = os.path.join(temp_dir, "train")
        if os.path.exists(train_dir):
            class_dirs = [
                d
                for d in os.listdir(train_dir)
                if os.path.isdir(os.path.join(train_dir, d))
            ]
            for i, class_name in enumerate(sorted(class_dirs)):
                class_names[i] = class_name

        data = {
            "path": temp_dir,
            "train": "train",
            "val": "val",
            "names": class_names,
            "nc": len(class_names),
        }
    else:
        # Build fresh YAML data from detected classes
        if data is None or not isinstance(data, dict):
            data = {}
        # Remove stale fields from inherited YAML (download, test, coco classes)
        data.pop("download", None)
        data.pop("test", None)
        data.pop("nc", None)
        data["path"] = temp_dir
        data["train"] = "images/train"
        data["val"] = "images/val"
        # Build names from converter.classes (never inherit old names)
        names_dict = {}
        for i, cls_name in enumerate(converter.classes):
            names_dict[i] = cls_name
        data["names"] = names_dict
        data["nc"] = len(names_dict)

    save_yaml_config(data, yaml_file)

    # ── Post-export integrity check ─────────────────────────────────
    if task_type != "Classify":
        nc = data.get("nc", 0)
        names = data.get("names", {})
        errors = []
        for split_dir, split_name in [(train_labels_dir, "train"), (val_labels_dir, "val")]:
            for txt_file in glob.glob(os.path.join(split_dir, "*.txt")):
                if not os.path.getsize(txt_file):
                    continue
                with open(txt_file, "r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if not parts:
                            continue
                        try:
                            cls_id = int(float(parts[0]))
                        except (ValueError, TypeError):
                            errors.append(f"{txt_file}:{line_no}: invalid class_id")
                            continue
                        if cls_id < 0 or cls_id >= nc:
                            errors.append(
                                f"{txt_file}:{line_no}: class_id {cls_id} out of range [0, {nc - 1}]"
                            )
        if errors:
            raise RuntimeError(
                f"Dataset integrity check failed ({len(errors)} errors):\n"
                + "\n".join(errors[:20])
                + ("\n..." if len(errors) > 20 else "")
            )

    return temp_dir


def format_classes_display(classes_value) -> str:
    """Formats class values for display.

    This function takes a classes value and formats it into a string representation.
    It handles None values, empty values, lists, and single values.

    Args:
        classes_value: The value to format. Can be None, a list, or a single value.

    Returns:
        A string representation of the classes value:
        - Empty string if input is None or empty
        - Comma-separated string if input is a list
        - String conversion of the input value otherwise
    """
    if classes_value is None or not classes_value:
        return ""
    if isinstance(classes_value, list):
        return ",".join(map(str, classes_value))
    return str(classes_value) if classes_value else ""


def parse_string_to_digit_list(input_string: str) -> List[int]:
    """Parses a string containing numbers into a list of integers.

    This function uses regular expressions to find all numerical digits
    in the input string, treating any non-digit characters as delimiters.
    It then converts the found sequences of digits into integers.

    Args:
        input_string: The string to parse. It can contain numbers
            separated by commas, spaces, or any other non-digit symbols.
            Example: "1, 2 3-4".

    Returns:
        A list of integers found in the string. For example, for the input
        "1, 2 3-4", the output would be [1, 2, 3, 4]. Returns None if
        no numbers are found, input is empty, or parsing fails.
    """
    try:
        if not input_string:
            return None

        numbers_as_strings = re.findall(r"\d+", input_string)
        if not numbers_as_strings:
            return None

        return [int(num) for num in numbers_as_strings]

    except Exception:
        return None
