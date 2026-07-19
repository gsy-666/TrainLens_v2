"""Convert between API payloads and the desktop label-file format."""

import json
import os.path as osp
from pathlib import Path
from typing import Any, Dict, List, Optional

from anylabeling.views.labeling.label_file import LabelFile
from anylabeling.views.labeling.schema import XLABEL_BASIC_FIELDS
from anylabeling.views.labeling.utils import rectangle_from_diagonal


def label_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(LabelFile.suffix)


def load_label_data(label_path: Path) -> Dict[str, Any]:
    """Read a desktop-format label JSON without touching the image bytes."""
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for shape in data.get("shapes", []):
        # Keep parity with LabelFile.load: upgrade the deprecated
        # two-point rectangle to the four-point form.
        if (
            shape.get("shape_type") == "rectangle"
            and len(shape.get("points", [])) == 2
        ):
            shape["points"] = rectangle_from_diagonal(shape["points"])

    data["imagePath"] = osp.basename(data.get("imagePath", ""))
    data.pop("imageData", None)
    return data


def extra_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fields outside XLABEL_BASIC_FIELDS, preserved across saves."""
    return {k: v for k, v in data.items() if k not in XLABEL_BASIC_FIELDS}


def save_label_data(
    image_path: Path,
    shapes: List[Dict[str, Any]],
    flags: Optional[Dict[str, Any]],
    other_data: Optional[Dict[str, Any]],
    image_height: int,
    image_width: int,
) -> Path:
    """Write a label file byte-compatible with the desktop app."""
    label_path = label_path_for(image_path)

    preserved: Dict[str, Any] = {}
    if label_path.exists():
        try:
            preserved = extra_fields(load_label_data(label_path))
        except Exception:
            preserved = {}
    # Never let caller-supplied extras collide with the template's basic
    # fields (LabelFile.save asserts on collision).
    safe_extra = {
        k: v for k, v in (other_data or {}).items() if k not in XLABEL_BASIC_FIELDS
    }
    merged = {**preserved, **safe_extra}

    label_file = LabelFile()
    label_file.save(
        filename=str(label_path),
        shapes=shapes,
        image_path=image_path.name,
        image_height=image_height,
        image_width=image_width,
        image_data=None,
        other_data=merged,
        flags=flags or {},
    )
    return label_path
