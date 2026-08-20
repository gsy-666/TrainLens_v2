"""Close the annotation -> training loop: build a YOLO training dataset
from the currently opened annotation directory."""

import asyncio
import json
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from anylabeling.services.auto_training.ultralytics.config import (
    TASK_SHAPE_MAPPINGS,
)
from anylabeling.services.auto_training.ultralytics.general import (
    create_yolo_dataset,
)

from ..deps import session

router = APIRouter()

TASK_TYPES = ["Detect", "OBB", "Segment", "Pose", "Classify"]

# Shape types counted for the object-level label statistics ("flags" are
# tracked separately as image-level classification labels).
_SHAPE_TYPES = ["rectangle", "rotation", "polygon", "point"]


class PrepareRequest(BaseModel):
    task_type: str = "Detect"
    dataset_ratio: float = 0.9  # train fraction
    skip_empty_files: bool = False
    only_checked_files: bool = False


def _read_info(dataset_dir: str) -> str:
    info = Path(dataset_dir) / "dataset_info.txt"
    try:
        return info.read_text(encoding="utf-8")
    except OSError:
        return ""


def _ensure_val_split(dataset_dir: str) -> str:
    """Ultralytics refuses an empty val split. On tiny datasets the
    stratified split can leave val empty; mirror train into val then."""
    ds = Path(dataset_dir)
    val_images = ds / "images" / "val"
    train_images = ds / "images" / "train"
    if not train_images.is_dir() or any(val_images.iterdir() if val_images.is_dir() else []):
        return ""
    if not any(train_images.iterdir()):
        return ""
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels = ds / "labels" / "val"
    val_labels.mkdir(parents=True, exist_ok=True)
    for img in train_images.iterdir():
        if img.is_file():
            shutil.copy2(img, val_images / img.name)
            lbl = (ds / "labels" / "train" / (img.stem + ".txt"))
            if lbl.exists():
                shutil.copy2(lbl, val_labels / lbl.name)
    return "Val split was empty; mirrored the train split into val.\n"


def _compute_stats(image_list: list[str]) -> dict:
    """Aggregate label statistics for the dataset-inspection panel.

    Single pass over the labelme JSON files (one read per image instead of
    one per check): per-task validity, per-label shape counts,
    classification flags and class totals all come from the same parse.
    Malformed files are skipped and surfaced as a warning instead of
    failing the whole request with a 500.
    """
    per_task_valid = {task: 0 for task in TASK_TYPES}
    shape_infos: dict[str, dict[str, int]] = {}
    flag_infos: dict[str, dict[str, int]] = {}
    malformed = 0

    for image_file in image_list:
        label_file = Path(image_file).with_suffix(".json")
        if not label_file.exists():
            continue
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            shapes = data.get("shapes", [])
            flags = data.get("flags", {})

            # per-task validity (mirrors get_task_valid_images)
            for task in TASK_TYPES:
                valid_shapes = TASK_SHAPE_MAPPINGS[task]
                if "flags" in valid_shapes:
                    if any(flags.values()):
                        per_task_valid[task] += 1
                elif any(
                    shape.get("shape_type") in valid_shapes
                    for shape in shapes
                    if "shape_type" in shape
                ):
                    per_task_valid[task] += 1

            # per-label shape counts (mirrors get_label_infos)
            for shape in shapes:
                if "label" not in shape or "shape_type" not in shape:
                    continue
                shape_type = shape["shape_type"]
                if shape_type not in _SHAPE_TYPES:
                    continue
                label = shape["label"]
                if label not in shape_infos:
                    shape_infos[label] = {st: 0 for st in _SHAPE_TYPES}
                shape_infos[label][shape_type] += 1

            # classification flags (mirrors get_label_infos(..., ["flags"]))
            selected_flag = None
            for flag_name, flag_value in flags.items():
                if flag_value:
                    selected_flag = flag_name
                    break
            if selected_flag:
                if selected_flag not in flag_infos:
                    flag_infos[selected_flag] = {"flags": 0, "_total": 0}
                flag_infos[selected_flag]["_total"] += 1
        except (json.JSONDecodeError, IOError, OSError, TypeError, AttributeError):
            malformed += 1
            continue

    shape_infos = {k: shape_infos[k] for k in sorted(shape_infos)}
    flag_infos = {k: flag_infos[k] for k in sorted(flag_infos)}
    label_infos = {**shape_infos, **flag_infos}

    class_counts: dict[str, int] = {}
    for label, infos in shape_infos.items():
        class_counts[label] = class_counts.get(label, 0) + sum(infos.values())
    for label, infos in flag_infos.items():
        class_counts[label] = class_counts.get(label, 0) + infos.get("_total", 0)

    recommended = "Detect"
    best_valid = 0
    for task in TASK_TYPES:
        if per_task_valid[task] > best_valid:
            best_valid = per_task_valid[task]
            recommended = task

    warnings = []
    if malformed:
        warnings.append(
            {
                "code": "malformed_labels",
                "message": f"{malformed} 个标注文件解析失败，已从统计中跳过",
            }
        )
    if len(image_list) > 0 and best_valid == 0:
        warnings.append(
            {
                "code": "no_valid_labels",
                "message": "没有找到任何有效标注，请先完成标注再训练",
            }
        )
    if per_task_valid[recommended] < 20:
        warnings.append(
            {
                "code": "few_images",
                "message": (
                    f"按推荐任务 {recommended} 统计的有效图片仅 "
                    f"{per_task_valid[recommended]} 张，少于 20 张，"
                    "训练效果可能较差"
                ),
            }
        )
    if class_counts:
        few = [label for label, n in class_counts.items() if n < 5]
        if few:
            warnings.append(
                {
                    "code": "few_instances",
                    "message": (
                        f"类别 {', '.join(few)} 的实例数少于 5，"
                        "建议补充样本"
                    ),
                }
            )
        counts = list(class_counts.values())
        lo, hi = min(counts), max(counts)
        if lo > 0 and hi / lo > 20:
            warnings.append(
                {
                    "code": "class_imbalance",
                    "message": (
                        f"类别不均衡：最多 {hi} 个实例，最少 {lo} 个，"
                        "比例超过 20:1"
                    ),
                }
            )

    return {
        "total_images": len(image_list),
        "per_task_valid": per_task_valid,
        "label_infos": label_infos,
        "class_counts": class_counts,
        "warnings": warnings,
        "recommended_task": recommended,
    }


@router.get("/dataset/stats")
async def dataset_stats():
    image_dir = session.get_dir()
    images = session.get_images()
    if image_dir is None or not images:
        raise HTTPException(status_code=400, detail="No image directory opened")

    image_list = [str(image_dir / name) for name in images]
    return await asyncio.to_thread(_compute_stats, image_list)


@router.post("/dataset/prepare")
async def prepare_dataset(req: PrepareRequest):
    if req.task_type not in TASK_TYPES:
        raise HTTPException(
            status_code=400, detail=f"task_type must be one of {TASK_TYPES}"
        )
    if not 0.0 < req.dataset_ratio < 1.0:
        raise HTTPException(status_code=400, detail="dataset_ratio must be in (0, 1)")

    image_dir = session.get_dir()
    images = session.get_images()
    if image_dir is None or not images:
        raise HTTPException(status_code=400, detail="No image directory opened")

    image_list = [str(image_dir / name) for name in images]
    try:
        result = await asyncio.to_thread(
            create_yolo_dataset,
            image_list,
            req.task_type,
            req.dataset_ratio,
            "",  # data_file: auto-extract classes from the JSON annotations
            None,  # output_dir: labels live next to the images
            None,  # pose_cfg_file
            req.skip_empty_files,
            req.only_checked_files,
        )
    except Exception as e:  # noqa
        raise HTTPException(status_code=500, detail=f"Dataset preparation failed: {e}")

    # create_yolo_dataset signals some misconfigurations (e.g. pose task
    # without a pose config file) by returning (None, message) instead of
    # raising; surface that as a client error instead of crashing.
    if isinstance(result, tuple):
        raise HTTPException(
            status_code=400,
            detail=result[1] or "Dataset preparation failed",
        )
    dataset_dir = result

    val_note = _ensure_val_split(dataset_dir)

    return {
        "dataset_dir": dataset_dir,
        "data_yaml": str(Path(dataset_dir) / "data.yaml"),
        "info": val_note + _read_info(dataset_dir),
    }
