"""Close the annotation -> training loop: build a YOLO training dataset
from the currently opened annotation directory."""

import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from anylabeling.services.auto_training.ultralytics.general import (
    create_yolo_dataset,
)

from ..deps import session

router = APIRouter()

TASK_TYPES = ["Detect", "OBB", "Segment", "Pose", "Classify"]


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
        dataset_dir = await asyncio.to_thread(
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

    val_note = _ensure_val_split(dataset_dir)

    return {
        "dataset_dir": dataset_dir,
        "data_yaml": str(Path(dataset_dir) / "data.yaml"),
        "info": val_note + _read_info(dataset_dir),
    }
