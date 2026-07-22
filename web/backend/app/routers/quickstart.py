"""One-click training: infer task type from annotations, generate the
YOLO dataset, pick sensible defaults and start training — zero form
filling required."""

import asyncio
import datetime
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from anylabeling.services.auto_training.ultralytics.general import (
    create_yolo_dataset,
)

from ..adapters import label_path_for, load_label_data
from ..deps import session
from ..training_service import get_training_service
from .dataset import _ensure_val_split, _read_info
from .system import _detect_device

router = APIRouter()

_SHAPE_TO_TASK = {
    "rectangle": "Detect",
    "polygon": "Segment",
    "rotation": "OBB",
}

_MODEL_BY_TASK = {
    "Detect": "yolov8n.pt",
    "Segment": "yolov8n-seg.pt",
    "OBB": "yolov8n-obb.pt",
}


def _infer_task_type(image_dir: Path, images: list[str], sample: int = 50) -> str:
    """Dominant shape type across a sample of label files."""
    votes: Counter = Counter()
    for name in images[:sample]:
        label_path = label_path_for(image_dir / name)
        if not label_path.exists():
            continue
        try:
            for shape in load_label_data(label_path).get("shapes", []):
                task = _SHAPE_TO_TASK.get(shape.get("shape_type", ""))
                if task:
                    votes[task] += 1
        except Exception:
            continue
    if not votes:
        return "Detect"
    return votes.most_common(1)[0][0]


class QuickstartRequest(BaseModel):
    task_type: str | None = None  # default: inferred from shapes
    dataset_ratio: float = 0.9
    epochs: int = 100
    model: str | None = None  # default: per-task yolov8n variant
    device: str | None = None  # default: auto-detected


@router.post("/training/quickstart")
async def training_quickstart(req: QuickstartRequest):
    image_dir = session.get_dir()
    images = session.get_images()
    if image_dir is None or not images:
        raise HTTPException(status_code=400, detail="No image directory opened")

    task = req.task_type or _infer_task_type(image_dir, images)
    device_info = _detect_device()
    device = req.device or device_info["recommended"]
    model = req.model or _MODEL_BY_TASK.get(task, "yolov8n.pt")
    batch = -1 if device != "cpu" else 16  # ultralytics auto-batch on GPU

    # 1. build dataset
    image_list = [str(image_dir / n) for n in images]
    try:
        dataset_dir = await asyncio.to_thread(
            create_yolo_dataset, image_list, task, req.dataset_ratio, "", None, None, False, False
        )
        val_note = _ensure_val_split(dataset_dir)
    except Exception as e:  # noqa
        raise HTTPException(status_code=500, detail=f"数据集生成失败: {e}")

    # 2. start training — artifacts live next to the dataset by default
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    params = {
        "task": task.lower(),
        "model": model,
        "data": str(Path(dataset_dir) / "data.yaml"),
        "project": str(Path(dataset_dir) / "runs"),
        "name": f"{task.lower()}_{ts}",
        "device": device,
        "epochs": req.epochs,
        "batch": batch,
        # spawned train-worker + multiprocessing dataloader hangs on Windows;
        # single-process loading is the safe default for one-click runs
        "workers": 0,
    }
    svc = get_training_service()
    try:
        status = await asyncio.to_thread(svc.start_guided, params)
    except Exception as e:  # noqa
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "task_type": task,
        "device": device,
        "model": model,
        "dataset_dir": dataset_dir,
        "dataset_info": val_note + _read_info(dataset_dir),
        "job": status["job"],
    }
