"""Auto-labeling inference endpoints (single image + batch)."""

import asyncio
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel

from .. import backup
from ..adapters import label_path_for, load_label_data, save_label_data
from ..deps import session
from ..model_service import get_model_service

router = APIRouter()


def _resolve_image(name: str):
    try:
        path = session.resolve_image(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Image not found: {name}")
    return path


class PredictRequest(BaseModel):
    image: str  # filename inside the opened dir
    text_prompt: Optional[str] = None
    conf: Optional[float] = None
    iou: Optional[float] = None


@router.post("/predict")
async def predict(req: PredictRequest):
    svc = get_model_service()
    image_path = _resolve_image(req.image)
    if not svc.loaded_info():
        raise HTTPException(status_code=400, detail="No model loaded")
    try:
        result = await asyncio.to_thread(
            svc.predict, str(image_path), req.text_prompt, req.conf, req.iou
        )
    except Exception as e:  # noqa
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return result


class BatchPredictRequest(BaseModel):
    images: List[str]
    preserve_existing: bool = False
    conf: Optional[float] = None
    iou: Optional[float] = None


def _run_batch(req: BatchPredictRequest):
    svc = get_model_service()
    svc.batch_state = {
        "running": True,
        "current": 0,
        "total": len(req.images),
        "current_image": None,
        "errors": [],
    }
    backup.start_backup("batch", str(session.get_dir()))
    try:
        for idx, name in enumerate(req.images):
            svc.batch_state["current"] = idx
            svc.batch_state["current_image"] = name
            try:
                image_path = session.resolve_image(name)
                result = svc.predict(str(image_path), None, req.conf, req.iou)
                new_shapes = result["shapes"]

                existing_shapes = []
                flags, other = {}, {}
                label_path = label_path_for(image_path)
                existed = label_path.exists()
                original_content = label_path.read_bytes() if existed else None
                if existed:
                    try:
                        data = load_label_data(label_path)
                        existing_shapes = data.get("shapes", [])
                        flags = data.get("flags", {})
                        other = {
                            k: v
                            for k, v in data.items()
                            if k
                            not in (
                                "version",
                                "flags",
                                "checked",
                                "shapes",
                                "imagePath",
                                "imageData",
                                "imageHeight",
                                "imageWidth",
                            )
                        }
                    except Exception:
                        pass

                if req.preserve_existing and not result.get("replace", True):
                    shapes = existing_shapes + new_shapes
                elif req.preserve_existing:
                    shapes = existing_shapes + new_shapes
                else:
                    shapes = new_shapes

                with Image.open(image_path) as im:
                    w, h = im.size
                save_label_data(
                    image_path=image_path,
                    shapes=shapes,
                    flags=flags,
                    other_data=other,
                    image_height=h,
                    image_width=w,
                )
                backup.add_item(name, label_path, existed, original_content)
            except Exception as e:  # noqa
                svc.batch_state["errors"].append({"image": name, "error": str(e)})
    finally:
        svc.batch_state["current"] = len(req.images)
        svc.batch_state["running"] = False


@router.post("/predict/batch")
async def predict_batch(req: BatchPredictRequest):
    svc = get_model_service()
    if not svc.loaded_info():
        raise HTTPException(status_code=400, detail="No model loaded")
    if svc.batch_state and svc.batch_state.get("running"):
        raise HTTPException(status_code=400, detail="Batch already running")
    asyncio.create_task(asyncio.to_thread(_run_batch, req))
    return {"started": True, "total": len(req.images)}


@router.get("/predict/batch/status")
def batch_status():
    svc = get_model_service()
    state = dict(svc.batch_state or {"running": False})
    bk = backup.get_backup()
    state["undo_available"] = bool(bk and bk["items"])
    state["backup_count"] = len(bk["items"]) if bk else 0
    state["batch_id"] = bk["batch_id"] if bk else None
    return state


@router.post("/predict/batch/undo")
def undo_batch():
    """Restore label files to their pre-batch state.

    Files whose content changed since the batch wrote them (e.g. manual
    edits afterwards) are skipped and reported.
    """
    try:
        return backup.undo()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
