"""Playground: one-off inference on an uploaded image.

Unlike /api/predict (which runs on images inside the opened dataset
directory), this endpoint takes a multipart upload, runs the currently
loaded model on it and returns the shapes. It never touches the session
or writes into any dataset directory.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..deps import IMAGE_EXTENSIONS
from ..model_service import get_model_service

router = APIRouter()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB


@router.post("/playground/predict")
async def playground_predict(
    file: UploadFile = File(...),
    conf: Optional[float] = Form(None),
    iou: Optional[float] = Form(None),
):
    svc = get_model_service()
    loaded = svc.loaded_info()
    if not loaded:
        raise HTTPException(
            status_code=400,
            detail="请先在标注页加载模型或在训练产物中点「试用」",
        )

    suffix = Path(file.filename or "").suffix.lower()
    content_type = (file.content_type or "").lower()
    if suffix not in IMAGE_EXTENSIONS and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image too large (max 20MB)")

    # mkstemp + fdopen avoids the NamedTemporaryFile handle-lock on Windows;
    # the file is always removed in the finally block.
    fd, tmp_path = tempfile.mkstemp(suffix=suffix or ".png")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        result = await asyncio.to_thread(svc.predict, tmp_path, None, conf, iou)
    except HTTPException:
        raise
    except Exception as e:  # noqa
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return {
        "shapes": result["shapes"],
        "model": {"display_name": loaded["display_name"], "type": loaded["type"]},
    }
