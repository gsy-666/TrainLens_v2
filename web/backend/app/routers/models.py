"""AI model catalog / load / unload endpoints."""

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..model_service import get_model_service

router = APIRouter()


class LoadModelRequest(BaseModel):
    config_file: str


@router.get("/models")
def list_models():
    svc = get_model_service()
    return {
        "models": svc.list_models(),
        "loaded": svc.loaded_info(),
    }


@router.post("/models/load")
async def load_model(req: LoadModelRequest):
    svc = get_model_service()
    try:
        await asyncio.to_thread(svc.load, req.config_file)
    except Exception as e:  # noqa
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}")
    return {"loaded": svc.loaded_info()}


@router.post("/models/unload")
def unload_model():
    svc = get_model_service()
    svc.unload()
    return {"loaded": None}


class OutputModeRequest(BaseModel):
    mode: str


@router.post("/models/output_mode")
def set_output_mode(req: OutputModeRequest):
    svc = get_model_service()
    try:
        return svc.set_output_mode(req.mode)
    except Exception as e:  # noqa
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/models/status")
def model_status():
    svc = get_model_service()
    return {
        "loading": svc.loading,
        "error": svc.load_error,
        "progress": svc.progress,
        "message": svc.status_message,
        "loaded": svc.loaded_info(),
    }
