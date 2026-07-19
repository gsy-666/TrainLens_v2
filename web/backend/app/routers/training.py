"""Guided training (ultralytics) endpoints."""

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..training_service import get_training_service

router = APIRouter()


class GuidedStartRequest(BaseModel):
    task: str = "detect"  # detect / segment / classify / pose / obb
    model: str  # e.g. yolov8n.pt
    data: str  # dataset yaml path
    project: str  # output project dir
    name: str = "train"
    device: str = "cpu"
    epochs: Optional[int] = 100
    batch: Optional[int] = 16
    imgsz: Optional[int] = 640
    patience: Optional[int] = None
    lr0: Optional[float] = None
    lrf: Optional[float] = None
    momentum: Optional[float] = None
    weight_decay: Optional[float] = None
    warmup_epochs: Optional[float] = None
    workers: Optional[int] = None
    optimizer: Optional[str] = None
    seed: Optional[int] = None
    save_period: Optional[int] = None
    val: Optional[bool] = None
    plots: Optional[bool] = None


class PreflightRequest(BaseModel):
    task: str = "detect"
    model: str
    data: str
    project: str
    name: str = "train"
    device: str = "cpu"
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640


@router.post("/training/guided/start")
async def start_guided(req: GuidedStartRequest):
    svc = get_training_service()
    try:
        return await asyncio.to_thread(svc.start_guided, req.model_dump())
    except Exception as e:  # noqa
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/training/stop")
def stop_training():
    svc = get_training_service()
    return svc.stop()


@router.get("/training/status")
def training_status():
    svc = get_training_service()
    return svc.status()


@router.get("/training/events")
def training_events(since: int = 0):
    svc = get_training_service()
    return svc.events_since(since)


@router.get("/training/metrics")
def training_metrics():
    svc = get_training_service()
    return svc.metrics()


@router.get("/training/history")
def training_history(limit: int = 50):
    svc = get_training_service()
    return {"jobs": svc.history(limit)}


@router.post("/training/preflight")
async def training_preflight(req: PreflightRequest):
    svc = get_training_service()
    try:
        return await asyncio.to_thread(svc.run_preflight, req.model_dump())
    except Exception as e:  # noqa
        raise HTTPException(status_code=400, detail=str(e))
