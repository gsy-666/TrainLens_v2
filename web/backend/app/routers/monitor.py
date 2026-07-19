"""Run monitor endpoints: scan workspace, run custom scripts, stream
output and resource metrics."""

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..run_service import get_run_service

router = APIRouter()


class ScanRequest(BaseModel):
    workspace: str


class StartRunRequest(BaseModel):
    workspace: str
    script_path: str
    python_path: str = ""
    arguments: str = ""


@router.post("/monitor/scan")
async def scan_workspace(req: ScanRequest):
    svc = get_run_service()
    try:
        return await asyncio.to_thread(svc.scan, req.workspace)
    except Exception as e:  # noqa
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/monitor/start")
async def start_run(req: StartRunRequest):
    svc = get_run_service()
    try:
        return await asyncio.to_thread(
            svc.start, req.workspace, req.script_path, req.python_path, req.arguments
        )
    except Exception as e:  # noqa
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/monitor/stop")
def stop_run():
    svc = get_run_service()
    return svc.stop()


@router.get("/monitor/status")
def run_status():
    svc = get_run_service()
    return svc.status()


@router.get("/monitor/logs")
def run_logs(since: int = 0):
    svc = get_run_service()
    return svc.logs_since(since)


@router.get("/monitor/resources")
def run_resources(limit: int = 300):
    svc = get_run_service()
    return svc.resource_history(limit)
