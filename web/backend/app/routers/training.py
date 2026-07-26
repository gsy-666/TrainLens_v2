"""Guided training (ultralytics) endpoints."""

import asyncio
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
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


class ArtifactInfo(BaseModel):
    name: str
    relative_path: str
    size: int
    modified_at: str
    is_downloadable: bool = True
    file_type: str


class ArtifactListResponse(BaseModel):
    job_id: str
    output_dir: str
    artifacts: List[ArtifactInfo]


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




def _job_output_dir(job_id: str) -> Path:
    svc = get_training_service()
    for record in svc.history(500):
        if str(record.get("job_id")) == job_id:
            out = record.get("output_directory") or record.get("output_dir") or record.get("workspace")
            if not out:
                raise HTTPException(status_code=404, detail=f"Output directory not found for job: {job_id}")
            path = Path(str(out)).expanduser().resolve()
            if not path.is_dir():
                raise HTTPException(status_code=404, detail=f"Output directory not found for job: {job_id}")
            return path
    raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}")


def _safe_child(base: Path, rel: str) -> Path:
    candidate = (base / rel).resolve()
    if candidate == base or base not in candidate.parents:
        raise HTTPException(status_code=400, detail=f"Path escapes job output directory: {rel}")
    return candidate


def _file_info(path: Path, base: Path) -> ArtifactInfo:
    rel = path.relative_to(base).as_posix()
    st = path.stat()
    return ArtifactInfo(
        name=path.name,
        relative_path=rel,
        size=st.st_size,
        modified_at=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        is_downloadable=path.is_file(),
        file_type=path.suffix.lstrip(".").lower() or "file",
    )


def _list_job_artifacts(base: Path) -> List[ArtifactInfo]:
    artifacts: List[ArtifactInfo] = []
    for path in sorted(base.rglob("*")):
        try:
            resolved = path.resolve()
        except Exception:
            continue
        if base not in resolved.parents and resolved != base:
            continue
        if resolved.is_file():
            artifacts.append(_file_info(resolved, base))
    return artifacts


@router.get("/training/history/{job_id}/artifacts", response_model=ArtifactListResponse)
def training_artifacts(job_id: str):
    base = _job_output_dir(job_id)
    return ArtifactListResponse(job_id=job_id, output_dir=str(base), artifacts=_list_job_artifacts(base))


@router.get("/training/history/{job_id}/artifacts/download")
def training_artifact_download(job_id: str, path: str):
    base = _job_output_dir(job_id)
    target = _safe_child(base, path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path}")
    return FileResponse(target, filename=target.name)


@router.get("/training/history/{job_id}/artifacts/download-all")
def training_artifact_download_all(job_id: str):
    base = _job_output_dir(job_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for artifact in _list_job_artifacts(base):
            fp = _safe_child(base, artifact.relative_path)
            if fp.is_file():
                zf.write(fp, artifact.relative_path)
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{job_id}_artifacts.zip"'})


class ExportModelRequest(BaseModel):
    path: str  # artifact relative path, must be a .pt under weights/
    format: str = "onnx"  # onnx / engine / openvino / coreml / tflite / torchscript


_SUPPORTED_EXPORT_FORMATS = ("onnx", "engine", "openvino", "coreml", "tflite", "torchscript")


def _run_model_export(job_id: str, rel_path: str, fmt: str, holder: dict):
    try:
        from ultralytics import YOLO

        base = _job_output_dir(job_id)
        target = _safe_child(base, rel_path)
        model = YOLO(str(target))
        out = model.export(format=fmt)
        holder["output"] = str(out)
    except Exception as e:  # noqa
        holder["error"] = str(e)


@router.post("/training/history/{job_id}/artifacts/export")
async def training_artifact_export(job_id: str, req: ExportModelRequest):
    """Export a trained .pt artifact to a deployable format (onnx etc.)."""
    base = _job_output_dir(job_id)
    target = _safe_child(base, req.path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {req.path}")
    if target.suffix.lower() != ".pt":
        raise HTTPException(status_code=400, detail="Only .pt artifacts can be exported")
    if req.format not in _SUPPORTED_EXPORT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{req.format}', choose from {_SUPPORTED_EXPORT_FORMATS}",
        )

    holder: dict = {}
    await asyncio.to_thread(_run_model_export, job_id, req.path, req.format, holder)
    if "error" in holder:
        raise HTTPException(status_code=500, detail=f"Export failed: {holder['error']}")

    out_path = Path(holder["output"]).resolve()
    try:
        rel = out_path.relative_to(base.resolve())
    except ValueError:
        raise HTTPException(status_code=500, detail=f"Unexpected export path: {out_path}")
    return {"exported": True, "relative_path": str(rel), "format": req.format}


