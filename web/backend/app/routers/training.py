"""Guided training (ultralytics) endpoints."""

import asyncio
import io
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ..model_service import get_model_service, register_custom_model_config
from ..training_service import get_training_service

router = APIRouter()


class GuidedStartRequest(BaseModel):
    task: str = "detect"  # detect / segment / classify / pose / obb
    model: str  # e.g. yolov8n.pt
    data: str  # dataset yaml path
    project: str  # output project dir
    name: str = "train"
    device: str = "cpu"
    epochs: Optional[int] = Field(default=100, ge=1)
    batch: Optional[int] = Field(default=16, ge=-1)  # -1 = ultralytics auto-batch
    imgsz: Optional[int] = Field(default=640, ge=32, le=4096)
    patience: Optional[int] = Field(default=None, ge=0)
    lr0: Optional[float] = Field(default=None, ge=0)
    lrf: Optional[float] = Field(default=None, ge=0)
    momentum: Optional[float] = Field(default=None, ge=0)
    weight_decay: Optional[float] = Field(default=None, ge=0)
    warmup_epochs: Optional[float] = Field(default=None, ge=0)
    workers: Optional[int] = Field(default=None, ge=0)
    optimizer: Optional[str] = None
    seed: Optional[int] = Field(default=None, ge=0)
    save_period: Optional[int] = Field(default=None, ge=0)
    val: Optional[bool] = None
    plots: Optional[bool] = None
    # remote execution (SSH)
    execution_mode: str = "local"  # local | remote_ssh（别名: remote）
    remote_profile_id: Optional[str] = None
    remote_password: Optional[str] = None  # 仅本次会话，不落盘

    @field_validator("batch")
    @classmethod
    def _batch_not_zero(cls, v: Optional[int]) -> Optional[int]:
        if v == 0:
            raise ValueError("batch 不能为 0(-1 表示 ultralytics 自动 batch)")
        return v


_MODEL_TASK_SUFFIXES = (
    ("-seg", "segment"),
    ("-obb", "obb"),
    ("-cls", "classify"),
    ("-pose", "pose"),
)


def _infer_task_from_model(model: str) -> Optional[str]:
    """Infer the task from a model filename, or None when undecidable.

    Official ultralytics weights encode the task in the suffix
    (``yolov8n-seg.pt`` etc.); detection weights carry no suffix.
    Custom ``.pt`` names without a known suffix return None (allowed
    through) rather than being misclassified.
    """
    # URLs may carry a query/fragment — strip it before taking the filename
    raw = str(model).split("?", 1)[0].split("#", 1)[0]
    stem = Path(raw).name.lower()
    if stem.endswith(".pt"):
        stem = stem[:-3]
    for suffix, task in _MODEL_TASK_SUFFIXES:
        if stem.endswith(suffix):
            return task
    if stem.startswith("yolo"):  # official detect weights have no suffix
        return "detect"
    return None


def _validate_task_model(req: GuidedStartRequest) -> None:
    """Reject obvious task×model mismatches with a 400 + suggestion."""
    inferred = _infer_task_from_model(req.model)
    task = (req.task or "").lower().strip()
    if inferred is not None and inferred != task:
        name = Path(str(req.model)).name
        raise HTTPException(
            status_code=400,
            detail=(
                f"模型 {name} 是 {inferred} 任务权重，"
                f"请将任务类型改为 {inferred} 或更换模型"
            ),
        )


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
    _validate_task_model(req)
    svc = get_training_service()
    try:
        return await asyncio.to_thread(svc.start_guided, req.model_dump())
    except Exception as e:  # noqa
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/training/preflight")
async def training_preflight(req: GuidedStartRequest):
    """Run preflight checks without starting training."""
    _validate_task_model(req)
    svc = get_training_service()
    try:
        return await asyncio.to_thread(svc.run_preflight, req.model_dump())
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




def _job_record(job_id: str) -> Dict[str, Any]:
    svc = get_training_service()
    for record in svc.history(500):
        if str(record.get("job_id")) == job_id:
            return record
    raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}")


def _job_output_dir(job_id: str) -> Path:
    record = _job_record(job_id)
    out = record.get("output_directory") or record.get("output_dir") or record.get("workspace")
    if not out:
        raise HTTPException(status_code=404, detail=f"Output directory not found for job: {job_id}")
    path = Path(str(out)).expanduser().resolve()
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Output directory not found for job: {job_id}")
    return path


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


# ---- register a trained artifact as an auto-labeling model -------------------

# Task -> auto-labeling model type. Only detect/segment are registrable for
# now; obb/classify/pose have no compatible baseline model type.
_REGISTER_MODEL_TYPES = {"detect": "yolov8", "segment": "yolov8_seg"}


class RegisterModelRequest(BaseModel):
    path: str  # artifact relative path, .pt or .onnx
    display_name: Optional[str] = None


def _infer_register_task(record: Optional[Dict[str, Any]], artifact_name: str) -> str:
    """Map the job's task to a registrable auto-labeling model type."""
    task = (record or {}).get("task")
    if not task:
        stem = Path(artifact_name).stem.lower()
        if stem.endswith("-seg"):
            task = "segment"
        else:
            raise HTTPException(
                status_code=400,
                detail="无法推断该产物的任务类型（history 记录缺少 task 字段）",
            )
    if task not in _REGISTER_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"任务类型 '{task}' 暂不支持注册为标注模型（目前仅支持 detect / segment）",
        )
    return task


def _resolve_run_data_yaml(base: Path, data: Any) -> Path:
    data_path = Path(str(data)).expanduser()
    # Only trust the recorded absolute path or a path relative to the run
    # directory — never the web process CWD.
    candidates = [data_path] if data_path.is_absolute() else [base / data_path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise HTTPException(
        status_code=400,
        detail=f"训练参数指向的数据配置不存在: {data}，无法推断类别列表",
    )


def _load_run_training_meta(
    base: Path, record: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Read classes + imgsz from the run's args.yaml / data.yaml.

    Remote (SSH) runs download an args.yaml whose ``data`` points at a
    *remote* path that does not exist locally, so resolution falls back
    to the local data.yaml recorded in the job history (``record["data"]``)
    before giving up with a 400.
    """
    args: Dict[str, Any] = {}
    args_file = base / "args.yaml"
    if args_file.is_file():
        try:
            loaded = yaml.safe_load(args_file.read_text(encoding="utf-8")) or {}
        except Exception as e:  # noqa
            raise HTTPException(status_code=400, detail=f"args.yaml 解析失败: {e}")
        if isinstance(loaded, dict):
            args = loaded

    data_yaml: Optional[Path] = None
    if args.get("data"):
        try:
            data_yaml = _resolve_run_data_yaml(base, args["data"])
        except HTTPException:
            data_yaml = None  # e.g. remote path of an SSH run — try history
    if data_yaml is None:
        recorded = (record or {}).get("data")
        if recorded:
            try:
                data_yaml = _resolve_run_data_yaml(base, recorded)
            except HTTPException:
                data_yaml = None
    if data_yaml is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "无法推断类别列表：args.yaml 缺失或其中的 data 配置不可读，"
                "且 history 记录中没有可用的本地数据配置"
            ),
        )

    try:
        data_cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa
        raise HTTPException(status_code=400, detail=f"数据配置 {data_yaml.name} 解析失败: {e}")
    names = data_cfg.get("names") if isinstance(data_cfg, dict) else None
    classes: List[str] = []
    if isinstance(names, dict):
        by_index: Dict[int, Any] = {}
        for k, v in names.items():
            try:
                by_index[int(k)] = v
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail=f"数据配置 names 字段的类别序号必须是整数，实际为 {k!r}",
                )
        if len(by_index) != len(names) or sorted(by_index) != list(range(len(by_index))):
            raise HTTPException(
                status_code=400,
                detail=(
                    "数据配置 names 字段的类别序号必须是从 0 开始的连续整数 "
                    f"(0..{len(names) - 1})，当前为 {sorted(by_index)}"
                ),
            )
        classes = [str(by_index[i]) for i in range(len(by_index))]
    elif isinstance(names, list):
        classes = [str(n) for n in names]
    if not classes:
        raise HTTPException(
            status_code=400,
            detail=f"数据配置 {data_yaml.name} 中缺少有效的 names 类别列表",
        )

    try:
        imgsz = int(args.get("imgsz") or 640)
    except (TypeError, ValueError):
        imgsz = 640
    return {"classes": classes, "imgsz": imgsz}


def _build_model_yaml_config(
    task: str,
    job_id: str,
    onnx_path: Path,
    meta: Dict[str, Any],
    display_name: Optional[str],
) -> Dict[str, Any]:
    # job_id looks like guided_20260814_122455_ab12cd — use its uuid tail
    # plus the artifact stem so that best/last of the same job (and the
    # same stem across jobs) never collapse onto a single model name.
    uuid_seg = re.sub(r"[^A-Za-z0-9_-]", "", job_id.rsplit("_", 1)[-1]) or "job"
    stem = re.sub(r"[^A-Za-z0-9_-]", "", onnx_path.stem) or "model"
    name = f"trainlens_{task}_{uuid_seg}_{stem}"
    parts = job_id.split("_")
    stamp = "_".join(parts[1:3]) if len(parts) >= 4 else uuid_seg
    config: Dict[str, Any] = {
        "type": _REGISTER_MODEL_TYPES[task],
        "name": name,
        "display_name": display_name or f"TrainLens {task} {stamp} {stem}",
        # bare filename: resolved relative to this YAML's directory
        "model_path": onnx_path.name,
    }
    if task == "detect":
        imgsz = meta["imgsz"]
        config.update(
            {
                "input_width": imgsz,
                "input_height": imgsz,
                "score_threshold": 0.25,
                "nms_threshold": 0.45,
                "confidence_threshold": 0.25,
                "classes": meta["classes"],
            }
        )
    else:  # segment
        config.update(
            {
                "conf_threshold": 0.25,
                "iou_threshold": 0.45,
                "classes": meta["classes"],
            }
        )
    return config


@router.post("/training/history/{job_id}/artifacts/register-model")
async def training_artifact_register_model(job_id: str, req: RegisterModelRequest):
    """Register a trained .pt/.onnx artifact as a loadable auto-labeling model."""
    base = _job_output_dir(job_id)
    target = _safe_child(base, req.path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {req.path}")
    suffix = target.suffix.lower()
    if suffix not in (".pt", ".onnx"):
        raise HTTPException(status_code=400, detail="仅支持注册 .pt / .onnx 产物")

    if suffix == ".pt":
        holder: dict = {}
        await asyncio.to_thread(_run_model_export, job_id, req.path, "onnx", holder)
        if "error" in holder:
            raise HTTPException(status_code=500, detail=f"ONNX 导出失败: {holder['error']}")
        onnx_path = Path(holder["output"]).resolve()
        if not onnx_path.is_file():
            raise HTTPException(status_code=500, detail=f"ONNX 导出产物不存在: {onnx_path}")
    else:
        onnx_path = target

    record = _job_record(job_id)
    task = _infer_register_task(record, target.name)
    meta = _load_run_training_meta(base, record)
    model_config = _build_model_yaml_config(task, job_id, onnx_path, meta, req.display_name)

    config_file = onnx_path.parent / f"{model_config['name']}.yaml"
    config_file.write_text(
        yaml.safe_dump(model_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    manager = get_model_service().manager
    register_custom_model_config(manager, model_config, config_file)

    return {
        "registered": True,
        "config_file": str(config_file),
        "display_name": model_config["display_name"],
        "model_type": model_config["type"],
    }


# ---- dynamic export format availability --------------------------------------


def _collect_export_formats() -> List[Dict[str, Any]]:
    try:
        from anylabeling.services.auto_training.ultralytics.export_formats import (
            FORMAT_BY_CODE,
            FormatStatus,
            get_format_status,
            get_missing_pip_packages,
        )
    except Exception:  # noqa - registry unavailable: expose bare list
        return [
            {"id": code, "name": code, "available": True, "reason": None}
            for code in _SUPPORTED_EXPORT_FORMATS
        ]

    formats: List[Dict[str, Any]] = []
    for code in _SUPPORTED_EXPORT_FORMATS:
        info = FORMAT_BY_CODE.get(code)
        if info is None:
            # not tracked by the registry (e.g. tflite, deprecated upstream):
            # no environment check available
            formats.append({"id": code, "name": code, "available": True, "reason": None})
            continue
        status = get_format_status(info)
        reason = None
        if status == FormatStatus.MISSING_DEPENDENCY:
            reason = "缺少依赖: " + ", ".join(get_missing_pip_packages(info))
        elif status == FormatStatus.UNSUPPORTED_PLATFORM:
            reason = "当前平台不支持该格式"
        elif status == FormatStatus.UNSUPPORTED_DEVICE:
            reason = "需要可用的 NVIDIA GPU"
        formats.append(
            {
                "id": code,
                "name": info.display_name,
                "available": status == FormatStatus.READY,
                "reason": reason,
            }
        )
    return formats


@router.get("/training/export-formats")
async def training_export_formats():
    """List export formats with environment availability for the artifact dialog."""
    return {"formats": await asyncio.to_thread(_collect_export_formats)}


