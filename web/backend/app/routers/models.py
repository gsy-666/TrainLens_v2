"""AI model catalog / load / unload endpoints."""

import asyncio
import copy
import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from anylabeling.config import get_config, save_config

from ..model_service import get_model_service, register_custom_model_config

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


# ---- local weight registration -------------------------------------------------

# config keys copied from the template model into the local-weight config
_TEMPLATE_KEYS = (
    "type",
    "provider",
    "input_width",
    "input_height",
    "score_threshold",
    "nms_threshold",
    "confidence_threshold",
    "conf_threshold",
    "iou_threshold",
    "classes",
)


def _models_root() -> Path:
    """Model cache root (honors the XANYLABELING_MODELS_DIR override)."""
    custom = os.environ.get("XANYLABELING_MODELS_DIR")
    if custom:
        return Path(custom).expanduser()
    return Path.home() / "anylabeling_data" / "models"


class RegisterLocalModelRequest(BaseModel):
    template_config_file: str  # built-in model whose config acts as template
    local_path: str  # local .onnx file
    display_name: Optional[str] = None


@router.post("/models/register-local")
def register_local_model(req: RegisterLocalModelRequest):
    """Register a local .onnx file as a loadable model, reusing an existing
    model's config (type/thresholds/classes) as the template."""
    svc = get_model_service()

    local = Path(req.local_path).expanduser()
    if not local.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {req.local_path}")
    if local.suffix.lower() != ".onnx":
        raise HTTPException(status_code=400, detail="自动标注仅支持 .onnx 权重文件")
    local = local.resolve()

    template = None
    for cfg in svc.manager.model_configs:
        if cfg.get("config_file") == req.template_config_file:
            template = cfg
            break
    if template is None:
        raise HTTPException(
            status_code=404, detail=f"模板模型不存在: {req.template_config_file}"
        )

    model_config = {
        k: copy.deepcopy(template[k]) for k in _TEMPLATE_KEYS if k in template
    }
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", local.stem)
    digest = hashlib.md5(str(local).encode()).hexdigest()[:6]
    model_config["name"] = f"local_{template.get('type', 'model')}_{stem}_{digest}"
    template_label = template.get("display_name") or template.get("name") or "model"
    model_config["display_name"] = (
        req.display_name or f"{template_label}(本地 {local.name})"
    )
    # absolute path: get_model_abs_path accepts existing absolute paths as-is
    model_config["model_path"] = str(local)

    config_dir = _models_root() / "_local"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / f"{model_config['name']}.yaml"
    config_file.write_text(
        yaml.safe_dump(model_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    register_custom_model_config(svc.manager, model_config, config_file)

    return {
        "registered": True,
        "config_file": str(config_file),
        "display_name": model_config["display_name"],
        "model_type": model_config["type"],
    }


@router.delete("/models/custom")
def delete_custom_model(config_file: str, delete_file: bool = False):
    """Remove a custom (user-registered) model from the catalog.

    Only entries in the user's ``custom_models`` list can be removed.
    ``delete_file=true`` additionally deletes the YAML config file, but only
    when it lives under the managed ``_local`` directory — user files in
    arbitrary locations are never touched.
    """
    svc = get_model_service()
    norm = os.path.normpath(config_file)

    config = get_config()
    custom_models = config.get("custom_models") or []
    remaining = [
        m for m in custom_models
        if os.path.normpath(str(m.get("config_file", ""))) != norm
    ]
    if len(remaining) == len(custom_models):
        raise HTTPException(status_code=404, detail=f"自定义模型不存在: {config_file}")

    config["custom_models"] = remaining
    save_config(config)
    svc.manager.load_model_configs()

    deleted_file = False
    path = Path(config_file)
    if (
        delete_file
        and path.is_file()
        and path.parent.name == "_local"
        and path.suffix.lower() in (".yaml", ".yml")
    ):
        try:
            path.unlink()
            deleted_file = True
        except OSError:
            pass
    return {"deleted": True, "deleted_file": deleted_file}


# ---- local model library scan ---------------------------------------------------


def _expected_weight_path(cfg: Dict[str, Any]) -> Optional[Path]:
    """Where this catalog model's weight file lives (or would be downloaded).

    http(s) models: <models_root>/<name>/<filename-from-url>.
    Local-path models: the path itself (relative paths resolve against the
    YAML's directory, mirroring get_model_abs_path)."""
    model_path = str(cfg.get("model_path") or "")
    name = str(cfg.get("name") or "")
    if model_path.startswith(("http://", "https://")):
        if not name:
            return None
        filename = os.path.basename(urlparse(model_path).path)
        return (_models_root() / name / filename) if filename else None
    if not model_path:
        return None
    p = Path(model_path).expanduser()
    if p.is_absolute():
        return p
    config_file = str(cfg.get("config_file") or "")
    if config_file and not config_file.startswith(":/") and Path(config_file).is_file():
        return Path(config_file).parent / p
    return None


def _file_entry(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "size_bytes": st.st_size,
        "modified_at": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }


@router.get("/models/local-files")
def local_model_files():
    """Model library scan: download state + disk usage for every catalog model.

    ``downloaded`` marks models already on disk (load will skip downloading);
    custom models whose files went missing show up with ``downloaded=false``.
    """
    svc = get_model_service()
    items = []
    total = 0
    for cfg in svc.manager.model_configs:
        expected = _expected_weight_path(cfg)
        downloaded = bool(expected and expected.is_file())
        size = expected.stat().st_size if downloaded and expected else 0
        total += size
        items.append(
            {
                "config_file": cfg.get("config_file"),
                "name": cfg.get("name"),
                "display_name": cfg.get("display_name"),
                "type": cfg.get("type"),
                "is_custom_model": cfg.get("is_custom_model", False),
                "downloaded": downloaded,
                "path": str(expected) if expected else None,
                "size_bytes": size,
            }
        )
    return {
        "root": str(_models_root()),
        "items": items,
        "total_bytes": total,
    }


@router.delete("/models/cache")
def delete_model_cache(config_file: str):
    """Delete a downloaded catalog model's cached weight file (frees disk).

    Only http-downloaded cache files under the models root are deletable;
    custom-model weight files on user paths are never touched."""
    svc = get_model_service()
    target_cfg = None
    for cfg in svc.manager.model_configs:
        if cfg.get("config_file") == config_file:
            target_cfg = cfg
            break
    if target_cfg is None:
        raise HTTPException(status_code=404, detail=f"模型不存在: {config_file}")
    if target_cfg.get("is_custom_model"):
        raise HTTPException(status_code=400, detail="自定义模型请使用「移除」而不是删除缓存")

    model_path = str(target_cfg.get("model_path") or "")
    if not model_path.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="该模型不是可下载缓存")

    expected = _expected_weight_path(target_cfg)
    root = _models_root().resolve()
    if not expected:
        raise HTTPException(status_code=404, detail="该模型没有缓存文件")
    resolved = expected.resolve()
    if root != resolved and root not in resolved.parents:
        raise HTTPException(status_code=400, detail="缓存路径不在模型目录内")
    if not resolved.is_file():
        return {"deleted": False, "freed_bytes": 0}

    freed = resolved.stat().st_size
    try:
        resolved.unlink()
        # drop the per-model dir if it is now empty
        if resolved.parent != root:
            try:
                resolved.parent.rmdir()
            except OSError:
                pass
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")
    return {"deleted": True, "freed_bytes": freed}


@router.get("/models/scan-dir")
def scan_models_dir(path: str):
    """Discover stray .onnx weight files under a user-chosen directory."""
    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"不是目录: {path}")
    files = []
    try:
        for p in sorted(directory.rglob("*")):
            if len(files) >= 200:
                break
            try:
                if p.is_file() and p.suffix.lower() == ".onnx":
                    files.append(_file_entry(p))
            except OSError:
                continue
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"无权限访问: {path}")
    files.sort(key=lambda f: f["modified_at"], reverse=True)
    return {"files": files}
