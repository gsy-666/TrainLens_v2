"""Remote SSH training-server profile endpoints.

Wraps the desktop training_center.remote building blocks (ProfileStore,
SSHConnectionService, run_remote_diagnostics) — all Qt-free. Blocking
SSH calls run in worker threads via asyncio.to_thread. Passwords are
session-only: accepted per request, never persisted by ProfileStore.
"""

import asyncio
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from anylabeling.services.training_center.remote.models import (
    AuthMethod,
    RemoteProfile,
)
from anylabeling.services.training_center.remote.storage import get_profile_store

router = APIRouter()


class RemoteProfilePayload(BaseModel):
    name: str
    host: str
    port: int = Field(default=22, ge=1, le=65535)
    username: str
    auth_method: str = "ssh_key"  # ssh_key | password
    private_key_path: str = ""
    remote_workspace: str
    remote_python: str


class PasswordPayload(BaseModel):
    password: Optional[str] = None


class HostKeyConfirmPayload(BaseModel):
    fingerprint: str


def _get_or_404(profile_id: str) -> RemoteProfile:
    profile = get_profile_store().get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"远程服务器档案不存在: {profile_id}")
    return profile


def _apply_payload(profile: RemoteProfile, req: RemoteProfilePayload) -> None:
    try:
        auth = AuthMethod(req.auth_method)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的认证方式: {req.auth_method!r}（可选: ssh_key / password）",
        )
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="名称不能为空")
    if not req.host.strip():
        raise HTTPException(status_code=400, detail="主机地址不能为空")
    if not req.username.strip():
        raise HTTPException(status_code=400, detail="用户名不能为空")
    if not req.remote_workspace.strip():
        raise HTTPException(status_code=400, detail="远程工作目录不能为空")
    if not req.remote_python.strip():
        raise HTTPException(status_code=400, detail="远程 Python 路径不能为空")
    profile.name = req.name.strip()
    profile.host = req.host.strip()
    profile.port = req.port
    profile.username = req.username.strip()
    profile.auth_method = auth
    profile.private_key_path = req.private_key_path.strip()
    profile.remote_workspace = req.remote_workspace.strip()
    profile.remote_python = req.remote_python.strip()


# ---- profile CRUD -------------------------------------------------------------


@router.get("/remote/profiles")
def list_profiles():
    return {"profiles": [p.to_dict() for p in get_profile_store().list_all()]}


@router.post("/remote/profiles")
def create_profile(req: RemoteProfilePayload):
    profile = RemoteProfile(profile_id=uuid.uuid4().hex[:12])
    _apply_payload(profile, req)
    get_profile_store().save(profile)
    return {"profile": profile.to_dict()}


@router.put("/remote/profiles/{profile_id}")
def update_profile(profile_id: str, req: RemoteProfilePayload):
    profile = _get_or_404(profile_id)
    old_host, old_port = profile.host, profile.port
    _apply_payload(profile, req)
    if profile.host != old_host or profile.port != old_port:
        # The saved host-key fingerprint belongs to the old endpoint —
        # force re-confirmation against the new host:port.
        profile.known_host_fingerprint = ""
    get_profile_store().save(profile)
    return {"profile": profile.to_dict()}


@router.delete("/remote/profiles/{profile_id}")
def delete_profile(profile_id: str):
    _get_or_404(profile_id)
    get_profile_store().delete(profile_id)
    return {"deleted": True}


# ---- connection test + host key confirmation -----------------------------------


def _test_connection(profile: RemoteProfile, password: str):
    """Blocking SSH connect (runs in a worker thread).

    The on_host_key callback enforces TOFU against the saved fingerprint;
    an unknown fingerprint is captured and surfaced for confirmation.
    """
    from anylabeling.services.training_center.remote.ssh_service import (
        SSHConnectionService,
    )

    saved = (profile.known_host_fingerprint or "").strip()
    holder = {"presented": ""}

    def _on_host_key(hostname, fingerprint, key_type):
        holder["presented"] = fingerprint
        if saved and saved == fingerprint:
            return True, False
        return False, False

    ssh = SSHConnectionService()
    try:
        ok, msg = ssh.connect(profile, password=password, on_host_key=_on_host_key)
    finally:
        ssh.close()

    if ok:
        fingerprint = msg if ":" in msg else holder["presented"]
        return {"ok": True, "need_host_key_confirm": False, "fingerprint": fingerprint}

    presented = holder["presented"]
    if presented and not saved:
        return {"ok": False, "need_host_key_confirm": True, "fingerprint": presented}
    if presented and saved and presented != saved:
        return {
            "ok": False,
            "need_host_key_confirm": False,
            "error": (
                "主机密钥指纹与已保存的不一致，可能存在安全风险（如服务器重装或中间人攻击）。"
                "确认无误后可删除档案重建以重置指纹。"
            ),
        }
    return {"ok": False, "need_host_key_confirm": False, "error": msg}


@router.post("/remote/profiles/{profile_id}/test")
async def test_profile(profile_id: str, req: PasswordPayload):
    profile = _get_or_404(profile_id)
    try:
        return await asyncio.to_thread(_test_connection, profile, req.password or "")
    except Exception as e:  # noqa
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/remote/profiles/{profile_id}/confirm-host-key")
def confirm_host_key(profile_id: str, req: HostKeyConfirmPayload):
    profile = _get_or_404(profile_id)
    fingerprint = req.fingerprint.strip()
    if not fingerprint:
        raise HTTPException(status_code=400, detail="fingerprint 不能为空")
    profile.known_host_fingerprint = fingerprint
    get_profile_store().save(profile)
    return {"saved": True}


# ---- remote environment diagnostics --------------------------------------------


def _run_diagnostics(profile: RemoteProfile, password: str):
    """Blocking connect + diagnostics (runs in a worker thread)."""
    from anylabeling.services.training_center.remote.diagnostics import (
        run_remote_diagnostics,
    )
    from anylabeling.services.training_center.remote.ssh_service import (
        SSHConnectionService,
    )

    saved = (profile.known_host_fingerprint or "").strip()

    def _on_host_key(hostname, fingerprint, key_type):
        if saved and saved == fingerprint:
            return True, False
        return False, False

    ssh = SSHConnectionService()
    try:
        ok, msg = ssh.connect(profile, password=password, on_host_key=_on_host_key)
        if not ok:
            raise RuntimeError(f"SSH 连接失败: {msg}")
        items = run_remote_diagnostics(profile, ssh, password=password)
    finally:
        ssh.close()

    gpus = []
    cuda_available = False
    for item in items:
        if item.stage == "pytorch" and isinstance(item.details, dict):
            cuda_available = bool(item.details.get("cuda_available"))
            gpus = item.details.get("gpus") or []
    recommended = "0" if (cuda_available and gpus) else "cpu"
    return {
        "items": [
            {
                "stage": i.stage,
                "label": i.label,
                "status": str(getattr(i.status, "value", i.status)),
                "message": i.message,
                "details": i.details,
            }
            for i in items
        ],
        "recommended_device": recommended,
        "gpus": gpus,
    }


@router.post("/remote/profiles/{profile_id}/diagnostics")
async def profile_diagnostics(profile_id: str, req: PasswordPayload):
    profile = _get_or_404(profile_id)
    try:
        return await asyncio.to_thread(_run_diagnostics, profile, req.password or "")
    except Exception as e:  # noqa
        raise HTTPException(status_code=400, detail=str(e))
