"""Persistent storage for RemoteProfile (JSON file)."""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import RemoteProfile, AuthMethod

_STORE_DIR = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))) / "TrainLens"
_STORE_PATH = _STORE_DIR / "remote_profiles.json"


class ProfileStore:
    """Thread-safe JSON file store for RemoteProfile.

    Passwords are NEVER written to disk.
    """

    def __init__(self, path: Optional[Path] = None):
        self._path = path or _STORE_PATH
        self._lock = threading.Lock()
        self._cache: Dict[str, RemoteProfile] = {}
        self._loaded = False

    @property
    def _file(self) -> Path:
        return self._path

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._file.parent.mkdir(parents=True, exist_ok=True)
        if not self._file.is_file():
            self._cache = {}
            self._loaded = True
            return
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                items = json.load(f)
            self._cache = {}
            for item in items:
                profile = RemoteProfile.from_dict(item)
                self._cache[profile.profile_id] = profile
        except (json.JSONDecodeError, OSError, TypeError):
            self._cache = {}
        self._loaded = True

    def _flush(self):
        """Write all profiles to disk (caller must hold _lock)."""
        items = [p.to_dict() for p in self._cache.values()]
        self._file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        tmp.replace(self._file)

    def list_all(self) -> List[RemoteProfile]:
        with self._lock:
            self._ensure_loaded()
            return list(self._cache.values())

    def get(self, profile_id: str) -> Optional[RemoteProfile]:
        with self._lock:
            self._ensure_loaded()
            return self._cache.get(profile_id)

    def save(self, profile: RemoteProfile):
        """Save or update a profile. Password is NOT persisted."""
        now = datetime.now().isoformat()
        if not profile.created_at:
            profile.created_at = now
        profile.updated_at = now
        with self._lock:
            self._ensure_loaded()
            self._cache[profile.profile_id] = profile
            self._flush()

    def delete(self, profile_id: str):
        with self._lock:
            self._ensure_loaded()
            self._cache.pop(profile_id, None)
            self._flush()


_store: Optional[ProfileStore] = None


def get_profile_store(path: Optional[Path] = None) -> ProfileStore:
    global _store
    if _store is None:
        _store = ProfileStore(path)
    return _store
