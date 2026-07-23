"""Thin web-friendly wrapper around the desktop ModelManager.

The desktop ModelManager drives model loading/inference through QThreads
and Qt signals. In the web backend we call the synchronous internals
(`_load_model`, `predict_shapes(batch=True)`) directly inside a thread
pool, and surface progress through plain attributes that the API layer
can poll.
"""

import os.path as osp
import threading
from typing import Any, Dict, List, Optional

import anylabeling.config as anylabeling_config
from anylabeling.config import get_work_directory
from anylabeling.services.auto_labeling import _AUTO_LABELING_MARKS_MODELS
from anylabeling.services.auto_labeling.model_manager import ModelManager
from anylabeling.services.auto_labeling.types import AutoLabelingResult


def _ensure_user_config():
    """Mirror what the desktop app does at startup (app.py)."""
    if anylabeling_config.current_config_file is None:
        rc_file = osp.join(get_work_directory(), ".xanylabelingrc")
        if not osp.exists(rc_file):
            with open(rc_file, "w", encoding="utf-8") as f:
                f.write("{}\n")
        anylabeling_config.current_config_file = rc_file


class WebModelService:
    def __init__(self):
        _ensure_user_config()
        self.manager = ModelManager()
        self.lock = threading.RLock()

        # load state
        self.loading = False
        self.load_error: Optional[str] = None
        self.progress: Optional[Dict[str, int]] = None  # downloaded/total
        self.status_message: str = ""

        # predict state
        self.predicting = False
        self.batch_state: Optional[Dict[str, Any]] = None

        # bridge Qt signals -> plain attributes (direct connection: the
        # emit happens in the same thread that runs _load_model). Some
        # signals don't exist on reduced ModelManager variants.
        if hasattr(self.manager, "download_progress"):
            self.manager.download_progress.connect(self._on_download_progress)
        if hasattr(self.manager, "new_model_status"):
            self.manager.new_model_status.connect(self._on_status_message)

    # ---- signal bridges -----------------------------------------------------
    def _on_download_progress(self, downloaded: int, total: int):
        with self.lock:
            self.progress = {"downloaded": downloaded, "total": total}

    def _on_status_message(self, message: str):
        with self.lock:
            self.status_message = message

    # ---- model catalog --------------------------------------------------------
    def list_models(self) -> List[Dict[str, Any]]:
        configs = self.manager.get_model_configs()
        return [
            {
                "name": c.get("name"),
                "display_name": c.get("display_name"),
                "type": c.get("type"),
                "provider": c.get("provider"),
                "config_file": c.get("config_file"),
                "is_custom_model": c.get("is_custom_model", False),
            }
            for c in configs
        ]

    def loaded_info(self) -> Optional[Dict[str, Any]]:
        cfg = self.manager.loaded_model_config
        if not cfg or not cfg.get("model"):
            return None
        meta = getattr(cfg["model"], "Meta", None)
        output_modes = getattr(meta, "output_modes", {}) if meta else {}
        return {
            "name": cfg.get("name"),
            "display_name": cfg.get("display_name"),
            "type": cfg.get("type"),
            "config_file": cfg.get("config_file"),
            "supports_marks": cfg.get("type") in _AUTO_LABELING_MARKS_MODELS,
            "output_modes": list(output_modes.keys()),
            "default_output_mode": getattr(meta, "default_output_mode", "rectangle")
            if meta
            else "rectangle",
            "widgets": list(getattr(meta, "widgets", [])) if meta else [],
        }

    def set_output_mode(self, mode: str) -> Dict[str, Any]:
        cfg = self.manager.loaded_model_config
        if not cfg or not cfg.get("model"):
            raise RuntimeError("No model loaded")
        self.manager.set_output_mode(mode)
        return {"output_mode": mode}

    # ---- load / unload --------------------------------------------------------
    def load(self, config_file: str):
        """Blocking load (download if needed). Run in a worker thread."""
        with self.lock:
            if self.loading:
                raise RuntimeError("Another model is being loaded")
            self.loading = True
            self.load_error = None
            self.progress = None
            self.status_message = ""
        try:
            model_id = None
            for i, cfg in enumerate(self.manager.model_configs):
                if cfg.get("config_file") == config_file:
                    model_id = i
                    break
            if model_id is None:
                raise ValueError(f"Unknown model config: {config_file}")
            self.manager._load_model(model_id)
            if not self.loaded_info():
                raise RuntimeError(self.status_message or "Model load failed")
        except Exception as e:  # noqa
            with self.lock:
                self.load_error = str(e)
            raise
        finally:
            with self.lock:
                self.loading = False

    def unload(self):
        self.manager.unload_model()

    # ---- inference ------------------------------------------------------------
    def predict(
        self,
        image_path: str,
        text_prompt: Optional[str] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Blocking prediction. Run in a worker thread."""
        cfg = self.manager.loaded_model_config
        if not cfg or not cfg.get("model"):
            raise RuntimeError("No model loaded")
        model = cfg["model"]
        if conf is not None and hasattr(model, "set_auto_labeling_conf"):
            model.set_auto_labeling_conf(conf)
        if iou is not None and hasattr(model, "set_auto_labeling_iou"):
            model.set_auto_labeling_iou(iou)

        # `image` is only used as a not-None sentinel: the model loads the
        # pixels from `filename` itself (see qt_img_to_rgb_cv_img).
        if text_prompt:
            result = model.predict_shapes(True, image_path, text_prompt=text_prompt)
        else:
            result = model.predict_shapes(True, image_path)

        if result is None:
            raise RuntimeError(self.status_message or "Prediction failed")
        if isinstance(result, AutoLabelingResult):
            return {
                "shapes": [s.to_dict() for s in result.shapes],
                "replace": result.replace,
                "description": getattr(result, "description", "") or "",
            }
        # some models return a bare shape list
        return {
            "shapes": [s.to_dict() for s in result],
            "replace": False,
            "description": "",
        }


# lazy singleton (ModelManager() touches user config at construction time)
_service: Optional[WebModelService] = None


def get_model_service() -> WebModelService:
    global _service
    if _service is None:
        _service = WebModelService()
    return _service
