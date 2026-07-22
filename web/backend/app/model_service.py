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
        return {
            "name": cfg.get("name"),
            "display_name": cfg.get("display_name"),
            "type": cfg.get("type"),
            "config_file": cfg.get("config_file"),
            "supports_marks": cfg.get("type") in _AUTO_LABELING_MARKS_MODELS,
        }

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
        if not self.loaded_info():
            raise RuntimeError("No model loaded")
        if conf is not None:
            self.manager.set_auto_labeling_conf(conf)
        if iou is not None:
            self.manager.set_auto_labeling_iou(iou)

        # `image` is only used as a not-None sentinel: the model loads the
        # pixels from `filename` itself (see qt_img_to_rgb_cv_img).
        result = self.manager.predict_shapes(
            image=True,
            filename=image_path,
            text_prompt=text_prompt if text_prompt else None,
            batch=True,
        )
        if result is None:
            raise RuntimeError(self.status_message or "Prediction failed")
        shapes = [s.to_dict() for s in result.shapes]
        return {
            "shapes": shapes,
            "replace": result.replace,
            "description": result.description or "",
        }


# lazy singleton (ModelManager() touches user config at construction time)
_service: Optional[WebModelService] = None


def get_model_service() -> WebModelService:
    global _service
    if _service is None:
        _service = WebModelService()
    return _service
