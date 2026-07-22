"""Auto-labeling model registry (baseline snapshot + UI category exports).

This file provides the model categorization lists required by
the labeling view UI (batch processing, conf/iou settings, etc.)
while restricting to the baseline model set.

Only models that have corresponding service implementations and
config files in configs/auto_labeling/ are listed here.
"""

# ── Custom models (have dedicated service implementations) ──────────────
_CUSTOM_MODELS = [
    "segment_anything",
    "segment_anything_3",
    "yolov5",
    "yolov6",
    "yolov7",
    "yolov8",
    "yolox",
    "yolov5_cls",
    "yolov5_resnet",
    "yolov6_face",
    "rtdetr",
]


# ── set_cache_auto_label ────────────────────────────────────────────────
_CACHED_AUTO_LABELING_MODELS = [
    # None of the baseline models require special caching
]


# ── set_auto_labeling_marks ─────────────────────────────────────────────
_AUTO_LABELING_MARKS_MODELS = [
    "segment_anything",
    "segment_anything_3",
]


# ── set_mask_fineness ───────────────────────────────────────────────────
_AUTO_LABELING_MASK_FINENESS_MODELS = [
    "segment_anything",
]


# ── set_cropping_mode ───────────────────────────────────────────────────
_AUTO_LABELING_CROPPING_MODE_MODELS = [
    "segment_anything",
]


# ── skip detection step ─────────────────────────────────────────────────
_SKIP_DET_MODELS = []


# ── skip_prediction_on_new_marks ────────────────────────────────────────
_SKIP_PREDICTION_ON_NEW_MARKS_MODELS = []


# ── set_auto_labeling_api_token ─────────────────────────────────────────
_AUTO_LABELING_API_TOKEN_MODELS = []


# ── set_auto_labeling_reset_tracker ─────────────────────────────────────
_AUTO_LABELING_RESET_TRACKER_MODELS = []


# ── set_auto_labeling_conf ──────────────────────────────────────────────
_AUTO_LABELING_CONF_MODELS = [
    "rtdetr",
    "yolov5",
    "yolov6",
    "yolov6_face",
    "yolov7",
    "yolov8",
    "yolox",
]


# ── set_auto_labeling_iou ───────────────────────────────────────────────
_AUTO_LABELING_IOU_MODELS = [
    "yolov5",
    "yolov6",
    "yolov7",
    "yolov8",
    "yolox",
]


# ── set_auto_labeling_preserve_existing_annotations_state ───────────────
_AUTO_LABELING_PRESERVE_EXISTING_ANNOTATIONS_STATE_MODELS = [
    "rtdetr",
    "yolov5",
    "yolov6",
    "yolov6_face",
    "yolov7",
    "yolov8",
    "yolox",
]


# ── set_auto_labeling_prompt ────────────────────────────────────────────
_AUTO_LABELING_PROMPT_MODELS = []


# ── on_next_files_changed ───────────────────────────────────────────────
_ON_NEXT_FILES_CHANGED_MODELS = [
    "segment_anything",
]


# ── update_thumbnail_display ────────────────────────────────────────────
_THUMBNAIL_RENDER_MODELS = {}


# ── batch_processing_invalid_models ─────────────────────────────────────
_BATCH_PROCESSING_INVALID_MODELS = [
    "segment_anything",
]


# ── batch_processing_text_prompt_models ─────────────────────────────────
_BATCH_PROCESSING_TEXT_PROMPT_MODELS = []


# ── batch_processing_video_models ───────────────────────────────────────
_BATCH_PROCESSING_VIDEO_MODELS = []
