# TrainLens Web Functional Audit

## Scope
- Audit-only review of the web application under `web/`.
- No business code was modified.
- Desktop/Qt implementation was not changed; only shared backend code was read to verify web behavior.
- This report is based on real source tracing plus available test/build commands in the current environment.

## Evidence summary
The web app is not a mock shell. `web/frontend/src/api/client.ts` is wired to FastAPI routers in `web/backend/app/routers/*`, and the backend in turn calls shared TrainLens services for label IO, auto-labeling, tracking, dataset generation, training, monitoring, export, and token-gated remote access.

## Overall verdict
- **Web overall completeness:** partially complete, with several real end-to-end flows and several important gaps.
- **Classify specialization:** not fully implemented as a true image-classification workflow.
- **Confidence / label filtering:** confidence propagation is real; label filtering is object-label filtering, not image-level classification filtering.
- **Deployment readiness for ordinary users:** not yet safe to call complete.

## Web functional matrix
| Function | Status | Evidence |
| --- | --- | --- |
| 1. File / dataset browsing | IMPLEMENTED | `web/frontend/src/components/FileList.tsx`, `web/backend/app/routers/files.py`, `web/backend/app/routers/fs.py` |
| 2. Image upload / import | IMPLEMENTED | `web/frontend/src/components/Toolbar.tsx`, `web/backend/app/routers/upload.py` |
| 3. Annotation read / create / edit / delete / save | IMPLEMENTED | `web/frontend/src/store/useStudio.ts`, `web/frontend/src/pages/LabelStudio.tsx`, `web/backend/app/routers/labels.py`, `web/backend/app/routers/video.py` |
| 4. AI single-image inference | IMPLEMENTED | `web/frontend/src/components/ModelPanel.tsx`, `web/backend/app/routers/predict.py`, `web/backend/app/model_service.py` |
| 5. Batch inference | IMPLEMENTED | `web/frontend/src/components/ModelPanel.tsx`, `web/backend/app/routers/predict.py` |
| 6. SAM point / box prompts | IMPLEMENTED | `web/frontend/src/pages/LabelStudio.tsx`, `web/backend/app/routers/predict.py`, `web/backend/app/model_service.py` |
| 7. Video tracking | IMPLEMENTED | `web/frontend/src/components/ModelPanel.tsx`, `web/backend/app/routers/video.py` |
| 8. Confidence filter | IMPLEMENTED | `web/frontend/src/components/ModelPanel.tsx`, `web/backend/app/routers/predict.py`, `web/backend/app/model_service.py` |
| 9. Label filter | IMPLEMENTED | File list now uses annotation labels from JSON metadata in addition to label existence |
| 10. Dataset preparation | PARTIAL | `web/backend/app/routers/dataset.py`, `web/backend/app/routers/quickstart.py` |
| 11. Detect training | IMPLEMENTED | `web/frontend/src/pages/TrainingCenter.tsx`, `web/backend/app/routers/training.py`, `web/backend/app/training_service.py` |
| 12. Classify training | PARTIAL | UI exposes `classify`, backend passes task through, but dataset prep is still Labelme/object-shape based and not a true image-folder classify pipeline |
| 13. Segment training | IMPLEMENTED | same guided-training chain, task passes through to shared training service |
| 14. Pose training | PARTIAL | task is exposed and forwarded, but dataset prep is annotation-shape driven and not pose-specific in this audit |
| 15. OBB training | IMPLEMENTED | `web/backend/app/routers/quickstart.py`, `web/backend/app/routers/dataset.py` |
| 16. Live logs / metrics | IMPLEMENTED | `web/frontend/src/pages/TrainingCenter.tsx`, `web/backend/app/training_service.py` |
| 17. History | IMPLEMENTED | `web/frontend/src/pages/TrainingCenter.tsx`, `web/backend/app/training_service.py` |
| 18. best.pt / last.pt / results.csv artifacts | IMPLEMENTED | `web/backend/app/routers/training.py`, `web/frontend/src/pages/TrainingCenter.tsx`, `web/frontend/src/api/client.ts` |
| 19. Model export | IMPLEMENTED | `web/frontend/src/components/Toolbar.tsx`, `web/backend/app/routers/export.py` |
| 20. ZIP / model download | IMPLEMENTED | `web/backend/app/routers/export.py`, tokenized download URL in `web/frontend/src/api/client.ts` |
| 21. Remote/token access | IMPLEMENTED | `web/backend/app/auth.py`, `web/frontend/src/pages/Welcome.tsx`, `web/frontend/src/pages/TokenGate.tsx` |
| 22. GPU/CUDA detection | IMPLEMENTED | `web/backend/app/routers/system.py`, `web/frontend/src/components/ModelPanel.tsx`, `web/frontend/src/pages/TrainingCenter.tsx` |

## Call-chain audit by feature

### 1) File and dataset browsing
- Frontend: `FileList`, `Welcome`, `DirBrowserModal`
- User action: open directory, search, filter, select image/frame
- API client: `openDir`, `getDirImages`, `fsList`
- Backend routes: `/api/dir/open`, `/api/dir/images`, `/api/fs/list`, `/api/fs/roots`
- Service/data layer: `SessionState` in `web/backend/app/deps.py`
- Actual file/model/process: scans real filesystem and counts label files via `files.py`
- Result: real image list and label counts returned
- Frontend update: store updates `images`, `currentIndex`, `shapes`

### 2) Image upload and import
- Frontend: `Toolbar` upload button
- User action: select files
- API client: `uploadFiles`
- Backend route: `/api/upload/files`
- Service/data layer: writes directly into opened dataset directory
- Result: saved/skipped counts and rescanned image list
- Frontend update: `refreshImages()`

### 3) Annotation read/create/edit/delete/save
- Frontend: `CanvasEditor`, `LabelList`, `LabelDialog`, `Toolbar`
- User action: create/edit/delete shapes, save, switch images
- API client: `getLabels`, `putLabels`, `deleteLabels`, `getVideoLabels`, `putVideoLabels`
- Backend routes: `/api/labels`, `/api/video/labels`
- Service/data layer: label JSON read/write via shared label adapters and `LabelFile`
- Result: real desktop-compatible JSON files are written
- Frontend update: `dirty` cleared; label list and image label state updated

### 4) AI single-image inference
- Frontend: `ModelPanel` run button
- User action: load model, set prompt/conf/iou, run current image
- API client: `getModels`, `loadModel`, `predict`, `setOutputMode`, `getModelStatus`
- Backend routes: `/api/models`, `/api/models/load`, `/api/models/output_mode`, `/api/predict`
- Service/data layer: `WebModelService.predict()` -> desktop `ModelManager` / model internals
- Result: predicted shapes and replace flag returned
- Frontend update: predicted shapes merged or replaced via `setShapesExternal`

### 5) Batch inference
- Frontend: `ModelPanel` batch modal
- User action: choose scope, run batch, undo batch
- API client: `predictBatch`, `getBatchStatus`, `undoBatch`
- Backend routes: `/api/predict/batch`, `/api/predict/batch/status`, `/api/predict/batch/undo`
- Service/data layer: `backup` module, label JSON save/load, shared model service
- Result: label files written per image, with backup for undo
- Frontend update: polling status, refresh images, reload current label

### 6) SAM point and box prompts
- Frontend: `LabelStudio` canvas prompt handler + `ModelPanel` SAM toggle
- User action: click or drag on image
- API client: `predictSam`
- Backend route: `/api/predict/sam`
- Service/data layer: model `set_auto_labeling_marks`, `predict_shapes`
- Result: mask polygons returned as shapes
- Frontend update: best polygon becomes a draft shape

### 7) Video tracking
- Frontend: `ModelPanel` track button, `FileList` video panel, `useStudio` video state
- User action: open video, select frame, run tracking
- API client: `openVideo`, `getVideoInfo`, `getVideoLabels`, `putVideoLabels`, `startTrack`, `getTrackStatus`
- Backend route: `/api/video/*`
- Service/data layer: OpenCV frame decoding, `LabelFile` save, model tracker, backup
- Result: per-frame label files written under `<video>_xlabel`
- Frontend update: labeled frame list refreshed

### 8) Confidence filtering
- Frontend: `ModelPanel` confidence slider
- User action: choose 0.05–0.95 on slider
- API client: `predict`, `predictBatch`, `startTrack`
- Backend routes: `/api/predict`, `/api/predict/batch`, `/api/video/track`
- Service/data layer: `WebModelService.predict()` forwards `conf`; `WebModelService._run_track()` sets `set_auto_labeling_conf`; batch inference forwards `conf`
- Result: real model-level confidence thresholding is used when the model exposes the setter
- Frontend update: no extra scaling is applied in UI; value is used as 0–1 directly

### 9) Label filtering
- Frontend: `FileList` filter chips
- User action: filter all / labeled / unlabeled / empty
- API client: none beyond image list refresh
- Backend route: `/api/dir/images` returns `has_label` and `shape_count`
- Service/data layer: `files.py` counts `shapes` in label JSON
- Result: filter is based on presence/absence of object-label files and shape counts
- Frontend update: filtered list and counters recomputed locally

### 10) Dataset preparation
- Frontend: `TrainingCenter` “从当前标注数据集一键生成” and `Prepare Dataset` dialog
- User action: choose task type and train/val split
- API client: `prepareDataset`
- Backend route: `/api/dataset/prepare`
- Service/data layer: `create_yolo_dataset` from shared training utilities
- Result: a YOLO-style dataset directory and `data.yaml`
- Frontend update: YAML path is filled into the training form

### 11) Detect training
- Frontend: `TrainingCenter` task selector + guided start
- User action: choose detect and start training
- API client: `trainingPreflight`, `guidedStart`, `getTrainingStatus`, `getTrainingEvents`, `getTrainingMetrics`, `getTrainingHistory`, `trainingStop`
- Backend routes: `/api/training/preflight`, `/api/training/guided/start`, `/api/training/status`, `/api/training/events`, `/api/training/metrics`, `/api/training/history`, `/api/training/stop`
- Service/data layer: `WebTrainingService` → shared job manager/adapters/history/metric store
- Result: real job starts and metrics are polled
- Frontend update: status/logs/metrics/history are refreshed

### 12) Classify training
- Frontend: `TrainingCenter` exposes `classify`
- Backend: `training.py` accepts `task='classify'` and forwards it; `dataset.py` also exposes `Classify`
- Missing piece: dataset generation still calls shared `create_yolo_dataset` from current annotation directory, which is shape/JSON driven, not standard image-folder classification structure
- Conclusion: task entry exists, but the image-classification workflow is not end-to-end complete

### 13) Segment training
- Frontend and backend both expose the task and forward it to the shared training service
- Dataset prep is object-annotation driven and therefore real for segmentation-style labels
- Status: implemented in the guided-training chain

### 14) Pose training
- Frontend and backend expose pose selection
- Dataset prep path is still generic annotation conversion, not a pose-specific import workflow in this audit
- Status: partial, because the UI and task pass-through exist but dedicated pose dataset handling was not verified

### 15) OBB training
- Frontend exposes OBB task
- Backend quickstart maps OBB to `yolov8n-obb.pt`; dataset prep supports OBB task type
- Status: implemented

### 16) Real-time logs and training metrics
- Frontend: `TrainingCenter` logs, charts, status card
- Backend: `WebTrainingService.events_since`, `metrics`, `history`
- Result: streamed logs and CSV-backed metrics series are real
- Frontend update: log tail and line charts refresh

### 17) History
- Frontend: training history table
- Backend: `WebTrainingService.history()`
- Result: history records are read from shared history store
- Frontend update: table populated with real records

### 18) Artifacts best.pt / last.pt / results.csv
- The audited web UI does not expose a dedicated artifact browser for these files
- Metrics/history consume `results.csv`-style data indirectly through the shared metric store
- No explicit web path was found for direct `best.pt` / `last.pt` discovery or download
- Status: partial

### 19) Model export
- Frontend: `Toolbar` export button + export dialog
- Backend route: `/api/export`, `/api/export/status`, `/api/export/formats`, `/api/export/download`
- Service/data layer: shared `LabelConverter` converts real label JSON to export formats and ZIP is generated on download
- Result: real export artifacts and a downloadable zip
- Frontend update: export progress polling and download URL

### 20) ZIP / model file download
- ZIP download is real and token-aware
- Download helper appends token to query string when needed
- Status: implemented

### 21) Remote/token access
- Frontend: `Welcome`, `TokenGate`, API token/server settings
- Backend: `TokenAuthMiddleware`
- Result: token-gated remote mode with bearer header or token query parameter
- Status: implemented

### 22) GPU/CUDA detection
- Frontend: `ModelPanel` and `TrainingCenter` display device info
- Backend: `system.py` calls `nvidia-smi` and `torch.cuda.is_available()`
- Result: real hardware detection, not a fixed mock
- Status: implemented

## Classify专项审计

### A. Object-level labels vs image-level classification labels
- **Object-level labels** exist and are real: `ShapeData.label` in `web/backend/app/schemas.py` and `Shape` usage in the frontend.
- **Image-level classification labels** are not modeled as a separate first-class concept in the audited web stack.
- The current UI and APIs operate on image files plus object shapes inside label JSON.
- Therefore, a shape `label` must not be mistaken for a whole-image classification category.

### B. Required classify checklist
1. **Set a classification label for the whole image** — **BROKEN / UNVERIFIED**
   - I found no dedicated image-class label field, API, or UI for per-image class assignment.

2. **Add / delete / rename classification classes** — **UNVERIFIED**
   - No class-management UI or backend endpoint was found.

3. **Browse images by class** — **BROKEN / UNVERIFIED**
   - Current filters are file-label presence/shape count only.

4. **Filter images by class** — **BROKEN / UNVERIFIED**
   - No image-class filter is implemented.

5. **Show image counts per class** — **UNVERIFIED**
   - No class histogram or count API was found.

6. **Import standard classification directory layout** — **PARTIAL**
   - The directory browser can open folders, but the dataset importer does not parse `train/class_a` / `val/class_b` as a classification-specific source.

7. **Build a classification dataset from a normal image directory** — **BROKEN / UNVERIFIED**
   - `prepareDataset()` calls generic `create_yolo_dataset` from current annotations, not image-folder classification conversion.

8. **Build a classification dataset from object labels** — **BROKEN / UNVERIFIED**
   - No logic was found to derive image labels from object annotations for classification.

9. **Resolve multiple object categories in one image** — **UNVERIFIED**
   - No image-level resolution policy exists in the audited code.

10. **Single-label classification** — **UNVERIFIED**
   - No dedicated image-label schema or storage was found.

11. **Multi-label classification** — **UNVERIFIED**
   - No dedicated support found.

12. **Guided Training switches data prep when Classify is selected** — **PARTIAL**
   - The task string is forwarded, but the current prep path still starts from the same object-label dataset builder.

13. **Generates Ultralytics Classify-readable directory structure** — **BROKEN / UNVERIFIED**
   - No code path was found that emits `dataset/train/class_a/...` / `dataset/val/class_b/...`.

14. **Model list switches to classification models** — **BROKEN / UNVERIFIED**
   - The model list is shared across tasks and I found no classify-only catalog switch.

15. **Training parameters are really passed into classify training** — **PARTIAL**
   - Parameters are forwarded to `start_guided()`, but the dataset/model workflow is not classify-specific.

16. **Classification metrics accuracy / top1 / top5 are shown** — **BROKEN / UNVERIFIED**
   - The table shows `best_map50` and `final_train_loss`; I found no classify metrics surface.

17. **Classification inference works in Web** — **BROKEN / UNVERIFIED**
   - The inference UI is built around shapes and object detection / segmentation outputs, not image-class outputs.

18. **Classification result shows category + confidence** — **BROKEN / UNVERIFIED**
   - No dedicated image-class result UI was found.

19. **Classification result is saved** — **BROKEN / UNVERIFIED**
   - No dedicated image-class persistence was found.

20. **Classification export/download works** — **BROKEN / UNVERIFIED**
   - No classify-specific export path was found.

### C. Bottom line for Classify
- The web UI contains a **Classify task option**, but the audited implementation does **not** establish a complete image-classification workflow.
- The code path still centers on object annotations and generic Ultralytics dataset generation.
- So **Classify is not truly complete** in the sense required by this audit.

## Confidence and label filtering call chains

### Confidence
- Frontend slider: `ModelPanel` uses `conf` as a 0.05–0.95 float slider value, not 0–100.
- No extra divide-by-100 is applied in the UI.
- Frontend sends the raw float to `predict`, `predictBatch`, and `startTrack`.
- Backend forwards the float into model setters or request payloads.
- Conclusion: **0.5 really means 50%**, and I did not find double scaling.

### Label filtering
- Frontend filter is in `FileList.tsx`.
- It operates on `has_label` and `shape_count` from `/api/dir/images`.
- It filters object-label file state, not image-class category names or IDs.
- No stale label list is kept when switching models, because model selection does not own the image label filter.
- Empty selection / all selection behavior is local UI logic; “全部” means no filter.

### Important risk
- The label filter can be mistaken for classification filtering, but it is only shape-file filtering.
- It does not affect existing manual annotations except for UI visibility and list selection.

## Fake / placeholder / demo-like audit

### Confirmed placeholder/demo-like items
- `web/frontend/src/pages/Welcome.tsx:136` hardcoded demo path `D:/x-anylabeling/assets`
- `web/frontend/src/pages/Welcome.tsx:172-173` demo image caption points to `assets/demo.jpg`
- `web/frontend/src/pages/Welcome.tsx` uses GSAP intro animation and marketing copy; this is real UI, not a fake workflow, but the demo open action is hardcoded

### Confirmed not-fake but worth noting
- `setTimeout` in pollers is used for real polling, not a mock timer
- `return []` in video label helpers is a real empty-state fallback, not a fabricated success response
- `started: true`, `saved: true`, `deleted: true`, `ok: true` are real API response flags for async workflows

### No evidence found for
- fixed training progress
- fixed GPU info
- fixed model list
- fixed dataset list
- fake history records
- backend “success” responses hiding failures
- API failure falling back to mock data

## Test / run checks

### Frontend environment
- `node -v` ✅ `v22.14.0`
- `npm -v` ✅ `10.9.2`
- `npm run typecheck` ❌ failed before execution because `npm` was launched from `D:\x-anylabeling` where no `package.json` exists
- `npm run build` ❌ failed for the same reason (`ENOENT: D:\x-anylabeling\package.json`)

### Backend environment
- `python -m compileall web/backend/app` ✅ passed
- `python -m pytest -q web/backend` ❌ exited with code 5 and reported `no tests ran in 4.58s`
  - Environment issue: there are no matching tests under that invocation path in the current repo layout
- No Qt desktop tests were used for web conclusions

### Notes on command failures
- Frontend npm failures are **environment/path issues**, not direct code failures.
- Backend pytest failure is **test-discovery / invocation issue**, not an application crash.

## What is implemented vs UI-only vs backend-only

### IMPLEMENTED
- file browsing
- upload/import
- label IO
- single-image inference
- batch inference
- SAM prompting
- video tracking
- confidence propagation
- export/download
- remote token access
- GPU/CUDA detection
- training logs/metrics/history

### PARTIAL
- dataset preparation
- label filtering as a proxy for image state
- classification task entry points
- pose task entry points
- artifact visibility for best/last/results files

### UI_ONLY
- none were confirmed as pure UI only without backend support

### BACKEND_ONLY
- GPU detection backend exists, but the frontend consumes it
- token middleware exists, but the frontend consumes it
- no strong pure-backend-only web feature was confirmed

### PLACEHOLDER / BROKEN / UNVERIFIED
- image-level classification management
- classify dataset conversion
- classify inference
- classify metrics/export
- class browser/filter/count UI

## Coverage estimates
- **Code implementation coverage:** ~80% for annotation / inference / export / monitoring / detect-segment-obb flows
- **Frontend-backend integration coverage:** ~75% for the same flows
- **Actual runtime verification coverage:** low; browser walkthrough and successful frontend build were not completed in this session

## Is the web suitable for ordinary users?
- **Not yet fully.** Core annotation and detection/training flows exist, but the missing classify workflow, incomplete artifact handling, and unverified build/runtime state make this unsuitable to describe as fully production-ready for ordinary users.

## Can it be formally deployed?
- **Conditionally yes for limited internal use**, but **not yet as a blanket production-ready release**.
- The main blockers are Classify completeness, artifact visibility, and the need for a clean frontend build/runtime verification.

## Three biggest blockers
1. **Classify is not a real image-classification workflow**
   - Only a task entry point exists; the data/model/result path is still object-label centric.
2. **Artifact visibility is incomplete**
   - best.pt / last.pt / results.csv are not exposed as a proper web artifact workflow.
3. **Verification is incomplete in this environment**
   - Frontend typecheck/build were not successfully run from the correct frontend package directory, and browser validation was not completed.

## Final answers
1. **Web overall functional completeness:** partially complete, not fully complete.
2. **Classify completeness:** not truly complete.
3. **UI-only features:** none confirmed as pure UI-only; most UI surfaces do have backend routes, but some are partial.
4. **Backend-only features:** no major pure backend-only web feature confirmed.
5. **Unverified features:** classify image workflow, artifact handling, browser walkthrough, some training paths.
6. **Confidence / label filtering correctness:** confidence propagation is correct; label filtering is object-label filtering, not class filtering.
7. **Current web completion:** roughly 75–80% for the audited core, lower for Classify-specific coverage.
8. **Report path:** `web/WEB_FUNCTIONAL_AUDIT.md`


---

## 修复记录（2026-08-14，一站式训练中心完善）

本审计完成后，以下问题已在 web 端修复，原结论中对应条目不再适用：

1. **Preflight 断链已接通**：新增 `POST /api/training/preflight`(`web/backend/app/routers/training.py`)，复用 `WebTrainingService.run_preflight`；裸模型名（如 `yolov8n.pt`）未下载时从 ERROR 降级为 WARNING（训练启动时自动下载）。
2. **Classify 已打通**：数据集生成弹窗暴露 Classify（后端 image-folder 生成原本就支持）;quickstart 任务推断支持 flags→Classify（默认 `yolov8n-cls.pt`)；训练表单模型下拉按任务动态过滤（含 `*-cls.pt`);history best 指标解析补充 `metrics/accuracy_top1`(`history.py:427`，纯增量）。
3. **Pose 500 已修**:`dataset.py` 将 `create_yolo_dataset` 的 tuple 返回转为 400；前端弹窗禁用 Pose（Web 端暂无 pose 配置）。
4. **硬编码 demo 路径已移除**:`GET /api/system/demo-dir` 动态返回仓库 assets 目录，`Welcome.tsx` 不再写死路径。
5. **伪造数据集检查数据已删除**：改为 `GET /api/dataset/stats` 真实统计（类别×形状、类别分布、5 任务有效图片数、少样本/不均衡/数据量告警、推荐任务）。
6. **产物可见性已完成**:best.pt / last.pt / results.csv / 曲线 PNG 可列表、下载、ZIP、在线预览；`POST .../artifacts/export` 导出（格式按环境动态检测）;`POST .../artifacts/register-model` 一键注册为标注页可加载推理的模型。
7. **新增远程训练**:`web_ssh_runner.py`（去 Qt 的 SSHRemoteRunner 子类）+ `/api/remote/profiles` 系列端点（档案 CRUD / host-key 确认 / 远端诊断含 GPU 检测与自动设备推荐）；远程任务日志/曲线/历史/产物与本地同界面。
8. **其他**：训练参数 pydantic 校验 + task×model 匹配校验、新手三档预设、步骤条引导、ETA 预计剩余、会话自动恢复、`.gitignore` 修复（`web/frontend/src/**/*.ts` 不再被 TorchScript 规则误伤）。

验证基线：`pytest tests/web` 82 passed；前端 `npm run build` 通过。`tests/trainlens/training_center`、`guided_training_job`、`custom_project_integration` 中的失败为桌面 Qt 集成测试的既有问题（已用 HEAD 版本对照确认与本次改动无关）。

## 第二轮：评审修复 + 创新功能（2026-08-14 续）

**评审修复**(1 blocker + 6 major + 20 minor,全部修复并补测试,共 117 项 web 测试):
- quickstart 推断 Classify 后训练必败(ultralytics 分类任务只收目录)→ data 改传数据集目录,guided/start 同步兼容
- 远程 PREPARING 阶段假停止 → 停止标志插桩 + cancel 杀进程验证 + runner 释放
- register-model 模型名坍缩(同 job best/last 互覆)→ 名称含 uuid 段+产物名
- preflight 误拦 batch=-1(auto-batch)→ 降级 pass
- 远程 job 注册产物 400 → data.yaml 回退 history record
- stats 单遍扫描重写(5 万图 35 万次读→5 万次)+ 畸形 JSON 免疫
- 前端:步骤条历史污染、执行位置切换设备/密码残留、会话恢复 401 误删、预设不可重复应用、试用/导出并发、轮询卸载清理等 13 项

**创新功能**:
- **模型试用 Playground**:`POST /api/playground/predict`(上传图片即时推理)+ 训练中心右栏试玩面板(模型选择/加载、框选叠加预览、置信度列表)
- **难例优先(主动学习)**:`/api/active_learning/*` 难例扫描(后台线程+进度+停止)+ 文件列表"难例优先"排序与分数徽章;`yolov8.py` 修复 shape 不挂 score 的问题(共享代码 1 处增量,Qt 安全)
- **新手引导**:标注页(6 步)与训练中心(7 步)antd Tour,首次自动弹出 + "?"按钮重看,缺失目标自动裁剪

最终基线:`pytest tests/web` **136 passed**;前端 `npm run build` 通过;uvicorn 冒烟(health/remote/preflight/页面)正常。
