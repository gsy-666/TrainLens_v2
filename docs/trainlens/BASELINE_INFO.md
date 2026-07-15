# X-AnyLabeling Baseline Information

**Generated**: 2026-07-15  
**Purpose**: Record the baseline state before TrainLens integration

## Repository Information

- **Root Directory**: `/d/x-anylabeling`
- **Current Branch**: `main`
- **Current Commit**: `36b063dde798f1a198bb90a88e2e6e59f139ba78`
- **Remote**: `https://github.com/cvhub520/x-anylabeling`
- **Working Tree**: Clean (no uncommitted changes)
- **Baseline Tag**: `baseline-x-anylabeling-before-trainlens`

## Application Information

- **Application Name**: X-AnyLabeling
- **Version**: 4.0.0-beta.13
- **Description**: Advanced Auto Labeling Solution with Added Features
- **License**: GPLv3
- **Python Required**: >=3.11
- **UI Framework**: PyQt6 >= 6.6.0

## Entry Points

- **Main Entry**: `anylabeling/app.py::main()`
- **CLI Command**: `xanylabeling` (defined in `pyproject.toml` as `anylabeling.app:main`)
- **Main Window Class**: `anylabeling/views/mainwindow.py::MainWindow`
- **Labeling Widget**: `anylabeling/views/labeling/label_wrapper.py::LabelingWrapper`
- **Core Widget**: `anylabeling/views/labeling/label_widget.py::LabelingWidget` (7118 lines)

## Core Components (MUST NOT BE DELETED OR REPLACED)

### 1. Canvas and Shape System
- **Canvas**: `anylabeling/views/labeling/widgets/canvas.py::Canvas` (5259 lines)
  - Handles all drawing, selection, editing, and rendering
  - Supports multiple shape types: rectangle, polygon, circle, line, point, rotated box, etc.
- **Shape**: `anylabeling/views/labeling/shape.py::Shape` (824 lines)
  - Shape data model with vertices, labels, attributes, groups, locks

### 2. Label File I/O
- **LabelFile**: `anylabeling/views/labeling/label_file.py::LabelFile`
  - Read/write JSON, XML formats
  - Supports image data embedding

### 3. Auto-Labeling Services
- **Location**: `anylabeling/services/auto_labeling/`
- **Model Manager**: `anylabeling/services/auto_labeling/model_manager.py`
- **Models**: 50+ model implementations (YOLO, SAM, Grounding-DINO, OCR, etc.)
- **Base Classes**: 
  - `__base__/yolo.py`
  - `__base__/sam.py`
  - `__base__/sam2.py`
  - `__base__/sam3.py`
  - `__base__/grounding_dino.py`

### 4. Existing Training System (MUST KEEP)
- **Ultralytics Dialog**: `anylabeling/views/training/ultralytics_dialog.py::UltralyticsDialog`
- **Training Manager**: `anylabeling/services/auto_training/ultralytics/trainer.py::TrainingManager`
- **Export Manager**: `anylabeling/services/auto_training/ultralytics/exporter.py::ExportManager`
- **Menu Entry**: LabelingWidget creates "Train" menu at line 2012
- **Menu Action**: `ultralytics_train` action triggers UltralyticsDialog

### 5. Format Converters
- **Location**: `anylabeling/views/common/converter.py`
- **CLI**: `xanylabeling convert --task <task>`
- **Supported**: YOLO, COCO, VOC, DOTA, MOT, MASK, PPOCR, etc.

## Dependencies

### Core Runtime Dependencies
- PyQt6 >= 6.6.0
- PyQt6-WebEngine >= 6.6.0
- opencv-contrib-python-headless >= 4.7.0.72
- numpy (version varies by extra: cpu uses >=2.0.0, gpu-cu11 uses <2.0.0)
- pillow >= 7.1.2
- matplotlib
- psutil
- scipy >= 1.4.1
- PyYAML
- requests
- openai

### Optional Dependencies (Extras)
- **cpu**: `onnx>=1.15.0`, `onnxruntime>=1.15.0`, `numpy>=2.0.0`
- **gpu**: `onnx>=1.15.0`, `onnxruntime-gpu>=1.18.1,<1.27.0`, `numpy>=2.0.0`
- **gpu-cu11**: `onnx>=1.15.0,<1.16.1`, `onnxruntime-gpu>=1.15.0,<1.19.0`, `numpy<2.0.0`
- **gpu-cu13**: `onnx>=1.15.0`, `onnxruntime-gpu>=1.27.0,<1.28.0`, `numpy>=2.0.0`
- **dev**: build, black, flake8, pyinstaller, pytest, PySide6, twine

### Conflicts
- cpu, gpu, gpu-cu11, gpu-cu13 are mutually exclusive (defined in `[tool.uv].conflicts`)

## Packaging and Distribution

- **Build System**: setuptools >= 70.0.0
- **Package Name**: `x-anylabeling-cvhub`
- **Installer**: PyInstaller (in dev dependencies)
- **Installation**:
  - CPU: `pip install -e ".[cpu,dev]"`
  - GPU (CUDA 12.x): `pip install -e ".[gpu,dev]"`
  - GPU (CUDA 11.x): `pip install -e ".[gpu-cu11,dev]"`
  - GPU (CUDA 13.x): `pip install -e ".[gpu-cu13,dev]"`

## Testing

- **Test Framework**: pytest
- **Test Location**: `tests/`
- **Test Suites**:
  - test_auto_labeling/
  - test_canvas/
  - test_config/
  - test_labeling/
  - test_models/
  - test_ppocr/
  - test_settings/
  - test_utils/
  - test_widgets/

## Configuration

- **Config Module**: `anylabeling/config.py`
- **Functions**: `get_config()`, `save_config()`, `get_work_directory()`, `set_work_directory()`
- **User Config**: Stored via QSettings

## Absolute Protection List

The following modules MUST NOT be deleted, replaced, or have breaking changes:

1. **Canvas system**: `anylabeling/views/labeling/widgets/canvas.py`
2. **Shape model**: `anylabeling/views/labeling/shape.py`
3. **LabelingWidget**: `anylabeling/views/labeling/label_widget.py`
4. **LabelFile I/O**: `anylabeling/views/labeling/label_file.py`
5. **Auto-labeling services**: `anylabeling/services/auto_labeling/*`
6. **Existing Ultralytics training**: 
   - `anylabeling/views/training/ultralytics_dialog.py`
   - `anylabeling/services/auto_training/ultralytics/*`
7. **Format converters**: `anylabeling/views/common/converter.py`
8. **All QAction definitions**: Menu items and shortcuts must remain functional
9. **Settings system**: `anylabeling/views/labeling/settings/*`
10. **All existing widgets**: `anylabeling/views/labeling/widgets/*`

## Integration Points for TrainLens Run Monitor

### Safe Integration Strategy

1. **New Module Location**: `anylabeling/services/run_monitor/`
   - All Run Monitor business logic goes here
   - Completely independent from existing services

2. **New View Location**: `anylabeling/views/run_monitor/`
   - All Run Monitor UI goes here
   - Separate window, not embedded in Canvas or LabelingWidget

3. **Menu Integration Point**: 
   - Add new "Run" menu in `LabelingWidget.menus` (around line 2012)
   - Add "Run Monitor" action
   - DO NOT replace or remove existing "Train" menu

4. **No Global Renames**:
   - Keep `from anylabeling...` imports
   - Keep package name `anylabeling`
   - Product branding is UI-only

5. **Process Isolation**:
   - Training processes run in subprocess
   - Use Qt signals for IPC
   - Training crashes must not crash main window

## Known Risks

1. **Large Core Files**: LabelingWidget (7118 lines) and Canvas (5259 lines) are very complex
   - Risk: Easy to break existing functionality
   - Mitigation: Do not modify these files unless absolutely necessary

2. **Menu System**: Menus created dynamically with utils.Struct
   - Risk: Menu entry conflicts
   - Mitigation: Add new "Run" menu, keep existing "Train" menu

3. **Settings and Config**: Multiple layers (QSettings, config.py, app_info.py)
   - Risk: Config conflicts
   - Mitigation: Use separate config namespace for Run Monitor

4. **Threading**: Existing training uses subprocess + QTimer + redirectors
   - Risk: Resource conflicts if both systems run simultaneously
   - Mitigation: Allow only one training task at a time across both systems

## Verification Checklist

Before considering TrainLens integration complete, verify:

- [ ] Original X-AnyLabeling GUI launches without errors
- [ ] All menu items present (File, Edit, View, Theme, Language, Upload, Export, Tool, Train, Help)
- [ ] Auto-labeling models load and run
- [ ] Canvas drawing and editing works
- [ ] Label file save/load works
- [ ] Format conversion works
- [ ] Original Ultralytics training dialog opens and functions
- [ ] All tests in `tests/` still pass
- [ ] No imports broken
- [ ] No missing dependencies

---

**Next Steps**: Create FULL_FEATURE_BASELINE.md with detailed inventory of all features
