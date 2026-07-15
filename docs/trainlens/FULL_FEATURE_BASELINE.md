# X-AnyLabeling Complete Feature Baseline

**Generated**: 2026-07-15  
**Purpose**: Comprehensive inventory of all existing X-AnyLabeling features that MUST be preserved

---

## 1. File and Directory Operations

### 1.1 File Opening and Navigation
- **Open Single File** (`Ctrl+O`): Open an image file
  - Entry: File → Open / Toolbar button
  - Source: `label_widget.py` line ~1090 `open_`
  - Supported: JPG, PNG, BMP, WEBP, HEIF, etc.
  
- **Open Directory** (`Ctrl+U`): Load all images from a directory
  - Entry: File → Open Dir
  - Source: `label_widget.py` line ~1109 `opendir`
  - Features: Recursive scanning, natural sorting, image filtering

- **Open Video** (`Ctrl+V`): Load video file for frame-by-frame annotation
  - Entry: File → Open Video
  - Source: `label_widget.py` line ~1119 `openvideo`
  - Features: Video player controls, frame extraction

- **Open Recent Files**: Quick access to recently opened files
  - Entry: File → Open Recent
  - Source: `label_widget.py` line 2014
  - Features: Persistent history

- **Next/Previous Image** (`D`/`A`): Navigate through image list
  - Entry: File → Next Image / Previous Image
  - Source: `label_widget.py` lines ~1100, ~1104
  - Features: Wraparound navigation

- **Next/Previous Unchecked Image**: Navigate to unreviewed images
  - Entry: File → Next Unchecked / Previous Unchecked
  - Source: `label_widget.py` lines ~1113, ~1117
  - Features: Annotation workflow support

### 1.2 File Saving
- **Save** (`Ctrl+S`): Save current annotations
  - Entry: File → Save / Toolbar button
  - Source: `label_widget.py` line ~1057
  - Format: JSON (X-AnyLabeling native format)

- **Save As**: Save annotations with new filename
  - Entry: File → Save As
  - Source: `label_widget.py` line ~1063
  - Features: Custom output path

- **Save Automatically** (`Ctrl+M`): Toggle auto-save mode
  - Entry: File → Save Automatically
  - Source: `label_widget.py` line ~1044
  - Features: Saves after each annotation change

- **Save With Image Data**: Embed image in annotation file
  - Entry: File → Save With Image Data
  - Source: `label_widget.py` line ~1069
  - Features: Base64-encoded image embedding

### 1.3 Output Directory Management
- **Change Output Directory** (`Ctrl+R`): Set annotation output location
  - Entry: File → Change Output Dir
  - Source: `label_widget.py` line ~1075
  - Features: Separate from source image directory

### 1.4 File Deletion
- **Delete File** (`Ctrl+Delete`): Remove annotation file
  - Entry: File → Delete File
  - Source: `label_widget.py` line ~1174
  - Features: Confirmation dialog, recoverable

- **Delete Image File** (`Ctrl+Shift+Delete`): Remove image and annotation
  - Entry: File → Delete Image File
  - Source: `label_widget.py` line ~1184
  - Features: Strong confirmation, irreversible

### 1.5 File Close
- **Close** (`Ctrl+W`): Close current file
  - Entry: File → Close
  - Source: `label_widget.py` line ~1166
  - Features: Save prompt if unsaved changes

---

## 2. Annotation Shapes and Drawing

### 2.1 Shape Types
All shapes are implemented in `canvas.py` and `shape.py`:

1. **Rectangle** (`R`): Axis-aligned bounding box
   - Entry: Edit → Create Rectangle / Toolbar
   - Source: `label_widget.py` line ~1811 `create_rectangle_mode`
   - Use cases: Object detection, HBB

2. **Polygon** (`P`): Arbitrary closed polygons
   - Entry: Edit → Create Polygon / Toolbar
   - Source: `label_widget.py` line ~1807 `create_mode`
   - Features: Multi-point drawing, auto-close

3. **Circle** (`Shift+C`): Circle annotation
   - Entry: Edit → Create Circle / Toolbar
   - Source: `label_widget.py` line ~1821 `create_circle_mode`
   - Features: Center + radius definition

4. **Line** (`L`): Single line segment
   - Entry: Edit → Create Line / Toolbar
   - Source: `label_widget.py` line ~1823 `create_line_mode`
   - Use cases: Lane detection

5. **Point** (`Shift+P`): Single point marker
   - Entry: Edit → Create Point / Toolbar
   - Source: `label_widget.py` line ~1825 `create_point_mode`
   - Use cases: Keypoint annotation

6. **Line Strip**: Connected line segments
   - Entry: Edit → Create LineStrip / Toolbar
   - Source: `label_widget.py` line ~1827 `create_linestrip_mode`
   - Use cases: Polyline, contours

7. **Rotated Box** (`O`): Oriented bounding box
   - Entry: Edit → Create Rotation / Toolbar
   - Source: `label_widget.py` line ~1815 `create_rotation_mode`
   - Features: Rotation handle, angle display
   - Use cases: OBB detection, DOTA format

8. **Quadrilateral** (`Q`): Four-point quadrilateral
   - Entry: Edit → Create Quadrilateral / Toolbar
   - Source: `label_widget.py` line ~1819 `create_quadrilateral_mode`
   - Use cases: Text detection, OCR

9. **Cuboid** (`3D Box`): 3D cuboid from rectangle
   - Entry: Edit → Create Cuboid / Toolbar
   - Source: `label_widget.py` line ~1813 `create_cuboid_mode`
   - Use cases: 3D object annotation

### 2.2 Drawing Modes
- **Edit Mode** (`Ctrl+J`): Select and edit existing shapes
  - Source: `label_widget.py` line ~1809 `edit_mode`
  - Features: Multi-selection, drag, resize, rotate

- **Brush Polygon Mode**: Paint polygon with brush
  - Source: `label_widget.py` line ~1808 `create_brush_polygon_mode`
  - Features: Brush size adjustment, pressure sensitivity

- **Edit Brush Mode**: Edit shapes with brush tool
  - Source: `label_widget.py` line ~1810 `edit_brush_mode`
  - Features: Add/remove regions

- **Fill Drawing Polygon**: Toggle polygon fill while drawing
  - Entry: View → Fill Drawing Polygon
  - Source: `label_widget.py` line ~1720 `fill_drawing`
  - Features: Visual feedback during creation

### 2.3 Shape Editing Operations
- **Move Shape**: Click and drag shape
  - Features: Constrain to axes with Shift

- **Resize Shape**: Drag vertex or edge
  - Features: Proportional resize with Shift

- **Rotate Shape**: Drag rotation handle (rotated boxes)
  - Features: Angle snapping, visual feedback

- **Add Vertex**: Click on edge (polygons, linestrips)
  - Features: Precise point insertion

- **Remove Vertex** (`Backspace`): Delete selected point
  - Entry: Edit → Remove Point
  - Source: `label_widget.py` line ~1288 `remove_point`
  - Features: Eraser tool support

- **Undo Last Point** (`Ctrl+Z` during drawing): Remove last added point
  - Entry: Edit → Undo Last Point
  - Source: `label_widget.py` line ~1283 `undo_last_point`

- **Duplicate** (`Ctrl+D`): Clone selected shape
  - Entry: Edit → Duplicate
  - Source: `label_widget.py` line ~1213 `duplicate`
  - Features: Offset placement

- **Copy** (`Ctrl+C`): Copy shape to clipboard
  - Entry: Edit → Copy
  - Source: `label_widget.py` line ~1218 `copy`

- **Paste** (`Ctrl+V`): Paste shape from clipboard
  - Entry: Edit → Paste
  - Source: `label_widget.py` line ~1233 `paste`

- **Copy Coordinates** (`Ctrl+Shift+C`): Copy shape coordinates as JSON
  - Entry: Edit → Copy Coordinates
  - Source: `label_widget.py` line ~1226 `copy_coordinates`
  - Features: Developer-friendly JSON format

- **Delete** (`Delete`): Remove selected shape
  - Entry: Edit → Delete
  - Source: `label_widget.py` line ~1198 `delete`
  - Features: Multi-selection support

- **Undo** (`Ctrl+Z`): Undo last operation
  - Entry: Edit → Undo
  - Source: `label_widget.py` line ~1293 `undo`
  - Features: Multi-level undo stack

### 2.4 Shape Selection
- **Single Selection**: Click on shape
  - Features: Highlight, show label, enable editing

- **Multi-Selection** (`Shift+Click`): Add to selection
  - Features: Group operations, union selection mode

- **Union Selection Mode** (`Ctrl+U`): Toggle additive selection
  - Entry: Edit → Union Selection
  - Source: `label_widget.py` line ~1194 `union_selection`

- **Intent-Aware Selection**: Prioritize nearby vertices and editable edges
  - Features: Smart selection for overlapping shapes

- **Group Selection**: Select all shapes in a group
  - Features: Move group together, safe group operations

---

## 3. Label and Attribute Management

### 3.1 Labeling
- **Edit Label** (`Ctrl+E`): Change shape label
  - Entry: Edit → Edit Label
  - Source: `label_widget.py` line ~1711 `edit`
  - Features: Label suggestion, autocomplete, recent labels

- **Label Dialog**: Assign label during or after creation
  - Features: Text input, dropdown, label list

- **Label List Widget**: Visual list of all shapes and labels
  - Location: Right dock panel
  - Source: `label_widget.py` LabelListWidget
  - Features: Click to select, visibility toggle, lock toggle

- **Unique Label List**: Show distinct labels in current image
  - Location: Bottom dock panel
  - Features: Filter by label, statistics

### 3.2 Attributes
- **Shape Attributes**: Custom key-value metadata per shape
  - Entry: Upload → Attributes (upload attrs config file)
  - Source: `label_widget.py` line ~1417 `upload_shape_attrs_file`
  - Features: Dropdown, checkbox, text input controls

- **Attribute Panel**: Edit attributes in right panel
  - Features: Dynamic UI based on attribute config

### 3.3 Group ID
- **Group ID Assignment**: Assign shapes to groups
  - Features: Track multiple instances, object relationships

- **Auto Use Last Group ID** (`Ctrl+Shift+G`): Reuse previous group ID
  - Entry: Edit → Auto Use Last Group ID
  - Source: `label_widget.py` line ~1152 `auto_use_last_gid_mode`

- **Group ID Filter**: Filter shapes by group
  - Location: Label list filter menu
  - Features: Show/hide groups

### 3.4 Label Flags and Image Flags
- **Label Flags**: Per-shape binary flags
  - Entry: Upload → Label Flags
  - Source: `label_widget.py` line ~1409 `upload_label_flags_file`
  - Features: Custom flag definitions

- **Image Flags**: Image-level binary flags
  - Entry: Upload → Image Flags
  - Source: `label_widget.py` line ~1402 `upload_image_flags_file`
  - Features: QA workflow support

### 3.5 Label Classes
- **Label Classes File**: Predefined label list
  - Entry: Upload → Label Classes
  - Source: `label_widget.py` line ~1423 `upload_label_classes_file`
  - Features: Class constraints, color mapping

---

## 4. Auto-Labeling (AI Models)

### 4.1 Auto-Labeling Widget
- **Toggle Auto Labeling** (`Ctrl+A`): Show/hide AI panel
  - Entry: View → Auto Labeling / Toolbar
  - Source: `label_widget.py` line ~1742 `toggle_auto_labeling_widget`
  - Location: Left dock panel

### 4.2 Model Types
All models in `anylabeling/services/auto_labeling/`:

#### Detection Models (50+ models)
- YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10
- YOLO11, YOLO12, YOLO26
- YOLOX, YOLO-NAS
- D-FINE, DAMO-YOLO, Gold_YOLO
- RT-DETR, RF-DETR, DEIMv2

#### Segmentation Models
- YOLOv5-Seg, YOLOv8-Seg, YOLO11-Seg, YOLO26-Seg
- SAM (Segment Anything Model), SAM-HQ, SAM-Med2D
- SAM 2 (video segmentation)
- SAM 3 (text-grounded segmentation)
- EdgeSAM, EfficientViT-SAM, MobileSAM

#### Classification Models
- YOLOv5-Cls, YOLOv8-Cls, YOLO11-Cls
- InternImage, PULC

#### Pose Estimation Models
- YOLOv8-Pose, YOLO11-Pose, YOLO26-Pose
- DWPose, RTMO
- SCRFD (face landmarks)

#### OCR Models
- PP-OCRv4, PP-OCRv5, PP-OCRv6
- PP-DocLayoutV3 (layout analysis)
- PaddleOCR-VL (document parsing)

#### Grounding Models
- Grounding DINO
- YOLO-World, YOLOE
- SAM 3 (grounded)
- LocateAnything

#### Vision-Language Models
- Rex-Omni
- Florence2
- Qwen3-VL, Gemini, ChatGPT, GLM

#### Other Models
- Depth Anything (depth estimation)
- RMBG 1.4/2.0 (background removal)
- RAM, RAM++ (image tagging)
- CountGD, GeCO, GeCo2 (object counting)
- CLRNet (lane detection)
- TrackTrack, Bot-SORT, ByteTrack (tracking)

### 4.3 Model Operations
- **Load Model**: Select and load from model list
  - Features: Local ONNX, remote API, model download

- **Run Auto-Labeling** (`F5`): Run model on current image
  - Features: Real-time inference, result overlay

- **Run All Images** (`Ctrl+Shift+A`): Batch inference
  - Entry: Edit → Run All Images
  - Source: `label_widget.py` line ~1188 `run_all_images`
  - Features: Progress bar, cancel support

- **Model Configuration**: Adjust model parameters
  - Features: Confidence threshold, NMS, input size

### 4.4 Remote Inference
- **Remote Server**: Use X-AnyLabeling-Server for inference
  - Entry: Tool → Remote Server
  - Features: HTTP/gRPC, load balancing

---

## 5. Format Import and Export

### 5.1 Export Formats
All exports in `label_widget.py` lines 2046-2130:

- **YOLO HBB**: Horizontal bounding boxes
- **YOLO OBB**: Oriented bounding boxes
- **YOLO Seg**: Segmentation masks
- **YOLO Pose**: Keypoint annotations
- **VOC Detection**: Pascal VOC XML
- **VOC Segmentation**: Pascal VOC with masks
- **COCO Detection**: COCO JSON (bbox)
- **COCO Segmentation**: COCO JSON (segmentation)
- **COCO Keypoints**: COCO JSON (keypoints)
- **DOTA**: Oriented bounding boxes
- **MASK**: Binary masks
- **MOT**: Multi-Object Tracking format
- **ODVG**: Object Detection Visual Grounding
- **MM-Grounding-DINO**: Multimodal grounding
- **PPOCR Rec**: Text recognition
- **PPOCR KIE**: Key information extraction

### 5.2 Upload/Import Formats
All uploads in `label_widget.py` lines 1400-1570:

- **YOLO annotations** (HBB, OBB, Seg, Pose)
- **VOC annotations** (Detection, Segmentation)
- **COCO annotations** (Detection, Segmentation, Keypoints)
- **DOTA annotations**
- **MASK annotations**
- **MOT annotations**
- **ODVG annotations**
- **PPOCR annotations** (Rec, KIE)

### 5.3 CLI Conversion
- **Convert Command**: `xanylabeling convert --task <task>`
  - Source: `app.py` line 85
  - Tasks: yolo2xlabel, xlabel2yolo, coco2xlabel, xlabel2coco, etc.

---

## 6. Video Annotation

### 6.1 Video Player
- **Open Video** (`Ctrl+V`): Load video file
  - Features: MP4, AVI, MOV support

- **Frame Navigation**: Step through video frames
  - Features: Play, pause, previous frame, next frame

- **Video Annotation**: Annotate individual frames
  - Features: Frame extraction, annotation persistence

### 6.2 Video Tracking
- **SAM2-Video**: Interactive video object segmentation
  - Entry: Load SAM2-Video model in Auto-Labeling
  - Features: Propagate mask across frames

- **SAM3-Video**: Text-grounded video segmentation
  - Features: Natural language prompts

- **Multi-Object Tracking**: TrackTrack, Bot-SORT, ByteTrack
  - Features: Automatic track ID assignment

### 6.3 Video Classifier
- **Video Classifier Panel** (`Ctrl+Shift+V`): Timeline segment classification
  - Entry: View → Video Classifier
  - Source: Referenced in `label_widget.py` line 82
  - Features: Segment descriptions, AI-assisted segmentation

### 6.4 Export Video
- **Save Visualization Video**: Export annotated video
  - Entry: Tool → Save Visualization Video
  - Features: Overlay annotations on video

---

## 7. View and Display Controls

### 7.1 Zoom and Fit
- **Zoom In** (`Ctrl++`): Increase magnification
- **Zoom Out** (`Ctrl+-`): Decrease magnification
- **Zoom to Original** (`Ctrl+0`): 100% size
- **Fit Window** (`Ctrl+F`): Fit image to window
- **Fit Width** (`Ctrl+Shift+F`): Fit image width
- **Zoom Widget**: Visual zoom slider and percentage display

### 7.2 Canvas Adjustment
- **Brightness/Contrast Dialog** (`Ctrl+B`): Adjust display
  - Entry: View → Brightness Contrast
  - Source: Referenced in `label_widget.py` line 71
  - Features: Real-time adjustment, reset button

- **Canvas Adjustment Widget**: Collapsible panel
  - Entry: View → Canvas Adjustment
  - Source: `label_widget.py` line 73
  - Features: Opacity slider, brightness/contrast controls

### 7.3 Visibility Controls
- **Visibility Shapes** (`Ctrl+H`): Toggle all shapes visibility
  - Entry: View → Visibility Shapes
  - Source: `label_widget.py` line ~1161 `visibility_shapes_mode`

- **Hide/Show Individual Shapes**: Checkboxes in label list
  - Features: Per-shape visibility

- **Keep Previous Mode** (`Ctrl+K`): Keep annotation on next image
  - Entry: View → Keep Previous Annotation
  - Source: `label_widget.py` line ~1133 `keep_prev_mode`

### 7.4 Compare View
- **Toggle Compare View** (`Ctrl+T`): Side-by-side image comparison
  - Entry: File → Toggle Compare View
  - Source: `label_widget.py` line ~1123 `toggle_compare_view`
  - Features: Synchronized zoom/pan, split-screen slider

### 7.5 Navigator
- **Show Navigator** (`Ctrl+N`): Mini-map overview window
  - Entry: View → Navigator
  - Source: `label_widget.py` line ~1731 `show_navigator`
  - Features: Full image thumbnail, viewport indicator

### 7.6 Crosshair Settings
- **Crosshair Settings Dialog**: Configure crosshair display
  - Entry: View → Crosshair Settings
  - Features: Size, color, opacity adjustment

---

## 8. Special Annotation Panels

### 8.1 Chatbot
- **Chatbot Dialog** (`Ctrl+Shift+C`): AI assistant
  - Entry: View → Chatbot
  - Source: `label_widget.py` line 74
  - Features: Natural language queries, image understanding

### 8.2 VQA (Visual Question Answering)
- **VQA Dialog** (`Ctrl+Shift+Q`): Ask questions about image
  - Entry: View → VQA
  - Source: `label_widget.py` line 78
  - Features: Question input, answer display

### 8.3 Image Classifier
- **Image Classifier Dialog** (`Ctrl+Shift+I`): Image-level classification
  - Entry: View → Image Classifier
  - Source: `label_widget.py` line 76
  - Features: Class selection, confidence scores

### 8.4 PaddleOCR Panel
- **PaddleOCR Dialog** (`Ctrl+Shift+P`): Document parsing
  - Entry: View → PaddleOCR
  - Source: `label_widget.py` line 81
  - Features: Layout analysis, text recognition, intelligent editing

---

## 9. Training System (Existing - MUST PRESERVE)

### 9.1 Ultralytics Training
- **Ultralytics Dialog** (`Ctrl+Shift+T`): Training interface
  - Entry: Train → Ultralytics
  - Source: `label_widget.py` line 3250, line 2045
  - Location: `anylabeling/views/training/ultralytics_dialog.py`

### 9.2 Training Features
- **Data Tab**: Dataset summary and validation
  - Features: Task type selection, class distribution, image count

- **Config Tab**: Training parameter configuration
  - Features: Model selection, hyperparameters, augmentation

- **Train Tab**: Training execution and monitoring
  - Features: Start/stop, real-time logs, progress display

### 9.3 Training Manager
- **TrainingManager**: Background training process
  - Location: `anylabeling/services/auto_training/ultralytics/trainer.py`
  - Features: Subprocess execution, log redirection, event callbacks

### 9.4 Export Manager
- **ExportManager**: Model export to ONNX, TensorRT, etc.
  - Location: `anylabeling/services/auto_training/ultralytics/exporter.py`
  - Features: Multiple export formats, optimization options

---

## 10. Tools and Utilities

### 10.1 Overview Dialog
- **Overview** (`Ctrl+Shift+O`): Dataset statistics
  - Entry: Tool → Overview
  - Source: `label_widget.py` line ~1197 `overview`
  - Features: Class distribution, shape count, annotation summary

### 10.2 Shape Converter
- **Shape Converter**: Convert between shape types
  - Features: Rectangle → Polygon, Polygon → Rectangle, etc.

### 10.3 Annotation Check Status
- **Toggle Annotation Checked** (`Ctrl+Shift+K`): Mark image as reviewed
  - Entry: File → Toggle Annotation Checked
  - Source: `label_widget.py` line ~1169 `toggle_annotation_checked`
  - Features: Visual indicator, workflow support

### 10.4 Visualization Export
- **Save Visualization Image**: Export annotated image
  - Entry: Tool → Save Visualization Image
  - Features: PNG/JPG output, overlay annotations

- **Save Visualization Video**: Export annotated video
  - Entry: Tool → Save Visualization Video
  - Features: Video output with annotations

### 10.5 Search and Filter
- **Search Bar** (`Ctrl+F`): Search filenames
  - Features: Regex support, case-sensitive option

- **Label Filter**: Filter shapes by label
  - Location: Label list context menu
  - Features: Show/hide by label

- **Group ID Filter**: Filter shapes by group ID
  - Location: Label list context menu
  - Features: Show/hide by group

---

## 11. Settings and Preferences

### 11.1 Settings Dialog
- **Settings** (`Ctrl+,`): Application preferences
  - Entry: Edit → Settings
  - Location: `anylabeling/views/labeling/settings/`
  - Features: Multi-tab settings interface

### 11.2 Settings Categories
- **General**: Default save format, auto-save interval
- **Display**: Canvas settings, shape appearance
- **Auto-Labeling**: Model cache, inference settings
- **Shortcuts**: Keyboard shortcut customization
- **Advanced**: Debug mode, experimental features

### 11.3 Theme
- **Theme Menu**: Visual theme selection
  - Entry: View → Theme
  - Source: `label_widget.py` line 2007
  - Themes: Light, Dark, System

### 11.4 Language
- **Language Menu**: UI localization
  - Entry: View → Language
  - Source: `label_widget.py` line 2008
  - Languages: English, Chinese (Simplified), Japanese, Korean

### 11.5 Config File
- **Config Command**: Show config file path
  - CLI: `xanylabeling config`
  - Features: User settings persistence

---

## 12. System Integration

### 12.1 Clipboard
- **Use System Clipboard** (`Ctrl+Shift+X`): System vs internal clipboard
  - Entry: Edit → Use System Clipboard
  - Source: `label_widget.py` line ~1157 `use_system_clipboard`

### 12.2 Drag and Drop
- **Drag Image**: Drop image file to open
- **Drag Folder**: Drop folder to open directory

### 12.3 Recent Files
- **Recent Files List**: Quick access to recent files
  - Entry: File → Open Recent
  - Features: Persistent history, clear history option

---

## 13. Shape Locking and Protection

### 13.1 Shape Lock
- **Toggle Shape Lock** (`Ctrl+L`): Lock selected shapes
  - Entry: Edit → Toggle Shape Lock
  - Source: `label_widget.py` line ~1238 `toggle_shape_lock`
  - Features: Prevent geometry changes, allow metadata editing

### 13.2 Per-Shape Lock
- **Lock Icon**: Individual shape lock in label list
  - Features: Visual indicator, click to toggle

---

## 14. Help and About

### 14.1 Help Menu
- **Help**: Links to documentation
  - Entry: Help → Documentation
  - Target: GitHub docs

### 14.2 About Dialog
- **About**: Version, license, credits
  - Entry: Help → About
  - Source: `label_widget.py` line 69
  - Features: Version display, GitHub link, license info

### 14.3 System Checks
- **Checks Command**: Display system info
  - CLI: `xanylabeling checks`
  - Features: Python version, package versions, GPU info

### 14.4 Version
- **Version Command**: Show version
  - CLI: `xanylabeling version`
  - Current: 4.0.0-beta.13

---

## 15. Critical File Paths

### 15.1 Core Source Files
```
anylabeling/
├── app.py                          # Main entry point (multiprocessing, CLI parser)
├── app_info.py                     # Version, app name, URLs
├── config.py                       # Config management
├── views/
│   ├── mainwindow.py               # MainWindow (QMainWindow)
│   └── labeling/
│       ├── label_wrapper.py        # LabelingWrapper (container)
│       ├── label_widget.py         # LabelingWidget (7118 lines - PROTECT)
│       ├── label_file.py           # LabelFile I/O
│       ├── shape.py                # Shape model (824 lines - PROTECT)
│       └── widgets/
│           ├── canvas.py           # Canvas (5259 lines - PROTECT)
│           ├── auto_labeling_widget.py
│           ├── chatbot_dialog.py
│           ├── classifier_dialog.py
│           ├── ppocr_dialog.py
│           └── ...
└── services/
    ├── auto_labeling/              # 50+ model implementations - PROTECT
    │   ├── model_manager.py
    │   ├── __base__/               # Base model classes
    │   └── ...
    └── auto_training/              # Existing training - PROTECT
        └── ultralytics/
            ├── trainer.py          # TrainingManager
            └── exporter.py         # ExportManager
```

### 15.2 Configuration Files
```
pyproject.toml                      # Build config, dependencies, extras
~/.config/X-AnyLabeling/            # User settings (Linux/Mac)
%APPDATA%/X-AnyLabeling/            # User settings (Windows)
```

---

## Summary Statistics

- **Total Features**: 100+ discrete features
- **Shape Types**: 9 types
- **Menu Items**: 100+ actions across 9 menus
- **AI Models**: 50+ supported models
- **Export Formats**: 15+ formats
- **Import Formats**: 10+ formats
- **Keyboard Shortcuts**: 50+ shortcuts
- **Core Source Files**: 7000+ lines in critical files

---

**CRITICAL**: All features listed above MUST remain functional after TrainLens integration. Any feature deletion or breakage is unacceptable.

**Next Document**: REUSE_AND_PROTECTION_MATRIX.md
