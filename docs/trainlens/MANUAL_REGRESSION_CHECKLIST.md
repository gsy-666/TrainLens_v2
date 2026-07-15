# Manual Regression Testing Checklist

**Generated**: 2026-07-15  
**Purpose**: Manual testing checklist to verify X-AnyLabeling features still work after TrainLens integration

---

## Testing Strategy

1. **Baseline Testing**: Test BEFORE any modifications (using tag `baseline-x-anylabeling-before-trainlens`)
2. **Incremental Testing**: Test after each phase of integration
3. **Final Regression**: Complete retest after all integration work
4. **Automated Tests**: Run `pytest tests/` before and after

---

## Environment Setup

### Test Environment Requirements
- [ ] Windows 10/11 or Linux (Ubuntu 20.04+)
- [ ] Python 3.11+
- [ ] Virtual environment with all dependencies installed
- [ ] Test dataset with varied images (JPEG, PNG, different sizes)
- [ ] Sample annotation files in multiple formats

### Quick Launch Test
```bash
# Launch application
python -m anylabeling.app

# Or if installed
xanylabeling
```

**Expected**: Application window opens without errors, no console warnings

---

## 1. Core Application Launch (Priority: CRITICAL)

| Test | Steps | Expected Result | Status | Notes |
|------|-------|-----------------|--------|-------|
| **Launch GUI** | Run `xanylabeling` or `python -m anylabeling.app` | Main window opens, no crashes | ⬜ |  |
| **Window Title** | Check window title bar | Shows "X-AnyLabeling" | ⬜ |  |
| **Status Bar** | Check bottom status bar | Shows version "4.0.0-beta.13" | ⬜ |  |
| **All Menus Present** | Check menu bar | File, Edit, View, Theme, Language, Upload, Export, Tool, Train, Help visible | ⬜ |  |
| **Toolbars Visible** | Check toolbar areas | Drawing tools, zoom tools, file tools visible | ⬜ |  |
| **Dock Panels** | Check left/right/bottom panels | Auto-labeling (left), label list (right), file list (bottom) visible | ⬜ |  |
| **Close Application** | File → Close or window X | Closes cleanly without crash | ⬜ |  |

---

## 2. File Operations (Priority: CRITICAL)

| Test | Steps | Expected Result | Status | Notes |
|------|-------|-----------------|--------|-------|
| **Open Single Image** | File → Open, select JPG | Image loads and displays | ⬜ |  |
| **Open PNG Image** | File → Open, select PNG | PNG loads correctly | ⬜ |  |
| **Open Directory** | File → Open Dir, select folder | All images listed in file list | ⬜ |  |
| **Navigate Next** | Press `D` or File → Next Image | Moves to next image | ⬜ |  |
| **Navigate Previous** | Press `A` or File → Previous Image | Moves to previous image | ⬜ |  |
| **Save Annotation** | Draw shape, press `Ctrl+S` | JSON file created/updated | ⬜ |  |
| **Save As** | File → Save As, choose name | Annotation saved with new name | ⬜ |  |
| **Auto Save Toggle** | Press `Ctrl+M` | Auto-save mode toggles on/off | ⬜ |  |
| **Change Output Dir** | File → Change Output Dir | Output directory changed | ⬜ |  |
| **Close File** | Press `Ctrl+W` | File closes, prompts if unsaved | ⬜ |  |
| **Delete Annotation** | File → Delete File | Annotation file deleted | ⬜ |  |
| **Recent Files** | File → Open Recent | Shows recently opened files | ⬜ |  |

---

## 3. Drawing and Annotation (Priority: CRITICAL)

| Test | Steps | Expected Result | Status | Notes |
|------|-------|-----------------|--------|-------|
| **Rectangle** | Press `R`, drag on canvas | Rectangle created | ⬜ |  |
| **Polygon** | Press `P`, click points, Enter | Polygon created | ⬜ |  |
| **Circle** | Press `Shift+C`, drag | Circle created | ⬜ |  |
| **Line** | Press `L`, drag | Line created | ⬜ |  |
| **Point** | Press `Shift+P`, click | Point marker created | ⬜ |  |
| **Line Strip** | Click linestrip tool, click points | Multi-segment line created | ⬜ |  |
| **Rotated Box** | Press `O`, drag, adjust rotation | Rotated rectangle with handle | ⬜ |  |
| **Quadrilateral** | Press `Q`, click 4 points | 4-point quad created | ⬜ |  |
| **Cuboid** | Draw rectangle, convert to cuboid | 3D box created | ⬜ |  |
| **Brush Polygon** | Select brush tool, paint | Painted polygon created | ⬜ |  |

---

## 4. Shape Editing (Priority: CRITICAL)

| Test | Steps | Expected Result | Status | Notes |
|------|-------|-----------------|--------|-------|
| **Select Shape** | Press `Ctrl+J`, click shape | Shape selected, vertices visible | ⬜ |  |
| **Move Shape** | Click and drag shape | Shape moves | ⬜ |  |
| **Resize Rectangle** | Drag corner vertex | Rectangle resizes | ⬜ |  |
| **Edit Polygon Vertex** | Drag polygon vertex | Vertex moves | ⬜ |  |
| **Add Vertex** | Click on polygon edge | New vertex added | ⬜ |  |
| **Remove Vertex** | Select vertex, press `Backspace` | Vertex removed | ⬜ |  |
| **Rotate Box** | Drag rotation handle on OBB | Box rotates | ⬜ |  |
| **Duplicate** | Select shape, press `Ctrl+D` | Shape duplicated with offset | ⬜ |  |
| **Copy/Paste** | `Ctrl+C`, `Ctrl+V` | Shape copied and pasted | ⬜ |  |
| **Delete** | Select shape, press `Delete` | Shape deleted | ⬜ |  |
| **Undo** | Press `Ctrl+Z` | Last action undone | ⬜ |  |
| **Multi-Select** | Hold `Shift`, click multiple | Multiple shapes selected | ⬜ |  |

---

## 5. Labeling and Attributes (Priority: HIGH)

| Test | Steps | Expected Result | Status | Notes |
|------|-------|-----------------|--------|-------|
| **Assign Label** | Create shape, enter label | Label assigned and visible | ⬜ |  |
| **Edit Label** | Select shape, press `Ctrl+E` | Label dialog opens, editable | ⬜ |  |
| **Label List** | Check right panel | All shapes listed with labels | ⬜ |  |
| **Upload Label Classes** | Upload → Label Classes, select file | Labels loaded from file | ⬜ |  |
| **Upload Attributes** | Upload → Attributes, select JSON | Attribute controls appear | ⬜ |  |
| **Set Attribute** | Select shape, change attribute | Attribute value saved | ⬜ |  |
| **Group ID** | Assign group ID to shapes | Group ID visible in label list | ⬜ |  |
| **Auto Last Label** | Toggle, create new shape | Previous label auto-applied | ⬜ |  |
| **Auto Last Group** | Press `Ctrl+Shift+G`, create shape | Previous group ID auto-applied | ⬜ |  |

---

## 6. View and Display (Priority: HIGH)

| Test | Steps | Expected Result | Status | Notes |
|------|-------|-----------------|--------|-------|
| **Zoom In** | Press `Ctrl++` | Image zooms in | ⬜ |  |
| **Zoom Out** | Press `Ctrl+-` | Image zooms out | ⬜ |  |
| **Fit Window** | Press `Ctrl+F` | Image fits window | ⬜ |  |
| **Fit Width** | Press `Ctrl+Shift+F` | Image width fits window | ⬜ |  |
| **Zoom 100%** | Press `Ctrl+0` | Image at original size | ⬜ |  |
| **Brightness/Contrast** | Press `Ctrl+B`, adjust sliders | Image brightness/contrast changes | ⬜ |  |
| **Hide All Shapes** | Press `Ctrl+H` | All annotations hidden | ⬜ |  |
| **Show All Shapes** | Press `Ctrl+H` again | All annotations visible | ⬜ |  |
| **Hide Individual** | Uncheck shape in label list | That shape hidden | ⬜ |  |
| **Compare View** | Press `Ctrl+T`, select 2nd image | Side-by-side view | ⬜ |  |
| **Navigator** | Press `Ctrl+N` | Mini-map window opens | ⬜ |  |
| **Fill Drawing** | View → Fill Drawing Polygon | Polygon fill toggles | ⬜ |  |

---

## 7. Auto-Labeling (Priority: HIGH)

| Test | Steps | Expected Result | Status | Notes |
|------|-------|-----------------|--------|-------|
| **Toggle Auto-Labeling** | Press `Ctrl+A` | Auto-labeling panel toggles | ⬜ |  |
| **Model List Loads** | Check auto-labeling panel | Model list populated | ⬜ |  |
| **Load Model** | Select any YOLO model, click Load | Model loads without error | ⬜ |  |
| **Run Inference** | Press `F5` with model loaded | Detections appear | ⬜ |  |
| **Adjust Confidence** | Lower confidence threshold, run again | More detections appear | ⬜ |  |
| **Run All Images** | Press `Ctrl+Shift+A` | Batch inference runs with progress | ⬜ |  |
| **SAM Model** | Load SAM, click point | Segmentation mask appears | ⬜ |  |
| **Clear Annotations** | Run model, clear, run again | Old results cleared | ⬜ |  |

---

## 8. Format Import/Export (Priority: HIGH)

| Test | Steps | Expected Result | Status | Notes |
|------|-------|-----------------|--------|-------|
| **Export YOLO HBB** | Export → YOLO HBB | YOLO txt files generated | ⬜ |  |
| **Export YOLO OBB** | Export → YOLO OBB | YOLO OBB txt files generated | ⬜ |  |
| **Export YOLO Seg** | Export → YOLO Seg | YOLO seg files generated | ⬜ |  |
| **Export VOC** | Export → VOC Detection | Pascal VOC XML generated | ⬜ |  |
| **Export COCO** | Export → COCO Detection | COCO JSON generated | ⬜ |  |
| **Export DOTA** | Export → DOTA | DOTA format files generated | ⬜ |  |
| **Upload YOLO** | Upload → YOLO HBB, select files | YOLO annotations imported | ⬜ |  |
| **Upload VOC** | Upload → VOC Detection, select XMLs | VOC annotations imported | ⬜ |  |
| **Upload COCO** | Upload → COCO Detection, select JSON | COCO annotations imported | ⬜ |  |

---

## 9. Existing Training System (Priority: CRITICAL)

| Test | Steps | Expected Result | Status | Notes |
|------|-------|-----------------|--------|-------|
| **Train Menu Exists** | Check menu bar | "Train" menu visible | ⬜ | MUST remain |
| **Open Ultralytics** | Train → Ultralytics | Ultralytics dialog opens | ⬜ |  |
| **Data Tab** | Check Data tab in dialog | Dataset summary visible | ⬜ |  |
| **Config Tab** | Check Config tab | Configuration options visible | ⬜ |  |
| **Train Tab** | Check Train tab | Training controls visible | ⬜ |  |
| **Select Task** | Choose detection task | Task type changes | ⬜ |  |
| **Configure Dataset** | Set dataset path | Dataset validated | ⬜ |  |
| **Close Dialog** | Close Ultralytics dialog | Dialog closes, main window unaffected | ⬜ |  |

**Note**: Do NOT actually start training unless you have time. Just verify the dialog opens and UI works.

---

## 10. Video Annotation (Priority: MEDIUM)

| Test | Steps | Expected Result | Status | Notes |
|------|-------|-----------------|--------|-------|
| **Open Video** | Press `Ctrl+V`, select MP4 | Video loads, player controls appear | ⬜ |  |
| **Play/Pause** | Click play button | Video plays | ⬜ |  |
| **Next Frame** | Click next frame | Advances one frame | ⬜ |  |
| **Annotate Frame** | Draw shape on frame | Annotation saved for frame | ⬜ |  |
| **Navigate Frames** | Use frame slider | Moves to different frames | ⬜ |  |
| **Video Classifier** | Press `Ctrl+Shift+V` | Video classifier panel opens | ⬜ |  |

---

## 11. Special Panels (Priority: MEDIUM)

| Test | Steps | Expected Result | Status | Notes |
|------|-------|-----------------|--------|-------|
| **Chatbot** | Press `Ctrl+Shift+C` | Chatbot dialog opens | ⬜ |  |
| **VQA** | Press `Ctrl+Shift+Q` | VQA dialog opens | ⬜ |  |
| **Image Classifier** | Press `Ctrl+Shift+I` | Classifier dialog opens | ⬜ |  |
| **PaddleOCR** | Press `Ctrl+Shift+P` | PaddleOCR panel opens | ⬜ |  |
| **Overview** | Press `Ctrl+Shift+O` | Overview dialog shows stats | ⬜ |  |

---

## 12. Settings and Preferences (Priority: MEDIUM)

| Test | Steps | Expected Result | Status | Notes |
|------|-------|-----------------|--------|-------|
| **Open Settings** | Press `Ctrl+,` | Settings dialog opens | ⬜ |  |
| **Change Setting** | Toggle any setting, apply | Setting saved and applied | ⬜ |  |
| **Theme Menu** | View → Theme | Theme options visible | ⬜ |  |
| **Change Theme** | Select Light/Dark theme | Theme changes | ⬜ |  |
| **Language Menu** | View → Language | Language options visible | ⬜ |  |
| **Change Language** | Select language, restart | UI in selected language | ⬜ |  |

---

## 13. Tools and Utilities (Priority: MEDIUM)

| Test | Steps | Expected Result | Status | Notes |
|------|-------|-----------------|--------|-------|
| **Shape Converter** | Tool → Shape Converter | Converter dialog opens | ⬜ |  |
| **Save Visualization** | Tool → Save Visualization Image | Annotated image exported | ⬜ |  |
| **Search Files** | Press `Ctrl+F`, enter search | File list filtered | ⬜ |  |
| **Filter by Label** | Right-click label list, filter | Shapes filtered | ⬜ |  |
| **Check Status** | Press `Ctrl+Shift+K` | Image marked as checked | ⬜ |  |
| **Shape Lock** | Select shape, press `Ctrl+L` | Shape locked (can't move) | ⬜ |  |

---

## 14. Help and About (Priority: LOW)

| Test | Steps | Expected Result | Status | Notes |
|------|-------|-----------------|--------|-------|
| **About Dialog** | Help → About | About dialog opens with version | ⬜ |  |
| **Documentation Link** | Help → Documentation | Link present (may not open if offline) | ⬜ |  |
| **CLI Help** | Run `xanylabeling help` | Help message displays | ⬜ |  |
| **CLI Version** | Run `xanylabeling version` | Version 4.0.0-beta.13 displayed | ⬜ |  |
| **CLI Checks** | Run `xanylabeling checks` | System info displayed | ⬜ |  |

---

## 15. TrainLens Integration Tests (After Integration Only)

| Test | Steps | Expected Result | Status | Notes |
|------|-------|-----------------|--------|-------|
| **Run Menu Exists** | Check menu bar | "Run" menu visible | ⬜ | NEW |
| **Open Run Monitor** | Run → Run Monitor | Run Monitor window opens | ⬜ | NEW |
| **Independent Window** | Verify Run Monitor is separate | Separate QDialog, not embedded | ⬜ | NEW |
| **Close Run Monitor** | Close Run Monitor window | Window closes, main app unaffected | ⬜ | NEW |
| **Both Systems Coexist** | Open Ultralytics AND Run Monitor | Both dialogs open independently | ⬜ | NEW |
| **Run Monitor Error** | Trigger error in Run Monitor | Error doesn't crash main window | ⬜ | NEW |
| **Main App After Run Monitor** | Use Run Monitor, then annotate | Annotation still works normally | ⬜ | NEW |

---

## 16. Automated Test Suite (Priority: CRITICAL)

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_canvas/ -v
pytest tests/test_labeling/ -v
pytest tests/test_auto_labeling/ -v
pytest tests/test_widgets/ -v
```

| Test Suite | Command | Expected Result | Status | Notes |
|------------|---------|-----------------|--------|-------|
| **All Tests** | `pytest tests/` | All pass (or known failures documented) | ⬜ | Before integration |
| **Canvas Tests** | `pytest tests/test_canvas/` | All pass | ⬜ |  |
| **Labeling Tests** | `pytest tests/test_labeling/` | All pass | ⬜ |  |
| **Auto-Labeling Tests** | `pytest tests/test_auto_labeling/` | All pass | ⬜ |  |
| **Widget Tests** | `pytest tests/test_widgets/` | All pass | ⬜ |  |
| **All Tests After** | `pytest tests/` | Same pass/fail as before | ⬜ | After integration |

---

## Testing Schedule

### Phase 0: Baseline (Before ANY modifications)
- [ ] Complete all tests in sections 1-14
- [ ] Run automated test suite
- [ ] Document any existing failures
- [ ] Take screenshots of all major features

### Phase 1: After Start Scripts
- [ ] Test sections 1 (launch with new scripts)
- [ ] Quick smoke test of sections 2-3

### Phase 2: After Run Monitor Service Layer
- [ ] No user-visible changes yet
- [ ] Run automated tests only

### Phase 3: After Run Monitor UI
- [ ] Test section 15 (new Run Monitor features)
- [ ] Retest sections 1-3 (core functionality)
- [ ] Run automated tests

### Phase 4: After Menu Integration
- [ ] Test section 9 (Train menu still works)
- [ ] Test section 15 (Run menu works)
- [ ] Retest sections 1-2 (menus and files)

### Phase 5: Final Regression
- [ ] Complete ALL tests in sections 1-15
- [ ] Run full automated test suite
- [ ] Compare with baseline results
- [ ] Fix any regressions before accepting integration

---

## Regression Criteria

### PASS Criteria
- All critical tests pass
- No new crashes or errors
- All existing features work as before
- Automated tests pass rate same or better
- Run Monitor works independently

### FAIL Criteria (Must Fix Before Accepting)
- Any critical test fails
- New crashes or exceptions
- Existing feature broken
- Automated test pass rate decreases
- Run Monitor errors crash main app

---

## Bug Reporting Template

```markdown
**Test**: [Test name from checklist]
**Section**: [Section number]
**Priority**: [CRITICAL/HIGH/MEDIUM/LOW]

**Steps to Reproduce**:
1. 
2. 
3. 

**Expected Result**:


**Actual Result**:


**Error Messages** (if any):


**Screenshots**:


**Environment**:
- OS: 
- Python: 
- X-AnyLabeling version: 
```

---

**Next Document**: RUN_MONITOR_ARCHITECTURE.md
