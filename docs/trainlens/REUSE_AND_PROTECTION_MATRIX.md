# Module Reuse and Protection Matrix

**Generated**: 2026-07-15  
**Purpose**: Classify each X-AnyLabeling module by modification policy for TrainLens integration

---

## Classification Legend

- **PROTECT**: Absolutely no modifications allowed. Read-only for Run Monitor.
- **REUSE**: Direct reuse without changes. Run Monitor may import and call.
- **EXTEND**: Safe to add new methods/classes, but never modify existing code.
- **STYLE_ONLY**: Only visual/styling changes allowed, no logic changes.
- **SAFE_MODIFY**: Can be carefully modified with full regression testing.
- **NEW**: New code created for Run Monitor only.

---

## 1. Core Application Layer

| Module | Path | Classification | Justification | Run Monitor Usage |
|--------|------|----------------|---------------|-------------------|
| Main Entry | `app.py` | **EXTEND** | Add CLI commands, don't touch existing GUI launch | May add `run-monitor` subcommand |
| App Info | `app_info.py` | **PROTECT** | Version and metadata, read-only | Read version only |
| Config | `config.py` | **REUSE** | Existing functions work fine | Use `get_config()`, `save_config()` |
| MainWindow | `views/mainwindow.py` | **PROTECT** | Only contains LabelingWrapper, minimal code | No changes needed |
| LabelingWrapper | `views/labeling/label_wrapper.py` | **PROTECT** | Container for LabelingWidget | No changes |

**Risk**: Modifying entry point can break existing launch behavior.

---

## 2. Labeling Core (Absolute Protection Zone)

| Module | Path | Classification | LOC | Justification | Run Monitor Usage |
|--------|------|----------------|-----|---------------|-------------------|
| LabelingWidget | `views/labeling/label_widget.py` | **SAFE_MODIFY** | 7118 | Needs ONE new menu entry for "Run" | Add Run menu at line ~2012 |
| Canvas | `views/labeling/widgets/canvas.py` | **PROTECT** | 5259 | Complex drawing engine, zero tolerance for bugs | No interaction |
| Shape | `views/labeling/shape.py` | **PROTECT** | 824 | Core data model, any change breaks serialization | No interaction |
| LabelFile | `views/labeling/label_file.py` | **PROTECT** | - | File I/O, fragile and critical | No interaction |
| Label Dialog | `views/labeling/widgets/label_dialog.py` | **PROTECT** | - | Base class for dialogs, inherited by LabelingWidget | Run Monitor window is separate QDialog |

**Critical Note**: LabelingWidget modification is MINIMAL and SURGICAL:
- Add ONE new menu: `self.menus.run = self.menu(self.tr("Run"))`
- Add ONE new action: `run_monitor = action(..., self.open_run_monitor, ...)`
- Add ONE new method: `def open_run_monitor(self): ...` that opens RunMonitorWindow
- Total: ~20 lines of new code, zero existing code modified

---

## 3. Auto-Labeling Services (Read-Only for Run Monitor)

| Module | Path | Classification | Justification | Run Monitor Usage |
|--------|------|----------------|---------------|-------------------|
| Model Manager | `services/auto_labeling/model_manager.py` | **PROTECT** | Manages 50+ models, complex state | No usage |
| All Model Implementations | `services/auto_labeling/*.py` | **PROTECT** | YOLO, SAM, OCR, etc. - fragile inference code | No usage |
| Base Models | `services/auto_labeling/__base__/*.py` | **PROTECT** | Abstract base classes | No usage |
| Engines | `services/auto_labeling/engines/*.py` | **PROTECT** | ONNX, TensorRT, OpenCV DNN backends | No usage |
| Trackers | `services/auto_labeling/trackers/*.py` | **PROTECT** | ByteTrack, Bot-SORT, etc. | No usage |

**Risk**: Any modification breaks auto-labeling, which is core value proposition.

---

## 4. Existing Training System (Absolute Protection)

| Module | Path | Classification | LOC | Justification | Run Monitor Usage |
|--------|------|----------------|-----|---------------|-------------------|
| Ultralytics Dialog | `views/training/ultralytics_dialog.py` | **PROTECT** | 80443 | Massive file, working training UI | ZERO - completely separate |
| Training Manager | `services/auto_training/ultralytics/trainer.py` | **REUSE** | - | Good example of subprocess management | Study implementation patterns |
| Export Manager | `services/auto_training/ultralytics/exporter.py` | **PROTECT** | - | Model export logic | No usage |
| Training Utils | `services/auto_training/ultralytics/*.py` | **PROTECT** | - | Validators, I/O, config | No usage |
| Training Widgets | `views/training/widgets/*.py` | **PROTECT** | - | Custom widgets for Ultralytics dialog | No usage |

**Critical**: Two training systems coexist peacefully. Run Monitor does NOT replace Ultralytics training.

**Menu Structure**:
```
Train → Ultralytics          # KEEP - existing entry
Run → Run Monitor            # NEW - Run Monitor entry
```

---

## 5. Widgets and Dialogs (Mostly Protected)

| Module | Path | Classification | Justification | Run Monitor Usage |
|--------|------|----------------|---------------|-------------------|
| AboutDialog | `views/labeling/widgets/about_dialog.py` | **PROTECT** | About dialog | No usage |
| AutoLabelingWidget | `views/labeling/widgets/auto_labeling_widget.py` | **PROTECT** | AI model panel | No usage |
| ChatbotDialog | `views/labeling/widgets/chatbot_dialog.py` | **PROTECT** | LLM integration | No usage |
| ClassifierDialog | `views/labeling/widgets/classifier_dialog.py` | **PROTECT** | Image classifier | No usage |
| PPOCRDialog | `views/labeling/widgets/ppocr_dialog.py` | **PROTECT** | OCR panel | No usage |
| VideoClassifierDialog | `views/labeling/widgets/video_classifier_dialog.py` | **PROTECT** | Video classifier | No usage |
| NavigatorDialog | `views/labeling/widgets/navigator_widget.py` | **PROTECT** | Mini-map | No usage |
| OverviewDialog | `views/labeling/widgets/overview_dialog.py` | **PROTECT** | Dataset stats | No usage |
| All Other Widgets | `views/labeling/widgets/*.py` | **PROTECT** | Specialized UI components | No usage |

**Strategy**: Run Monitor creates its own widgets, never modifies existing ones.

---

## 6. Settings and Configuration

| Module | Path | Classification | Justification | Run Monitor Usage |
|--------|------|----------------|---------------|-------------------|
| SettingsController | `views/labeling/settings/` | **REUSE** | Settings management | May add Run Monitor settings tab |
| SettingsDialog | `views/labeling/settings/` | **EXTEND** | Settings UI | Add Run Monitor section if needed |
| Config Module | `config.py` | **REUSE** | Global config | Store Run Monitor state |

**Safe Extension**: Add new settings namespace `run_monitor` without touching existing keys.

---

## 7. Utilities and Helpers

| Module | Path | Classification | Justification | Run Monitor Usage |
|--------|------|----------------|---------------|-------------------|
| Qt Utils | `views/labeling/utils/qt.py` | **REUSE** | Icon loading, action creation | Reuse `new_icon()`, `action()` |
| Style Utils | `views/labeling/utils/style.py` | **REUSE** | Stylesheet helpers | Reuse for consistent styling |
| File Utils | `views/labeling/utils/file_search.py` | **REUSE** | Search and filter | May reuse for workspace scanning |
| General Utils | `views/labeling/utils/*.py` | **REUSE** | Various helpers | Import as needed |
| Shape Utils | `views/labeling/utils/shape.py` | **PROTECT** | Shape calculations | No usage |

**Strategy**: Reuse utility functions freely, never modify them.

---

## 8. Common Services

| Module | Path | Classification | Justification | Run Monitor Usage |
|--------|------|----------------|---------------|-------------------|
| Converter | `views/common/converter.py` | **PROTECT** | Format conversion | No usage |
| Device Manager | `views/common/device_manager.py` | **REUSE** | GPU detection | May use for resource monitoring |
| Toaster | `views/common/toaster.py` | **REUSE** | Toast notifications | May use for notifications |
| Checks | `views/common/checks.py` | **REUSE** | System info | May use for environment detection |

---

## 9. Resources and Assets

| Module | Path | Classification | Justification | Run Monitor Usage |
|--------|------|----------------|---------------|-------------------|
| Resources | `anylabeling/resources/` | **EXTEND** | Icons, translations | Add Run Monitor icons |
| Icons | `anylabeling/resources/images/` | **EXTEND** | Icon files | Add new icons for Run Monitor |
| Translations | `anylabeling/resources/translations/` | **EXTEND** | i18n files | Add Run Monitor strings |

**Safe Extension**: Add new resources without touching existing ones.

---

## 10. Run Monitor Modules (All New)

| Module | Path | Classification | Justification |
|--------|------|----------------|---------------|
| Run Monitor Window | `views/run_monitor/run_monitor_window.py` | **NEW** | Main Run Monitor UI |
| Workspace Panel | `views/run_monitor/workspace_panel.py` | **NEW** | Workspace selector |
| Run Config Panel | `views/run_monitor/run_configuration_panel.py` | **NEW** | Script/Python selector |
| Run Status Panel | `views/run_monitor/run_status_panel.py` | **NEW** | Status display |
| Console Panel | `views/run_monitor/console_panel.py` | **NEW** | Log viewer |
| Resource Panel | `views/run_monitor/resource_panel.py` | **NEW** | CPU/GPU charts |
| Metrics Panel | `views/run_monitor/metrics_panel.py` | **NEW** | Training curves |
| Run Monitor Service | `services/run_monitor/` | **NEW** | All business logic |
| Workspace Scanner | `services/run_monitor/workspace_scanner.py` | **NEW** | Scan for scripts |
| Script Detector | `services/run_monitor/script_detector.py` | **NEW** | Identify training scripts |
| Environment Detector | `services/run_monitor/environment_detector.py` | **NEW** | Find Python/venv |
| Process Manager | `services/run_monitor/process_manager.py` | **NEW** | Run training subprocess |
| Resource Monitor | `services/run_monitor/resource_monitor.py` | **NEW** | Monitor CPU/GPU |
| Run Storage | `services/run_monitor/run_storage.py` | **NEW** | Persist run history |
| Event Protocol | `services/run_monitor/event_protocol.py` | **NEW** | Structured events |
| Metrics Readers | `services/run_monitor/metrics/*.py` | **NEW** | Parse metrics from various sources |

---

## 11. Tests (Extend Safely)

| Module | Path | Classification | Justification |
|--------|------|----------------|---------------|
| Existing Tests | `tests/test_*/*.py` | **PROTECT** | Must continue passing | Run before/after integration |
| Run Monitor Tests | `tests/trainlens/*.py` | **NEW** | Test new functionality | Full coverage of Run Monitor |

---

## Modification Summary Table

| Classification | Module Count | Modification Policy |
|----------------|--------------|---------------------|
| **PROTECT** | ~35 | Zero modifications, read-only |
| **REUSE** | ~15 | Import and call, no changes |
| **EXTEND** | ~8 | Add new code, preserve existing |
| **SAFE_MODIFY** | 1 | LabelingWidget: surgical 20-line addition |
| **NEW** | ~20 | Run Monitor modules |

---

## Risk Assessment by Module

### Zero Risk (No Interaction)
- Canvas, Shape, LabelFile
- All auto-labeling models
- Ultralytics training system
- All specialized dialogs (Chatbot, VQA, Classifier, etc.)
- Format converters

### Minimal Risk (Read-Only Usage)
- Config module (use existing functions)
- Qt utils (use existing helpers)
- Device manager (read GPU info)

### Low Risk (Safe Extension)
- Resources (add new icons/translations)
- Settings (add new namespace)

### Medium Risk (Surgical Modification)
- **LabelingWidget** (add Run menu entry)
  - Risk: Breaking existing menu system
  - Mitigation: Minimal change, add only, don't modify existing
  - Regression: Test all existing menus still work

### High Risk (Avoid)
- Modifying any PROTECT module
- Renaming package `anylabeling`
- Deleting existing code

---

## Integration Points (Safe Connection Strategy)

### 1. Menu Integration
**File**: `anylabeling/views/labeling/label_widget.py`  
**Location**: Around line 2012 in `self.menus = utils.Struct(...)`  
**Change**:
```python
self.menus = utils.Struct(
    file=self.menu(self.tr("File")),
    edit=self.menu(self.tr("Edit")),
    view=self.menu(self.tr("View")),
    theme=self.menu(self.tr("Theme")),
    language=self.menu(self.tr("Language")),
    upload=self.menu(self.tr("Upload")),
    export=self.menu(self.tr("Export")),
    tool=self.menu(self.tr("Tool")),
    train=self.menu(self.tr("Train")),  # KEEP - existing
    run=self.menu(self.tr("Run")),      # ADD - new
    help=self.menu(self.tr("Help")),
    recent_files=QtWidgets.QMenu(self.tr("Open Recent")),
)
```

### 2. Action Creation
**File**: `anylabeling/views/labeling/label_widget.py`  
**Location**: Around line 1750-1850 (after other actions)  
**Change**:
```python
# Run Monitor action
run_monitor = action(
    self.tr("Run Monitor"),
    self.open_run_monitor,
    shortcuts.get("run_monitor", "Ctrl+Shift+R"),
    "run",  # icon name
    self.tr("Open Run Monitor for training script execution"),
)
```

### 3. Menu Population
**File**: `anylabeling/views/labeling/label_widget.py`  
**Location**: Around line 2045 (after train menu)  
**Change**:
```python
utils.add_actions(self.menus.train, (ultralytics_train,))  # KEEP
utils.add_actions(self.menus.run, (run_monitor,))          # ADD
```

### 4. Handler Method
**File**: `anylabeling/views/labeling/label_widget.py`  
**Location**: Anywhere after `__init__`, around line 3500+  
**Change**:
```python
def open_run_monitor(self):
    """Open Run Monitor window for training script execution"""
    from anylabeling.views.run_monitor import RunMonitorWindow
    
    if not hasattr(self, '_run_monitor_window') or self._run_monitor_window is None:
        self._run_monitor_window = RunMonitorWindow(parent=self)
    
    self._run_monitor_window.show()
    self._run_monitor_window.raise_()
    self._run_monitor_window.activateWindow()
```

**Total Lines Added**: ~25 lines  
**Total Lines Modified**: 0 lines  
**Total Lines Deleted**: 0 lines

---

## Forbidden Modifications

### Never Do These:
1. ❌ Rename `anylabeling` package to `trainlens`
2. ❌ Delete or comment out any existing menu entry
3. ❌ Modify Canvas, Shape, or LabelFile classes
4. ❌ Change existing auto-labeling model implementations
5. ❌ Modify Ultralytics training dialog or manager
6. ❌ Remove existing QAction definitions
7. ❌ Change existing keyboard shortcuts
8. ❌ Modify existing widget layouts
9. ❌ Change existing signal/slot connections
10. ❌ Modify existing file I/O logic

### Always Do These:
1. ✅ Create new modules in separate directories
2. ✅ Import existing utilities, never copy-paste
3. ✅ Use composition over inheritance for UI
4. ✅ Keep Run Monitor as independent QDialog
5. ✅ Use Qt signals for IPC, not direct calls
6. ✅ Handle all Run Monitor exceptions gracefully
7. ✅ Write tests for all new code
8. ✅ Run existing tests before/after changes
9. ✅ Keep Git history clean with atomic commits
10. ✅ Document all integration points

---

## Verification Checklist

After integration, verify:

- [ ] All existing menus present and functional
- [ ] All existing shortcuts work
- [ ] Canvas drawing works (all 9 shape types)
- [ ] Auto-labeling models load and run
- [ ] Ultralytics training dialog opens
- [ ] All import/export formats work
- [ ] Video annotation works
- [ ] Chatbot, VQA, Classifier panels open
- [ ] Settings dialog works
- [ ] All existing tests pass
- [ ] New "Run" menu appears
- [ ] Run Monitor window opens independently
- [ ] Run Monitor close doesn't crash main window
- [ ] Run Monitor errors don't break labeling

---

**Next Document**: MANUAL_REGRESSION_CHECKLIST.md
