# Training Center Migration Plan

**Date**: 2026-07-15  
**Status**: Architecture Design Phase  
**Risk Level**: Medium

---

## Migration Strategy: Progressive Integration (NOT Big Rewrite)

**Principle**: Unify through adapters, preserve existing implementations

---

## Phase A: Audit Only (CURRENT)

✅ Complete training system analysis  
✅ Identify duplicate implementations  
✅ Map capability matrix  
⬜ Get stakeholder approval

**Deliverables**: 3 audit documents (this file + 2 others)

---

## Phase B: Unified Core Models

Create shared abstractions WITHOUT touching existing code:

```
anylabeling/services/training_center/
├── models.py              # TrainingJob, JobStatus, JobEvent
├── job_manager.py         # Global training lock, job lifecycle
└── adapters/
    ├── base.py            # AbstractTrainingAdapter
    ├── ultralytics_adapter.py  # Wraps existing TrainingManager
    └── custom_script_adapter.py  # Wraps existing ProcessManager
```

**Key**: Adapters CALL existing code, don't replace it

---

## Phase C: UI Shell (Training Center Window)

Single window with 3 tabs, delegates to existing systems:

- **Guided Training** → Opens/embeds UltralyticsDialog
- **Custom Project** → Embeds Run Monitor components
- **Run History** → New unified view

**Preserves**: All existing Train/Run menu entries during migration

---

*Detailed phase breakdown in sections below (to be expanded)*

---

## PHASE B: UNIFIED CORE MODELS (2-3 days)

### New Directory Structure

```
anylabeling/services/training_center/
├── models.py              # TrainingJob, JobStatus
├── job_manager.py         # Global training lock
├── process_utils.py       # Extracted duplicates
└── adapters/
    ├── base.py
    ├── ultralytics_adapter.py
    └── custom_script_adapter.py
```

### Adapter Pattern (Wraps Existing Code)

```python
class UltralyticsAdapter(AbstractTrainingAdapter):
    def __init__(self):
        self.manager = get_training_manager()  # Existing
    
    def start(self, config):
        return self.manager.start_training(config)  # Delegates
```

**Key**: Adapters CALL existing code, never replace

---

## PHASE C: TRAINING CENTER UI (3-4 days)

### Three-Tab Window

- **Tab 1: Guided Training** → Opens UltralyticsDialog
- **Tab 2: Custom Project** → Embeds Run Monitor widgets
- **Tab 3: Run History** → Unified view

### Integration

Minimal modification to label_widget.py: Add "Training Center" menu

---

## PHASE D: GLOBAL TRAINING LOCK (1 day)

Prevent concurrent training from both systems using JobManager singleton

---

## PHASE E: MENU TRANSITION (After Verification)

**Stage 1**: Keep both Train and Run menus  
**Stage 2**: Promote Training Center, deprecate Run  
**Stage 3**: Single Training Center entry

---

## FILE CHECKLIST

### Phase B (Create, Modify: NONE)
- [ ] services/training_center/*.py (8 new files)

### Phase C (Create + 1 modify)
- [ ] views/training_center/*.py (4 new files)
- [ ] Modify: label_widget.py (add menu entry)

### Phase D (Modify: 3 files)
- [ ] Add lock checks to ultralytics_dialog.py
- [ ] Add lock checks to run_monitor_window.py

---

## RISKS & ROLLBACK

**Low Risk**: Adapters only wrap, no deletions  
**Rollback**: Delete training_center/, restore menu

---

## SUCCESS CRITERIA

- [ ] Both training types work through adapters
- [ ] Global lock prevents conflicts
- [ ] All existing tests pass
- [ ] No user-reported regressions

---

*End of Plan*
