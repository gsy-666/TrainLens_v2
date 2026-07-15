# Training System Audit

**Date**: 2026-07-15  
**Auditor**: Claude  
**Purpose**: Complete analysis of existing Ultralytics training vs new Run Monitor

---

## Systems Overview

### System 1: Ultralytics Training (Existing)
- **Entry**: `anylabeling/views/training/ultralytics_dialog.py` (2046 lines)
- **Service**: `anylabeling/services/auto_training/ultralytics/trainer.py`
- **Purpose**: Integrated Ultralytics YOLO training with dataset creation

### System 2: Run Monitor (New)
- **Entry**: `anylabeling/views/run_monitor/run_monitor_window.py` (601 lines)
- **Service**: `anylabeling/services/run_monitor/process_manager.py`
- **Purpose**: Generic training script executor with monitoring

---

## Call Chain Summary

**Ultralytics Flow**: Annotation Data → Dataset Creation → train-worker subprocess → YOLO.train()

**Run Monitor Flow**: User Workspace → Script Detection → Direct subprocess → Generic execution

---

## Key Findings

1. **60% Feature Overlap** - Both manage subprocess, capture output, terminate process tree
2. **Different Purposes** - Ultralytics: tightly integrated; Run Monitor: framework-agnostic
3. **4 Duplicate Implementations** - Process termination, log redirection, event parsing, subprocess management
4. **10 Unique Ultralytics Features** - Dataset creation, YOLO conversion, model export, training images
5. **9 Unique Run Monitor Features** - Workspace scanning, script detection, resource monitoring, run persistence

---

*Detailed analysis continues in TRAINING_CAPABILITY_MATRIX.md*

---

## SYSTEM 1: ULTRALYTICS COMPLETE CALL CHAIN

### Entry: label_widget.py:3259 → UltralyticsDialog

**Integration**: Modal dialog, receives image_list + output_dir from parent

### Dataset Creation: general.py:17-272

1. LabelConverter loads annotation format
2. Creates temp dir: `datasets/{task}/{name}_{timestamp}`
3. Splits train/val by ratio
4. Converts JSON → YOLO txt format
5. Generates data.yaml

### Training: trainer.py:70-186

1. Creates temp payload.json
2. Builds: `[python, -m, anylabeling.app, train-worker, --payload, <path>]`
3. Spawns subprocess with process group
4. Reads stdout, parses events with prefix `__XANYLABELING_TRAIN_EVENT__=`

### Progress: results.csv polling every 1s

### Termination: taskkill /T or killpg with SIGKILL

---

## SYSTEM 2: RUN MONITOR COMPLETE CALL CHAIN

### Entry: label_widget.py:3329 → RunMonitorWindow

**Integration**: Non-modal, independent, no data passed

### Workspace Scan: workspace_scanner_thread.py (QThread)

1. Detects Python envs (.venv, venv, system)
2. Finds *.py files (max 10000, excludes .git, __pycache__)
3. Script detection via heuristics (confidence >= 0.5)

### Execution: process_manager.py

Direct subprocess: `[python, script.py, *args]`
Output: Separate QThreads read stdout/stderr line-by-line

### Monitoring: resource_monitor.py

Collects CPU/Memory/GPU every 1s via psutil + nvidia-smi

### Storage: .trainlens/runs/<run_id>/

run.json, console.log, events.jsonl, resources.jsonl

---

## DUPLICATE CODE TO UNIFY

1. **Process Termination** - Identical taskkill/killpg (extract to utils)
2. **Log Redirection** - Similar QObject+signal (create base class)
3. **Event Parsing** - Different protocols, same interface concept

## NON-REPLACEABLE FEATURES

**Ultralytics-Only**: Dataset creation, YOLO conversion, export, training images
**Run Monitor-Only**: Workspace scan, script detection, resource monitoring, run storage

## RECOMMENDATION

**Training Center Shell** with adapters, preserve both implementations

