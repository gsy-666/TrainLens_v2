# Run Monitor Architecture

**Generated**: 2026-07-15  
**Purpose**: Technical architecture and implementation design for Run Monitor feature

---

## Overview

Run Monitor is a **new, independent module** that allows users to:
1. Open any training project workspace
2. Automatically detect training scripts
3. Detect Python environments
4. Execute training scripts in isolated subprocesses
5. Monitor real-time logs, resources (CPU/Memory/GPU), and training metrics
6. View structured training events and curves (when available)
7. Persist run history for later review

**Key Constraint**: Run Monitor must NOT interfere with existing X-AnyLabeling features, especially the existing Ultralytics training system.

---

## Architecture Principles

### 1. Complete Independence
- Run Monitor is a **separate QDialog window**, not embedded in Canvas or LabelingWidget
- Business logic lives in `services/run_monitor/`, completely isolated
- UI lives in `views/run_monitor/`, independent from labeling views
- Run Monitor errors never crash the main annotation window

### 2. Non-Invasive Integration
- **One new menu**: "Run" menu in LabelingWidget (alongside existing "Train" menu)
- **One new action**: "Run Monitor" menu item
- **One new method**: `open_run_monitor()` handler in LabelingWidget
- **Zero modifications** to Canvas, Shape, LabelFile, or auto-labeling services

### 3. Process Isolation
- Training scripts run in **independent subprocesses**
- Main Qt event loop never blocks
- Subprocess crashes don't crash GUI
- Use QThread for I/O operations (log reading, resource monitoring)

### 4. Data Integrity
- Never modify user's training project files
- Store Run Monitor metadata in `.trainlens/` subdirectory within workspace
- Gracefully handle missing permissions or disk failures

---

## Module Structure

```
anylabeling/
├── services/
│   └── run_monitor/                      # Business logic (NEW)
│       ├── __init__.py
│       ├── models.py                     # Data models (Run, Workspace, etc.)
│       ├── workspace_scanner.py          # Scan workspace for scripts
│       ├── script_detector.py            # Identify training scripts
│       ├── environment_detector.py       # Find Python/venv
│       ├── process_manager.py            # Manage training subprocess
│       ├── resource_monitor.py           # Monitor CPU/GPU/Memory
│       ├── run_storage.py                # Persist run history
│       ├── event_protocol.py             # Structured event definitions
│       └── metrics/                      # Metrics parsing (NEW)
│           ├── __init__.py
│           ├── base.py                   # Base metric reader
│           ├── jsonl_reader.py           # Read TrainLens JSONL
│           ├── ultralytics_reader.py     # Read Ultralytics results.csv
│           └── tensorboard_reader.py     # Read TensorBoard events
│
└── views/
    └── run_monitor/                      # UI layer (NEW)
        ├── __init__.py
        ├── run_monitor_window.py         # Main window (QDialog)
        ├── workspace_panel.py            # Workspace selector
        ├── run_configuration_panel.py    # Script/Python/Args selector
        ├── run_status_panel.py           # Status display
        ├── console_panel.py              # Real-time log viewer
        ├── resource_panel.py             # CPU/GPU charts
        ├── metrics_panel.py              # Training curves (optional)
        └── widgets/                      # Reusable widgets (NEW)
            ├── empty_state.py            # Empty state UI
            └── status_badge.py           # Status indicator
```

---

## Data Models

### Run
Represents a single training execution.

```python
@dataclass
class Run:
    run_id: str                    # Unique ID (e.g., run_20260715_143022)
    workspace_path: Path           # Workspace root directory
    script_path: Path              # Training script path
    python_path: Path              # Python executable used
    arguments: List[str]           # Command-line arguments
    framework: Optional[str]       # Detected framework (ultralytics, pytorch, etc.)
    start_time: datetime
    end_time: Optional[datetime]
    exit_code: Optional[int]
    status: RunStatus              # idle, preparing, running, completed, failed, stopped
    pid: Optional[int]             # Process ID
```

### RunStatus
```python
class RunStatus(Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
```

### Workspace
Represents a training project directory.

```python
@dataclass
class Workspace:
    path: Path
    detected_scripts: List[DetectedScript]
    detected_environments: List[PythonEnvironment]
    scan_timestamp: datetime
```

### DetectedScript
Represents a potential training script.

```python
@dataclass
class DetectedScript:
    path: Path
    framework: Optional[str]       # ultralytics, pytorch, lightning, etc.
    confidence: float              # 0.0 to 1.0
    reasons: List[str]             # Why this was identified as training script
```

### PythonEnvironment
Represents a Python installation or virtual environment.

```python
@dataclass
class PythonEnvironment:
    python_path: Path
    version: str                   # e.g., "3.11.5"
    env_type: str                  # system, venv, conda, uv
    env_path: Optional[Path]       # Path to venv root
    is_valid: bool                 # Can execute scripts
```

### TrainingEvent
Structured event emitted during training.

```python
@dataclass
class TrainingEvent:
    schema_version: int = 1
    run_id: str
    event: str                     # Event type
    timestamp: float               # Unix timestamp
    payload: Dict[str, Any]        # Event-specific data
```

**Event Types**:
- `run_created`: Run initialized
- `environment_detected`: Python environment found
- `process_started`: Training process started
- `console_output`: stdout/stderr line
- `batch_progress`: Batch-level progress (if available)
- `epoch_metrics`: Epoch-level metrics (loss, accuracy, etc.)
- `validation_started`: Validation phase started
- `validation_completed`: Validation results
- `checkpoint_saved`: Model checkpoint saved
- `resource_sample`: CPU/GPU/Memory sample
- `process_completed`: Training finished successfully
- `process_failed`: Training failed
- `process_stopped`: User stopped training

---

## Service Layer Design

### WorkspaceScanner

**Responsibility**: Scan a directory for training scripts and environments.

```python
class WorkspaceScanner:
    def scan(self, workspace_path: Path, 
             progress_callback: Optional[Callable] = None) -> Workspace:
        """
        Scan workspace directory.
        
        Returns:
            Workspace object with detected scripts and environments
        """
```

**Features**:
- Runs in background QThread to avoid blocking GUI
- Excludes: `.git`, `__pycache__`, `node_modules`, `build`, `dist`, `.venv*`
- Handles permission errors gracefully
- Emits progress signals for large directories
- Supports cancellation

### ScriptDetector

**Responsibility**: Determine if a Python file is a training script.

```python
class ScriptDetector:
    def detect(self, script_path: Path) -> Optional[DetectedScript]:
        """
        Analyze Python file to determine if it's a training script.
        
        Returns:
            DetectedScript if identified, None otherwise
        """
```

**Detection Heuristics**:
1. **Filename patterns**: `train.py`, `main.py`, `run.py`, `run_train.py`
2. **Import patterns**:
   - Ultralytics: `from ultralytics import YOLO`
   - PyTorch: `import torch`, `DataLoader`, `optimizer`
   - Lightning: `from pytorch_lightning import Trainer`
   - TensorFlow: `import tensorflow`, `model.fit`
3. **Code patterns**:
   - `model.train(`
   - `Trainer.fit(`
   - `for epoch in range(`
   - `optimizer.step(`
   - `loss.backward(`
4. **Entry point**: `if __name__ == "__main__"`
5. **CLI frameworks**: `argparse`, `click`, `typer`

**Confidence Scoring**:
- Filename match: +0.3
- Framework imports: +0.4
- Training loops: +0.3
- Entry point: +0.1
- Threshold: 0.5 for detection

**Safety**: Only static analysis, never execute unknown code.

### EnvironmentDetector

**Responsibility**: Find Python installations and virtual environments.

```python
class EnvironmentDetector:
    def detect(self, workspace_path: Path) -> List[PythonEnvironment]:
        """
        Find all Python environments in workspace.
        
        Returns:
            List of PythonEnvironment objects
        """
```

**Detection Strategy**:
1. Check workspace root for `.venv`, `venv`, `env`
2. Check for `pyproject.toml`, `requirements.txt`, `environment.yml`
3. Check for `uv.lock`, `poetry.lock`, `Pipfile.lock`
4. Check system Python (fallback)
5. Validate each environment (run `python --version`)

### ProcessManager

**Responsibility**: Execute training script in subprocess and manage its lifecycle.

```python
class ProcessManager(QObject):
    # Signals
    process_started = pyqtSignal(int)  # pid
    process_finished = pyqtSignal(int, int)  # pid, exit_code
    stdout_ready = pyqtSignal(str)  # log line
    stderr_ready = pyqtSignal(str)  # error line
    
    def start(self, run: Run) -> bool:
        """Start training subprocess"""
        
    def stop(self) -> bool:
        """Stop training subprocess (graceful then force)"""
        
    def is_running(self) -> bool:
        """Check if process is running"""
```

**Implementation Details**:
- Use `subprocess.Popen` with `stdout=PIPE`, `stderr=PIPE`
- Set `cwd` to workspace root
- Use selected Python executable
- Capture output with `QThread` + `QTextStream` (non-blocking)
- Emit stdout/stderr via Qt signals in real-time
- On Windows: Use `taskkill /F /T /PID` to kill process tree
- On Linux: Use `os.killpg(pgid, signal.SIGTERM)` then `SIGKILL`

**Error Handling**:
- `FileNotFoundError`: Python executable not found
- `PermissionError`: Script not executable
- `OSError`: Subprocess creation failed
- Emit signals for all errors, never crash

### ResourceMonitor

**Responsibility**: Monitor system and process resources in real-time.

```python
class ResourceMonitor(QObject):
    # Signals
    resource_sample = pyqtSignal(dict)  # Resource snapshot
    
    def start_monitoring(self, pid: int, interval_ms: int = 1000):
        """Start monitoring process resources"""
        
    def stop_monitoring(self):
        """Stop monitoring"""
```

**Metrics Collected**:
- **Process CPU**: `psutil.Process(pid).cpu_percent()`
- **Process Memory**: `psutil.Process(pid).memory_info().rss`
- **System CPU**: `psutil.cpu_percent(percpu=True)`
- **System Memory**: `psutil.virtual_memory()`
- **GPU Utilization**: `nvidia-smi --query-gpu=utilization.gpu --format=csv` (if available)
- **GPU Memory**: `nvidia-smi --query-gpu=memory.used,memory.total --format=csv`

**Sampling**:
- Default interval: 1000ms (1 second)
- Run in background QThread
- Emit `resource_sample` signal with dict payload
- Gracefully handle missing GPU

**Fallback**:
- If NVIDIA GPU not detected: Skip GPU metrics, no error

### RunStorage

**Responsibility**: Persist run metadata and events to disk.

```python
class RunStorage:
    def save_run(self, run: Run):
        """Save run metadata to <workspace>/.trainlens/runs/<run_id>/run.json"""
        
    def save_event(self, event: TrainingEvent):
        """Append event to <workspace>/.trainlens/runs/<run_id>/events.jsonl"""
        
    def save_console_line(self, run_id: str, line: str):
        """Append log line to <workspace>/.trainlens/runs/<run_id>/console.log"""
        
    def save_resource_sample(self, run_id: str, sample: dict):
        """Append resource sample to <workspace>/.trainlens/runs/<run_id>/resources.jsonl"""
```

**Directory Structure**:
```
<workspace>/
└── .trainlens/
    ├── workspace.json          # Workspace metadata
    └── runs/
        └── <run_id>/
            ├── run.json        # Run metadata
            ├── console.log     # Full console output
            ├── events.jsonl    # Structured training events
            ├── metrics.jsonl   # Training metrics (optional)
            └── resources.jsonl # Resource samples
```

**File Formats**:
- `run.json`: Single JSON object
- `events.jsonl`: One JSON object per line (JSON Lines)
- `console.log`: Plain text, UTF-8
- `resources.jsonl`: One JSON object per line

**Error Handling**:
- If `.trainlens/` cannot be created: Fallback to user config dir
- If write fails: Log error, emit warning, continue without persistence
- Never crash on I/O errors

---

## UI Layer Design

### RunMonitorWindow

**Type**: Independent `QDialog`  
**Parent**: LabelingWidget (for modality, but not embedded)  
**Lifecycle**: Created on first open, kept alive, shown/hidden on subsequent opens

**Layout**:
```
┌────────────────────────────────────────────────────────────────┐
│ Run Monitor                              [_][□][X]              │
├────────────────────────────────────────────────────────────────┤
│ Workspace: /path/to/project           [Open Workspace]  [Help] │
├─────────────────────┬──────────────────────────────────────────┤
│ Scripts & Envs      │ Status: Idle                             │
│                     ├──────────────────────────────────────────┤
│ ✓ train.py          │ Script: [train.py         ▼]            │
│   (ultralytics)     │ Python: [.venv/bin/python ▼]            │
│                     │ Args:   [                               ] │
│ • main.py           ├──────────────────────────────────────────┤
│   (pytorch)         │ Resources                                │
│                     │ CPU: ▓▓▓▓▓▓░░░░ 60%  GPU: ▓▓▓░░░░░░░ 30% │
│ Environments:       │ Mem: ▓▓▓▓░░░░░░ 40%  VMem: ▓▓░░░░░░░░ 20%│
│ • .venv (3.11.5)    ├──────────────────────────────────────────┤
│ • system (3.12.0)   │ Metrics (if available)                   │
│                     │ [Placeholder for curves]                 │
├─────────────────────┴──────────────────────────────────────────┤
│ Console                                                         │
│ [2026-07-15 14:30:22] Starting training...                     │
│ [2026-07-15 14:30:23] Epoch 1/100 - Loss: 0.123               │
│ ...                                                             │
├────────────────────────────────────────────────────────────────┤
│          [Open Workspace]   [Stop]   [Start Training]          │
└────────────────────────────────────────────────────────────────┘
```

**Panels**:
1. **Workspace Panel** (top): Shows current workspace, button to open new
2. **Scripts & Environments** (left): List detected scripts and Python envs
3. **Configuration** (right-top): Select script, Python, args
4. **Status** (right-top): Show current run status
5. **Resources** (right-middle): Real-time CPU/GPU/Memory charts
6. **Metrics** (right-bottom): Training curves (optional, only if metrics available)
7. **Console** (bottom): Full log output

### WorkspacePanel

**Responsibility**: Display current workspace and allow opening new workspace.

**Features**:
- Show workspace path
- "Open Workspace" button → `QFileDialog.getExistingDirectory()`
- "Scan" button → Trigger re-scan
- Progress indicator during scan

### RunConfigurationPanel

**Responsibility**: Allow user to select script, Python, and arguments.

**Widgets**:
- **Script Dropdown**: List detected scripts, show confidence and framework
- **Python Dropdown**: List detected Python environments, show version and type
- **Arguments LineEdit**: Free-form command-line arguments

**Validation**:
- Disable "Start Training" if script or Python not selected
- Show warning if Python environment invalid

### RunStatusPanel

**Responsibility**: Display current run status.

**States**:
- **Idle**: No run in progress, gray color
- **Preparing**: About to start, yellow color
- **Running**: Training in progress, green color, show elapsed time
- **Completed**: Finished successfully, blue color, show duration
- **Failed**: Errored, red color, show exit code
- **Stopped**: User stopped, orange color

**Widgets**:
- Status badge with color
- Elapsed time / total duration
- PID display
- Exit code (if finished)

### ConsolePanel

**Responsibility**: Display real-time log output.

**Features**:
- `QPlainTextEdit` with monospace font
- Auto-scroll to bottom (with sticky scroll)
- Color coding:
  - stdout: default color
  - stderr: red
  - timestamps: gray
- Search/filter (optional)
- Copy, clear, save to file buttons

### ResourcePanel

**Responsibility**: Display real-time resource usage.

**Charts** (using Matplotlib):
- CPU usage line chart (rolling window, last 60 seconds)
- Memory usage line chart
- GPU usage line chart (if available)
- GPU memory line chart (if available)

**Fallback**:
- If no GPU: Show only CPU and Memory, display "No GPU detected"

### MetricsPanel

**Responsibility**: Display training metrics (if available).

**Version 1** (Current Phase):
- Placeholder widget with message: "Metrics will be displayed when training script outputs structured events"
- OR: Simple line chart if metrics available

**Future Versions**:
- Multi-line charts (loss, accuracy, mAP, etc.)
- Training wheel visualization
- Stage progress indicator

---

## Integration with LabelingWidget

### Minimal Surgical Modification

**File**: `anylabeling/views/labeling/label_widget.py`

**Change 1: Add Run Menu** (line ~2012)
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
    train=self.menu(self.tr("Train")),  # EXISTING - keep
    run=self.menu(self.tr("Run")),      # NEW - add
    help=self.menu(self.tr("Help")),
    recent_files=QtWidgets.QMenu(self.tr("Open Recent")),
)
```

**Change 2: Create Run Monitor Action** (line ~1850, after other actions)
```python
# Run Monitor
run_monitor = action(
    self.tr("Run Monitor"),
    self.open_run_monitor,
    shortcuts.get("run_monitor", "Ctrl+Shift+R"),
    "run",  # Icon name (will add to resources)
    self.tr("Open Run Monitor for training script execution and monitoring"),
)
```

**Change 3: Populate Run Menu** (line ~2046, after train menu)
```python
utils.add_actions(self.menus.train, (ultralytics_train,))  # EXISTING
utils.add_actions(self.menus.run, (run_monitor,))          # NEW
```

**Change 4: Add Handler Method** (anywhere after `__init__`, e.g., line ~3500)
```python
def open_run_monitor(self):
    """Open Run Monitor window for training script execution and monitoring"""
    from anylabeling.views.run_monitor import RunMonitorWindow
    
    # Singleton pattern: reuse window if exists
    if not hasattr(self, '_run_monitor_window') or self._run_monitor_window is None:
        self._run_monitor_window = RunMonitorWindow(parent=self)
    
    self._run_monitor_window.show()
    self._run_monitor_window.raise_()
    self._run_monitor_window.activateWindow()
```

**Total Lines**: ~25 lines added, 0 lines modified, 0 lines deleted

---

## Event Protocol Design

### Structured Events

Training scripts can emit structured events to communicate with Run Monitor.

**Format**: JSON Lines (one JSON object per line) written to stdout

**Schema**:
```json
{
  "schema_version": 1,
  "run_id": "run_20260715_143022",
  "event": "epoch_metrics",
  "timestamp": 1721041822.123,
  "payload": {
    "epoch": 10,
    "loss": 0.0123,
    "accuracy": 0.985,
    "val_loss": 0.0145,
    "val_accuracy": 0.982
  }
}
```

**Detection**: Lines starting with `{"schema_version":` are parsed as events

### Metrics Sources (Priority Order)

1. **TrainLens JSON Events** (stdout): Native structured events
2. **Workspace metrics.jsonl**: User script writes metrics
3. **Ultralytics results.csv**: Read from Ultralytics runs
4. **TensorBoard Events**: Parse TensorBoard event files
5. **Framework-Specific Adapters**: Known framework log patterns

### No Metrics Fallback

If no structured metrics available:
- Display console output only
- Show resources only
- Display message: "This script does not output structured training metrics. Add TrainLens event logging to see real-time curves."

---

## Error Handling Strategy

### Graceful Degradation

| Error | Handling | User Impact |
|-------|----------|-------------|
| Workspace scan fails | Show error, allow retry | Can't auto-detect scripts |
| Script detection fails | Skip that file, continue | Fewer detected scripts |
| Python detection fails | Warn, allow manual entry | Can manually enter Python path |
| Subprocess creation fails | Show error dialog, stay in idle | Training doesn't start, can retry |
| Training script crashes | Show exit code, preserve logs | Can review logs to debug |
| Resource monitoring fails | Disable resource panel | Training continues, no resource display |
| File write fails | Warn, continue without persistence | Run not saved, but training continues |
| GPU not found | Hide GPU metrics | CPU/Memory monitoring still works |

### Never Crash Main App

- All Run Monitor code wrapped in try-except
- Exceptions logged but never propagated to LabelingWidget
- If Run Monitor window errors: Close window, show error dialog, main app unaffected

---

## Testing Strategy

### Unit Tests

**Location**: `tests/trainlens/`

**Coverage**:
- `test_script_detector.py`: Test script detection with fixture scripts
- `test_environment_detector.py`: Test Python environment detection
- `test_process_manager.py`: Test subprocess lifecycle (with mock script)
- `test_run_storage.py`: Test file I/O and JSONL writing
- `test_event_protocol.py`: Test event parsing and validation
- `test_resource_monitor.py`: Test resource sampling (mock psutil)

### Integration Tests

- Test full workflow: Open workspace → Detect → Configure → Start → Monitor → Stop
- Test with real short-lived training scripts (e.g., 5-second dummy script)
- Test with invalid workspace, missing Python, broken script
- Test resource monitoring with real subprocess
- Test graceful shutdown with running process

### Regression Tests

- Run existing X-AnyLabeling tests: `pytest tests/`
- Verify all tests still pass
- Verify no new warnings or errors

### Manual Testing

- Use MANUAL_REGRESSION_CHECKLIST.md (Section 15: TrainLens Integration Tests)
- Test on both Windows and Linux
- Test with and without GPU
- Test with Ultralytics training and Run Monitor open simultaneously

---

## Performance Considerations

### Workspace Scanning
- **Problem**: Large workspaces (10K+ files) slow to scan
- **Solution**: 
  - Run in background thread
  - Emit incremental results
  - Skip large directories (node_modules, .git)
  - Support cancellation

### Resource Monitoring
- **Problem**: Polling every 1 second adds overhead
- **Solution**:
  - Use efficient psutil calls
  - Run in separate thread
  - Allow user to adjust interval (1-10 seconds)

### Log Streaming
- **Problem**: High-frequency log output floods GUI
- **Solution**:
  - Buffer lines (batch updates every 100ms)
  - Limit console to last 10,000 lines
  - Provide "save full log" button

### Memory Usage
- **Problem**: Long-running training accumulates log data
- **Solution**:
  - Write logs to file immediately
  - Keep only recent N lines in memory
  - Clear old resource samples after display window passes

---

## Future Enhancements (Not in Phase 1)

### Phase 2: Auto-Create Project Environment
- Detect `requirements.txt`, `pyproject.toml`
- Offer to create `.venv` and install dependencies
- Use `uv` for fast environment creation

### Phase 3: Training Curves and Wheel
- Implement interactive Matplotlib charts with zoom/pan
- Implement training stage wheel animation
- Support multi-run comparison

### Phase 4: Dataset Integration
- Link Run Monitor to annotation workspace
- One-click dataset export for training
- Auto-reload trained model in Auto-Labeling

### Phase 5: Distributed Training
- Support multi-GPU training monitoring
- Support multi-node training (if possible)

---

## Summary

**Architecture**: Clean separation of concerns, independent module, process isolation  
**Integration**: Minimal 25-line surgical modification to LabelingWidget  
**Safety**: Comprehensive error handling, graceful degradation, never crash main app  
**Testing**: Unit tests, integration tests, manual regression checklist  
**Performance**: Background threads, efficient polling, memory limits  

**Next Step**: Implement Phase 1 with this architecture
