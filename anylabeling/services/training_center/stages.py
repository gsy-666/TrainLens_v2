"""Training Center - Training Stages

Event-driven stage tracking for training lifecycle.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Callable
import time


class TrainingStage(Enum):
    """Training lifecycle stages"""
    ENVIRONMENT = "environment"
    DATA_VALIDATION = "data_validation"
    DATA_PREPARATION = "data_preparation"
    MODEL_LOADING = "model_loading"
    TRAINING = "training"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"
    RESULTS = "results"
    EXPORT = "export"
    COMPLETED = "completed"


class StageStatus(Enum):
    """Stage execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageInfo:
    """Stage information"""
    stage: TrainingStage
    status: StageStatus
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    message: str = ""


class StageTracker:
    """Tracks training stages
    
    Stages transition based on real events, not auto-completion.
    """
    
    def __init__(self):
        """Initialize tracker"""
        self.stages: List[StageInfo] = []
        self._init_stages()
        self._callbacks: List[Callable[[StageInfo], None]] = []
        
    def _init_stages(self):
        """Initialize all stages as pending"""
        for stage in TrainingStage:
            self.stages.append(StageInfo(
                stage=stage,
                status=StageStatus.PENDING,
            ))
            
    def start_stage(self, stage: TrainingStage, message: str = ""):
        """Mark stage as running"""
        for info in self.stages:
            if info.stage == stage:
                info.status = StageStatus.RUNNING
                info.started_at = time.time()
                info.message = message
                self._notify(info)
                break
                
    def complete_stage(self, stage: TrainingStage, message: str = ""):
        """Mark stage as completed"""
        for info in self.stages:
            if info.stage == stage:
                info.status = StageStatus.COMPLETED
                info.completed_at = time.time()
                info.message = message
                self._notify(info)
                break
                
    def fail_stage(self, stage: TrainingStage, message: str = ""):
        """Mark stage as failed"""
        for info in self.stages:
            if info.stage == stage:
                info.status = StageStatus.FAILED
                info.completed_at = time.time()
                info.message = message
                self._notify(info)
                break
                
    def warn_stage(self, stage: TrainingStage, message: str = ""):
        """Mark stage with warning"""
        for info in self.stages:
            if info.stage == stage:
                info.status = StageStatus.WARNING
                info.message = message
                self._notify(info)
                break
                
    def skip_stage(self, stage: TrainingStage, message: str = ""):
        """Mark stage as skipped"""
        for info in self.stages:
            if info.stage == stage:
                info.status = StageStatus.SKIPPED
                info.message = message
                self._notify(info)
                break
                
    def get_current_stage(self) -> Optional[StageInfo]:
        """Get currently running stage"""
        for info in self.stages:
            if info.status == StageStatus.RUNNING:
                return info
        return None
        
    def get_stage_info(self, stage: TrainingStage) -> Optional[StageInfo]:
        """Get info for specific stage"""
        for info in self.stages:
            if info.stage == stage:
                return info
        return None
        
    def subscribe(self, callback: Callable[[StageInfo], None]):
        """Subscribe to stage changes"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
            
    def unsubscribe(self, callback: Callable[[StageInfo], None]):
        """Unsubscribe from stage changes"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            
    def _notify(self, stage_info: StageInfo):
        """Notify subscribers of stage change"""
        for callback in self._callbacks[:]:
            try:
                callback(stage_info)
            except Exception:
                pass
                
    def reset(self):
        """Reset all stages"""
        self.stages.clear()
        self._init_stages()
