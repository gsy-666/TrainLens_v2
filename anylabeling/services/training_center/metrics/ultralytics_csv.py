"""Training Center - Ultralytics CSV Reader

Incremental reader for Ultralytics results.csv with column normalization.
"""

import csv
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from .models import MetricPoint, MetricsSnapshot


class UltralyticsCSVReader:
    """Incremental reader for Ultralytics results.csv
    
    Features:
    - Incremental reading (only new lines)
    - Column name normalization
    - Graceful handling of incomplete writes
    - File truncation detection
    """
    
    COLUMN_MAPPINGS = {
        # Loss columns
        'train/box_loss': ['train/box_loss', 'train/box loss'],
        'train/cls_loss': ['train/cls_loss', 'train/cls loss'],
        'train/dfl_loss': ['train/dfl_loss', 'train/dfl loss'],
        'val/box_loss': ['val/box_loss', 'val/box loss'],
        'val/cls_loss': ['val/cls_loss', 'val/cls loss'],
        'val/dfl_loss': ['val/dfl_loss', 'val/dfl loss'],
        
        # Metrics
        'metrics/precision': ['metrics/precision(B)', 'metrics/precision', 'precision(B)', 'precision'],
        'metrics/recall': ['metrics/recall(B)', 'metrics/recall', 'recall(B)', 'recall'],
        'metrics/mAP50': ['metrics/mAP50(B)', 'metrics/mAP50', 'mAP50(B)', 'mAP50'],
        'metrics/mAP50-95': ['metrics/mAP50-95(B)', 'metrics/mAP50-95', 'mAP50-95(B)', 'mAP50-95'],
        
        # Learning rates
        'lr/pg0': ['lr/pg0'],
        'lr/pg1': ['lr/pg1'],
        'lr/pg2': ['lr/pg2'],
        
        # Epoch
        'epoch': ['epoch'],
    }
    
    def __init__(self, csv_path: Path, job_id: str):
        self.csv_path = csv_path
        self.job_id = job_id
        self.last_size = 0
        self.last_line_count = 0
        self.column_map = None
        self.headers = []
        
    def read_new(self) -> List[MetricPoint]:
        if not self.csv_path.exists():
            return []
            
        try:
            current_size = self.csv_path.stat().st_size
            
            if current_size < self.last_size:
                self.last_size = 0
                self.last_line_count = 0
                self.column_map = None
                
            if current_size == self.last_size:
                return []
                
            points = []
            
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                if self.column_map is None and reader.fieldnames:
                    self.headers = [h.strip() for h in reader.fieldnames]
                    self.column_map = self._build_column_map(self.headers)
                    
                for i, row in enumerate(reader):
                    if i < self.last_line_count:
                        continue
                        
                    row_points = self._parse_row(row, i)
                    points.extend(row_points)
                    self.last_line_count = i + 1
                    
            self.last_size = current_size
            return points
            
        except (IOError, csv.Error, UnicodeDecodeError):
            return []
            
    def _build_column_map(self, headers: List[str]) -> Dict[str, str]:
        col_map = {}
        headers_stripped = [h.strip() for h in headers]
        
        for normalized, variants in self.COLUMN_MAPPINGS.items():
            for variant in variants:
                if variant in headers_stripped:
                    col_map[normalized] = variant
                    break
                    
        return col_map
        
    def _parse_row(self, row: Dict[str, str], row_index: int) -> List[MetricPoint]:
        if not self.column_map:
            return []
            
        points = []
        timestamp = time.time()
        
        epoch = None
        if 'epoch' in self.column_map:
            try:
                epoch = int(float(row[self.column_map['epoch']].strip()))
            except (ValueError, KeyError):
                pass
                
        for normalized, actual in self.column_map.items():
            if normalized == 'epoch':
                continue
                
            try:
                value_str = row[actual].strip()
                if value_str:
                    value = float(value_str)
                    points.append(MetricPoint(
                        job_id=self.job_id,
                        step=row_index,
                        epoch=epoch,
                        timestamp=timestamp,
                        name=normalized,
                        value=value,
                        source="ultralytics_csv",
                    ))
            except (ValueError, KeyError):
                continue
                
        return points
        
    def reset(self):
        self.last_size = 0
        self.last_line_count = 0
        self.column_map = None
        self.headers = []
