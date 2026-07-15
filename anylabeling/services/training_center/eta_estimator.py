"""Training Center - ETA Estimator

Estimates training completion time using weighted moving average of epoch durations.
Handles first-epoch initialization overhead, outlier detection, and confidence scoring.
"""

import time
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class ETAConfidence(Enum):
    """Confidence level for ETA estimates"""
    INSUFFICIENT = "insufficient"  # Less than 2 epochs
    LOW = "low"                    # 2-3 epochs
    MEDIUM = "medium"              # 4-5 epochs
    HIGH = "high"                  # 6+ epochs with low variance


@dataclass
class ETAEstimate:
    """ETA estimation result"""
    eta_seconds: Optional[float]
    estimated_finish: Optional[float]  # Unix timestamp
    confidence: ETAConfidence
    avg_epoch_time: Optional[float]
    epochs_completed: int
    epochs_remaining: Optional[int]
    is_stale: bool = False


class ETAEstimator:
    """Estimates training completion time

    Uses weighted moving average with:
    - Outlier detection via median-based filtering
    - First epoch exclusion (initialization overhead)
    - Recent epoch weighting
    - Variance-based confidence
    """

    def __init__(self, window_size: int = 5):
        """Initialize estimator

        Args:
            window_size: Number of recent epochs to consider
        """
        self.window_size = window_size
        self.epoch_times: List[Tuple[int, float, float]] = []  # (epoch, start_time, duration)
        self.total_epochs: Optional[int] = None
        self.training_start_time: Optional[float] = None
        self.last_update_time: Optional[float] = None
        self.stale_threshold: float = 600.0  # 10 minutes

    def set_total_epochs(self, total: int):
        """Set total epochs for the training run"""
        self.total_epochs = total

    def record_epoch_start(self, epoch: int, timestamp: Optional[float] = None):
        """Record when an epoch started"""
        if timestamp is None:
            timestamp = time.time()

        if self.training_start_time is None:
            self.training_start_time = timestamp

        self.last_update_time = timestamp

    def record_epoch_complete(self, epoch: int, timestamp: Optional[float] = None):
        """Record when an epoch completed

        Args:
            epoch: Epoch number (0-indexed or 1-indexed)
            timestamp: Unix timestamp of completion
        """
        if timestamp is None:
            timestamp = time.time()

        # Calculate epoch duration
        if len(self.epoch_times) > 0:
            last_epoch, last_start, _ = self.epoch_times[-1]
            if last_epoch == epoch - 1 or last_epoch == epoch:
                # Found matching start
                duration = timestamp - last_start
                self.epoch_times[-1] = (last_epoch, last_start, duration)
            else:
                # No matching start, estimate from last completion
                if len(self.epoch_times) >= 2:
                    _, prev_start, prev_duration = self.epoch_times[-1]
                    est_duration = prev_duration
                    est_start = timestamp - est_duration
                    self.epoch_times.append((epoch, est_start, est_duration))
                else:
                    # First epoch, use elapsed time
                    if self.training_start_time:
                        duration = timestamp - self.training_start_time
                        self.epoch_times.append((epoch, self.training_start_time, duration))
        else:
            # First epoch completion
            if self.training_start_time:
                duration = timestamp - self.training_start_time
                self.epoch_times.append((epoch, self.training_start_time, duration))

        self.last_update_time = timestamp

    def estimate(self) -> ETAEstimate:
        """Calculate ETA estimate

        Returns:
            ETAEstimate with time remaining and confidence
        """
        epochs_completed = len(self.epoch_times)

        # Insufficient data
        if epochs_completed < 2:
            return ETAEstimate(
                eta_seconds=None,
                estimated_finish=None,
                confidence=ETAConfidence.INSUFFICIENT,
                avg_epoch_time=None,
                epochs_completed=epochs_completed,
                epochs_remaining=None,
            )

        # Check if stale
        is_stale = False
        if self.last_update_time:
            time_since_update = time.time() - self.last_update_time
            is_stale = time_since_update > self.stale_threshold

        # Unknown total epochs
        if self.total_epochs is None:
            return ETAEstimate(
                eta_seconds=None,
                estimated_finish=None,
                confidence=ETAConfidence.INSUFFICIENT,
                avg_epoch_time=self._calculate_avg_epoch_time(),
                epochs_completed=epochs_completed,
                epochs_remaining=None,
                is_stale=is_stale,
            )

        epochs_remaining = self.total_epochs - epochs_completed
        if epochs_remaining <= 0:
            # Training complete
            return ETAEstimate(
                eta_seconds=0.0,
                estimated_finish=time.time(),
                confidence=ETAConfidence.HIGH,
                avg_epoch_time=self._calculate_avg_epoch_time(),
                epochs_completed=epochs_completed,
                epochs_remaining=0,
                is_stale=False,
            )

        # Calculate weighted average epoch time
        avg_epoch_time = self._calculate_weighted_avg()

        if avg_epoch_time is None:
            return ETAEstimate(
                eta_seconds=None,
                estimated_finish=None,
                confidence=ETAConfidence.INSUFFICIENT,
                avg_epoch_time=None,
                epochs_completed=epochs_completed,
                epochs_remaining=epochs_remaining,
                is_stale=is_stale,
            )

        # Calculate ETA
        eta_seconds = epochs_remaining * avg_epoch_time
        estimated_finish = time.time() + eta_seconds

        # Determine confidence
        confidence = self._calculate_confidence(epochs_completed)

        return ETAEstimate(
            eta_seconds=eta_seconds,
            estimated_finish=estimated_finish,
            confidence=confidence,
            avg_epoch_time=avg_epoch_time,
            epochs_completed=epochs_completed,
            epochs_remaining=epochs_remaining,
            is_stale=is_stale,
        )

    def _calculate_avg_epoch_time(self) -> Optional[float]:
        """Calculate simple average of all epoch times"""
        durations = [d for _, _, d in self.epoch_times if d > 0]
        if not durations:
            return None
        return sum(durations) / len(durations)

    def _calculate_weighted_avg(self) -> Optional[float]:
        """Calculate weighted moving average of recent epochs

        Excludes first epoch (initialization overhead).
        Uses outlier detection and recency weighting.
        """
        if len(self.epoch_times) < 2:
            return None

        # Exclude first epoch (initialization overhead)
        durations = [d for _, _, d in self.epoch_times[1:] if d > 0]

        if not durations:
            # Fallback to all epochs if excluding first leaves nothing
            durations = [d for _, _, d in self.epoch_times if d > 0]

        if not durations:
            return None

        # Take most recent window
        recent_durations = durations[-self.window_size:]

        if len(recent_durations) < 2:
            return sum(recent_durations) / len(recent_durations)

        # Outlier detection using median
        median = sorted(recent_durations)[len(recent_durations) // 2]
        mad = sum(abs(d - median) for d in recent_durations) / len(recent_durations)

        # Filter outliers (> 3 MAD from median)
        if mad > 0:
            threshold = 3 * mad
            filtered = [d for d in recent_durations if abs(d - median) < threshold]
            if filtered:
                recent_durations = filtered

        # Weighted average (more recent = higher weight)
        weights = list(range(1, len(recent_durations) + 1))
        weighted_sum = sum(d * w for d, w in zip(recent_durations, weights))
        weight_sum = sum(weights)

        return weighted_sum / weight_sum

    def _calculate_confidence(self, epochs_completed: int) -> ETAConfidence:
        """Calculate confidence based on epochs and variance"""
        if epochs_completed < 2:
            return ETAConfidence.INSUFFICIENT
        elif epochs_completed <= 3:
            return ETAConfidence.LOW
        elif epochs_completed <= 5:
            return ETAConfidence.MEDIUM
        else:
            # Check variance for high confidence
            durations = [d for _, _, d in self.epoch_times[1:] if d > 0]
            if len(durations) >= 3:
                avg = sum(durations) / len(durations)
                variance = sum((d - avg) ** 2 for d in durations) / len(durations)
                cv = (variance ** 0.5) / avg if avg > 0 else 1.0

                # Coefficient of variation < 0.2 indicates stable timing
                if cv < 0.2:
                    return ETAConfidence.HIGH

            return ETAConfidence.MEDIUM

    def reset(self):
        """Reset all state"""
        self.epoch_times.clear()
        self.total_epochs = None
        self.training_start_time = None
        self.last_update_time = None
