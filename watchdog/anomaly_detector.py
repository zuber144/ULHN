"""
watchdog/anomaly_detector.py
============================
Stateful anomaly detector that wraps the Autoencoder and threshold.

This is the OOD watchdog component used by the Safety Controller.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np

from watchdog.autoencoder import LiDARAutoencoder
from environment.lidar import select_forward_sector
from watchdog.threshold import ThresholdStats, select_threshold

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Wraps the Autoencoder and threshold to provide a clean
    ``is_ood(lidar_obs)`` interface to the Safety Controller.

    Supports an optional moving-average smoothing of the anomaly score
    to reduce false positives from single noisy observations.
    """

    def __init__(
        self,
        model: LiDARAutoencoder,
        threshold: float,
        smoothing_window: int = 3,
        forward_sector_fraction: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Trained and loaded LiDARAutoencoder.
        threshold:
            Anomaly score above which OOD is declared.
        smoothing_window:
            Number of recent scores to average before comparing to threshold.
            Set to 1 to disable smoothing.
        """
        self.model = model
        self.threshold = threshold
        self.forward_sector_fraction = forward_sector_fraction
        self.smoothing_window = max(1, smoothing_window)
        self._score_buffer: deque = deque(maxlen=self.smoothing_window)
        self._raw_score: float = 0.0
        self._smoothed_score: float = 0.0

    def update(self, lidar_obs: np.ndarray) -> float:
        """
        Process a LiDAR observation and return the (smoothed) anomaly score.

        Parameters
        ----------
        lidar_obs:
            Normalised LiDAR array of shape (num_rays,).

        Returns
        -------
        float — smoothed anomaly score.
        """
        raw = self.score(lidar_obs)
        self._raw_score = raw
        self._score_buffer.append(raw)
        self._smoothed_score = float(np.mean(self._score_buffer))
        return self._smoothed_score

    def score(self, lidar_obs: np.ndarray) -> float:
        """Score only the forward-facing LiDAR sector, without smoothing."""
        forward_obs = select_forward_sector(
            lidar_obs, self.forward_sector_fraction
        )
        return self.model.anomaly_score_numpy(forward_obs)

    def is_ood(self, lidar_obs: Optional[np.ndarray] = None) -> bool:
        """
        Returns True if the current smoothed anomaly score exceeds threshold.

        If *lidar_obs* is provided, ``update()`` is called first.
        """
        if lidar_obs is not None:
            self.update(lidar_obs)
        return self._smoothed_score > self.threshold

    def reset(self) -> None:
        """Clear the score buffer (call at episode start)."""
        self._score_buffer.clear()
        self._raw_score = 0.0
        self._smoothed_score = 0.0

    @property
    def raw_score(self) -> float:
        return self._raw_score

    @property
    def smoothed_score(self) -> float:
        return self._smoothed_score

    @classmethod
    def from_stats(
        cls,
        model: LiDARAutoencoder,
        stats: ThresholdStats,
        percentile: float = 95.0,
        smoothing_window: int = 3,
        forward_sector_fraction: float = 0.5,
    ) -> "AnomalyDetector":
        """Convenience constructor that derives threshold from ThresholdStats."""
        threshold = select_threshold(stats, percentile)
        return cls(model, threshold, smoothing_window, forward_sector_fraction)
