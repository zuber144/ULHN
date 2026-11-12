"""
watchdog/threshold.py
=====================
Statistical threshold computation for the anomaly detector.

The threshold is derived from normal validation data so it is not an
arbitrary constant — it reflects the actual reconstruction error
distribution of the trained Autoencoder on in-distribution inputs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ThresholdStats:
    """Statistics derived from normal validation scores."""
    mean: float
    std: float
    p50: float
    p90: float
    p95: float
    p99: float
    p995: float
    n_samples: int


def compute_threshold_stats(
    normal_scores: np.ndarray,
) -> ThresholdStats:
    """
    Compute threshold statistics from an array of anomaly scores on
    normal (in-distribution) data.

    Parameters
    ----------
    normal_scores:
        1-D array of MSE reconstruction errors on normal LiDAR observations.

    Returns
    -------
    ThresholdStats dataclass with various percentile thresholds.
    """
    return ThresholdStats(
        mean=float(np.mean(normal_scores)),
        std=float(np.std(normal_scores)),
        p50=float(np.percentile(normal_scores, 50)),
        p90=float(np.percentile(normal_scores, 90)),
        p95=float(np.percentile(normal_scores, 95)),
        p99=float(np.percentile(normal_scores, 99)),
        p995=float(np.percentile(normal_scores, 99.5)),
        n_samples=len(normal_scores),
    )


def select_threshold(
    stats: ThresholdStats,
    percentile: float = 95.0,
) -> float:
    """
    Select a threshold from ThresholdStats at a given percentile.

    Parameters
    ----------
    stats:
        Computed from normal validation data.
    percentile:
        Target percentile (e.g. 95.0).

    Returns
    -------
    Threshold value (float).
    """
    mapping = {
        50.0:  stats.p50,
        90.0:  stats.p90,
        95.0:  stats.p95,
        99.0:  stats.p99,
        99.5:  stats.p995,
    }
    if percentile in mapping:
        threshold = mapping[percentile]
    else:
        # Fallback: use mean + z * std (approximate)
        from scipy import stats as scipy_stats
        z = scipy_stats.norm.ppf(percentile / 100.0)
        threshold = stats.mean + z * stats.std
        logger.warning(
            "Percentile %.1f not pre-computed; using mean+%.2f*std", percentile, z
        )

    logger.info(
        "Threshold @ p%.0f = %.6f  (mean=%.6f, std=%.6f)",
        percentile, threshold, stats.mean, stats.std,
    )
    return threshold


def save_threshold_stats(stats: ThresholdStats, path: str) -> None:
    """Persist threshold statistics to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(asdict(stats), f, indent=2)
    logger.info("Threshold stats saved to %s", p)


def load_threshold_stats(path: str) -> ThresholdStats:
    """Load previously computed threshold statistics."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Threshold stats not found: {p.resolve()}. "
            "Run: python main.py train-watchdog"
        )
    with p.open() as f:
        d = json.load(f)
    return ThresholdStats(**d)
