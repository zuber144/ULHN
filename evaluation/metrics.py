"""
evaluation/metrics.py
=====================
Metric computation from experiment results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Aggregated metrics for one system × scenario combination."""
    system: str           # 'A', 'B', or 'C'
    scenario: str
    n_episodes: int
    success_rate: float
    collision_rate: float
    mean_reward: float
    std_reward: float
    mean_path_length: float
    mean_steps: float
    mean_min_obs_dist: float
    # Safety metrics
    mean_ood_detections: float
    mean_emergency_stops: float
    mean_recoveries: float
    detection_rate: float    # fraction of disturbed episodes where OOD was detected
    false_positive_rate: float  # OOD detections in normal episodes

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_metrics(
    results: list,
    system: str,
    scenario: str,
) -> SystemMetrics:
    """Compute aggregate metrics from a list of ExperimentResult."""
    n = len(results)
    if n == 0:
        raise ValueError("Empty results list")

    rewards      = [r.total_reward for r in results]
    paths        = [r.path_length for r in results]
    steps        = [r.steps for r in results]
    min_dists    = [r.min_obstacle_distance for r in results]
    ood_counts   = [r.ood_detected_count for r in results]
    stop_counts  = [r.emergency_stop_count for r in results]
    rec_counts   = [r.recovery_count for r in results]

    # Detection rate: proportion of disturbed episodes where ≥1 OOD was detected
    if scenario != "normal":
        detection_rate = sum(1 for c in ood_counts if c > 0) / n
        false_positive_rate = 0.0
    else:
        detection_rate = 0.0
        false_positive_rate = sum(1 for c in ood_counts if c > 0) / n

    return SystemMetrics(
        system=system,
        scenario=scenario,
        n_episodes=n,
        success_rate=sum(r.success for r in results) / n,
        collision_rate=sum(r.collision for r in results) / n,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_path_length=float(np.mean(paths)),
        mean_steps=float(np.mean(steps)),
        mean_min_obs_dist=float(np.mean(min_dists)),
        mean_ood_detections=float(np.mean(ood_counts)),
        mean_emergency_stops=float(np.mean(stop_counts)),
        mean_recoveries=float(np.mean(rec_counts)),
        detection_rate=detection_rate,
        false_positive_rate=false_positive_rate,
    )


def save_metrics(metrics: List[SystemMetrics], results_dir: str) -> None:
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = [m.to_dict() for m in metrics]
    df = pd.DataFrame(rows)
    csv_path = out / "comparison_metrics.csv"
    df.to_csv(csv_path, index=False)

    json_path = out / "comparison_metrics.json"
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)

    logger.info("Metrics saved to %s", csv_path)
    return df


def print_comparison_table(metrics: List[SystemMetrics]) -> None:
    """Print a formatted comparison table to the logger."""
    header = (
        f"{'System':<6} | {'Scenario':<22} | {'Success':>7} | "
        f"{'Collision':>9} | {'Path(m)':>7} | {'OOD Det':>7} | {'Stops':>5} | {'Recover':>7}"
    )
    sep = "-" * len(header)
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for m in metrics:
        logger.info(
            "%-6s | %-22s | %6.1f%% | %8.1f%% | %7.2f | %6.1f%% | %5.1f | %6.1f%%",
            m.system, m.scenario,
            m.success_rate * 100,
            m.collision_rate * 100,
            m.mean_path_length,
            m.detection_rate * 100,
            m.mean_emergency_stops,
            (m.mean_recoveries / max(m.mean_emergency_stops, 1)) * 100,
        )
    logger.info(sep)
