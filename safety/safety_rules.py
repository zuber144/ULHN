"""
safety/safety_rules.py
======================
Configurable deterministic safety rules.

Rules are evaluated in order; the first matching rule wins.
The SafetyController applies these rules — it is NOT a neural network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SafetyConfig:
    """Configurable thresholds for the safety controller."""
    critical_distance: float = 0.15   # metres — triggers STOP
    slowdown_distance: float = 0.5    # metres — triggers SLOWDOWN
    slowdown_factor:   float = 0.5    # velocity multiplier during SLOWDOWN
    recovery_speed:    float = 0.2    # m/s for re-orientation
    resume_consecutive: int  = 10     # consecutive normal readings before RESUME
    stop_requires_ood:  bool = False  # proximity STOP only fires when OOD also active
    ood_confirmation_steps: int = 3   # consecutive readings required to confirm OOD


def load_safety_config(cfg_dict: dict) -> SafetyConfig:
    """Build SafetyConfig from the 'safety' section of config.yaml."""
    return SafetyConfig(
        critical_distance=cfg_dict.get("critical_distance", 0.15),
        slowdown_distance=cfg_dict.get("slowdown_distance", 0.5),
        slowdown_factor=cfg_dict.get("slowdown_factor", 0.5),
        recovery_speed=cfg_dict.get("recovery_speed", 0.2),
        resume_consecutive=cfg_dict.get("resume_consecutive", 10),
        stop_requires_ood=cfg_dict.get("stop_requires_ood", True),
        ood_confirmation_steps=cfg_dict.get("ood_confirmation_steps", 3),
    )


def min_lidar_distance(lidar_obs: np.ndarray, max_range: float) -> float:
    """
    Convert a normalised LiDAR observation to the minimum physical distance.

    Parameters
    ----------
    lidar_obs:
        Normalised LiDAR array in [0, 1].
    max_range:
        LiDAR max range in metres.

    Returns
    -------
    Minimum distance in metres.
    """
    return float(np.min(lidar_obs)) * max_range
