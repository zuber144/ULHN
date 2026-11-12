"""
experiments/disturbances.py
============================
Configurable LiDAR disturbance functions for OOD experiments.

All functions take a *clean* normalised LiDAR array and return a disturbed
version.  The simulator environment itself is never permanently corrupted.

Disturbance modes:
    gaussian_noise     – Add Gaussian noise beyond training distribution
    missing_readings   – Replace a fraction of readings (max / zero / prev)
    corrupt_sector     – Zero out a consecutive sector of rays
    partial_blockage   – Gradual fade-out of a sector (simulates obstruction)
"""

from __future__ import annotations

import math
import numpy as np


def gaussian_noise(
    obs: np.ndarray,
    std: float = 0.15,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Add strong Gaussian noise — well outside the training distribution."""
    if rng is None:
        rng = np.random.default_rng()
    noisy = obs + rng.normal(0.0, std, size=obs.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def missing_readings(
    obs: np.ndarray,
    fraction: float = 0.3,
    replacement: str = "max",
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Randomly drop a fraction of readings."""
    if rng is None:
        rng = np.random.default_rng()
    result = obs.copy()
    n = len(obs)
    n_drop = max(1, int(fraction * n))
    indices = rng.choice(n, size=n_drop, replace=False)
    if replacement == "max":
        result[indices] = 1.0
    elif replacement == "zero":
        result[indices] = 0.0
    elif replacement == "prev":
        for idx in sorted(indices):
            result[idx] = result[idx - 1] if idx > 0 else 1.0
    return result


def corrupt_sector(
    obs: np.ndarray,
    start_ray: int,
    width_rays: int = 8,
    value: float = 0.0,
) -> np.ndarray:
    """Zero out a contiguous sector of rays."""
    result = obs.copy()
    n = len(obs)
    for i in range(width_rays):
        result[(start_ray + i) % n] = value
    return result


def partial_blockage(
    obs: np.ndarray,
    centre_ray: int,
    half_width: int = 6,
    blockage_value: float = 0.0,
) -> np.ndarray:
    """
    Gradually attenuate a sector (simulate sensor obstruction).
    Rays at the centre of the sector are fully blocked;
    rays at the edge are partially attenuated.
    """
    result = obs.copy()
    n = len(obs)
    for offset in range(-half_width, half_width + 1):
        i = (centre_ray + offset) % n
        # Cosine taper: 1 at centre, 0 at edges
        weight = math.cos(math.pi * offset / (2 * half_width + 1))
        attenuation = weight ** 2
        # Blend towards blockage_value
        result[i] = float(obs[i] * (1.0 - attenuation) +
                          blockage_value * attenuation)
    return result.astype(np.float32)
