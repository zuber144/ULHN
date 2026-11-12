"""
environment/lidar.py
====================
Simulated 2D LiDAR sensor using ray casting.

Configuration:
    num_rays  — number of beams
    fov_deg   — total field of view in degrees (360 = full circle)
    max_range — maximum detection range in metres
    noise_std — standard deviation of additive Gaussian noise (normal operation)

Output:
    Normalised float32 array in [0, 1], where 0 = object at sensor and
    1 = maximum range (or beyond).

Disturbance modes (for OOD experiments — Phase 8):
    add_noise(obs, std)         — extra Gaussian noise
    drop_readings(obs, frac)    — replace readings with max_range
    corrupt_sector(obs, start, width) — zero out a sector of consecutive rays
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from environment.obstacles import ObstacleMap


def select_forward_sector(
    observations: np.ndarray,
    fraction: float = 0.5,
) -> np.ndarray:
    """Select the centered forward-facing sector from the last array axis."""
    observations = np.asarray(observations)
    if observations.ndim == 0:
        raise ValueError("LiDAR observations must have at least one dimension")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("forward sector fraction must be in (0, 1]")

    ray_count = observations.shape[-1]
    sector_count = max(1, int(round(ray_count * fraction)))
    start = (ray_count - sector_count) // 2
    return observations[..., start:start + sector_count]


@dataclass
class LiDARConfig:
    """LiDAR sensor parameters."""

    num_rays: int = 36
    fov_deg: float = 360.0
    max_range: float = 10.0
    noise_std: float = 0.01
    noise_enabled: bool = False


class LiDAR:
    """
    Simulated 2D LiDAR sensor.

    Casts *num_rays* evenly-spaced rays from the robot position and
    returns normalised distances to the first obstacle hit.
    """

    def __init__(self, config: Optional[LiDARConfig] = None) -> None:
        self.config = config or LiDARConfig()

        # Pre-compute ray angle offsets relative to robot heading
        fov_rad = math.radians(self.config.fov_deg)
        if self.config.num_rays == 1:
            self._ray_offsets = np.array([0.0], dtype=np.float64)
        else:
            self._ray_offsets = np.linspace(
                -fov_rad / 2.0,
                fov_rad / 2.0,
                self.config.num_rays,
                endpoint=(self.config.fov_deg < 360.0),
                dtype=np.float64,
            )

    # ------------------------------------------------------------------
    # Primary sensor interface
    # ------------------------------------------------------------------

    def scan(
        self,
        x: float,
        y: float,
        theta: float,
        obstacle_map: ObstacleMap,
    ) -> np.ndarray:
        """
        Perform a LiDAR scan from position (x, y) with heading *theta*.

        Returns
        -------
        np.ndarray of shape (num_rays,), dtype float32
            Normalised distances in [0, 1].
            0 → object at sensor origin
            1 → maximum range or no obstacle detected
        """
        raw = self._raw_scan(x, y, theta, obstacle_map)
        normalised = (raw / self.config.max_range).astype(np.float32)
        normalised = np.clip(normalised, 0.0, 1.0)

        if self.config.noise_enabled and self.config.noise_std > 0.0:
            normalised = LiDAR.add_gaussian_noise(normalised, self.config.noise_std)

        return normalised

    # ------------------------------------------------------------------
    # Raw (unnormalised) scan
    # ------------------------------------------------------------------

    def _raw_scan(
        self,
        x: float,
        y: float,
        theta: float,
        obstacle_map: ObstacleMap,
    ) -> np.ndarray:
        """Return raw distances in metres."""
        distances = np.empty(self.config.num_rays, dtype=np.float64)
        for i, offset in enumerate(self._ray_offsets):
            angle = theta + offset
            distances[i] = obstacle_map.cast_ray(
                x, y, angle, self.config.max_range
            )
        return distances

    # ------------------------------------------------------------------
    # Disturbance methods (used in OOD experiments)
    # ------------------------------------------------------------------

    @staticmethod
    def add_gaussian_noise(obs: np.ndarray, std: float) -> np.ndarray:
        """
        Add zero-mean Gaussian noise and clip to [0, 1].

        Parameters
        ----------
        obs:
            Normalised LiDAR observation.
        std:
            Noise standard deviation (in normalised units, i.e. [0, 1]).
        """
        noisy = obs + np.random.normal(0.0, std, size=obs.shape).astype(np.float32)
        return np.clip(noisy, 0.0, 1.0)

    @staticmethod
    def drop_readings(
        obs: np.ndarray,
        fraction: float = 0.2,
        replacement: str = "max",
    ) -> np.ndarray:
        """
        Randomly replace a fraction of readings.

        Parameters
        ----------
        obs:
            Normalised LiDAR observation.
        fraction:
            Fraction of rays to drop (0–1).
        replacement:
            One of 'max' (1.0), 'zero' (0.0), or 'prev' (copy previous).
        """
        result = obs.copy()
        n = len(obs)
        n_drop = max(1, int(fraction * n))
        indices = np.random.choice(n, size=n_drop, replace=False)

        if replacement == "max":
            result[indices] = 1.0
        elif replacement == "zero":
            result[indices] = 0.0
        elif replacement == "prev":
            for idx in sorted(indices):
                result[idx] = result[idx - 1] if idx > 0 else 1.0
        else:
            raise ValueError(f"Unknown replacement mode: '{replacement}'")

        return result

    @staticmethod
    def corrupt_sector(
        obs: np.ndarray,
        start_ray: int,
        width_rays: int,
        value: float = 0.0,
    ) -> np.ndarray:
        """
        Set a contiguous sector of LiDAR rays to a fixed value.

        Parameters
        ----------
        obs:
            Normalised LiDAR observation.
        start_ray:
            Index of the first ray in the sector.
        width_rays:
            Number of consecutive rays to corrupt.
        value:
            Value to fill (0.0 = blocked, 1.0 = max range).
        """
        result = obs.copy()
        n = len(obs)
        for i in range(width_rays):
            result[(start_ray + i) % n] = value
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def num_rays(self) -> int:
        return self.config.num_rays

    @property
    def max_range(self) -> float:
        return self.config.max_range
