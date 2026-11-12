"""tests/test_lidar.py — Unit tests for the simulated 2D LiDAR sensor."""

from __future__ import annotations

import math
import pytest
import numpy as np
from environment.lidar import LiDAR, LiDARConfig
from environment.obstacles import ObstacleMap, RectObstacle, CircleObstacle


def _empty_map(w: float = 20.0, h: float = 20.0) -> ObstacleMap:
    return ObstacleMap(world_width=w, world_height=h)


class TestLiDAR:
    def setup_method(self):
        self.lidar = LiDAR(LiDARConfig(num_rays=36, fov_deg=360.0,
                                       max_range=10.0, noise_enabled=False))

    def test_output_shape(self):
        omap = _empty_map()
        obs = self.lidar.scan(10.0, 10.0, 0.0, omap)
        assert obs.shape == (36,)
        assert obs.dtype == np.float32

    def test_output_normalised(self):
        omap = _empty_map()
        obs = self.lidar.scan(10.0, 10.0, 0.0, omap)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    def test_max_range_in_empty_room(self):
        """With no obstacles (robot near centre), rays may hit walls but should
        not exceed max_range."""
        omap = _empty_map(20.0, 20.0)
        obs = self.lidar.scan(10.0, 10.0, 0.0, omap)
        assert np.all(obs <= 1.0)

    def test_wall_detection(self):
        """Robot placed very close to a wall should have at least one short ray."""
        omap = _empty_map(20.0, 20.0)
        # Robot at (0.5, 10) — the closest wall is only 0.5 m away (left wall x=0)
        obs = self.lidar.scan(0.5, 10.0, 0.0, omap)
        # At least one ray must hit the left wall (0.5 m → norm 0.05)
        assert np.min(obs) < 0.1, f"min={np.min(obs):.3f}, expected < 0.1"

    def test_obstacle_detection(self):
        """A circle obstacle placed close in front should shorten the minimum ray."""
        omap_clear = _empty_map(20.0, 20.0)
        omap_obs   = _empty_map(20.0, 20.0)
        omap_obs.add(CircleObstacle(cx=12.0, cy=10.0, radius=0.5))
        obs_without = self.lidar.scan(10.0, 10.0, 0.0, omap_clear)
        obs_with    = self.lidar.scan(10.0, 10.0, 0.0, omap_obs)
        # The nearest ray to the obstacle should be shorter than without it
        assert np.min(obs_with) < np.min(obs_without)

    def test_drop_readings_fraction(self):
        original = np.linspace(0.1, 0.9, 36, dtype=np.float32)
        dropped = LiDAR.drop_readings(original, fraction=0.5, replacement="max")
        assert np.sum(dropped == 1.0) >= 1

    def test_corrupt_sector(self):
        original = np.full(36, 0.5, dtype=np.float32)
        corrupted = LiDAR.corrupt_sector(original, start_ray=0, width_rays=5, value=0.0)
        assert np.all(corrupted[:5] == 0.0)
        assert np.all(corrupted[5:] == 0.5)

    def test_gaussian_noise_clipping(self):
        original = np.full(36, 0.5, dtype=np.float32)
        noisy = LiDAR.add_gaussian_noise(original, std=10.0)  # extreme noise
        assert np.all(noisy >= 0.0) and np.all(noisy <= 1.0)
