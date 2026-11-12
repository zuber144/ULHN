"""tests/test_obstacles.py — Unit tests for obstacle primitives and ObstacleMap."""

from __future__ import annotations

import math
import pytest
import numpy as np
from environment.obstacles import RectObstacle, CircleObstacle, ObstacleMap


class TestRectObstacle:
    def setup_method(self):
        # 2×2 rectangle centred at (5, 5)
        self.rect = RectObstacle(cx=5.0, cy=5.0, half_w=1.0, half_h=1.0)

    def test_contains_centre(self):
        assert self.rect.contains(5.0, 5.0)

    def test_contains_edge(self):
        assert self.rect.contains(6.0, 5.0)

    def test_outside(self):
        assert not self.rect.contains(6.1, 5.0)

    def test_margin(self):
        # Just outside with margin should be True
        assert self.rect.contains(6.05, 5.0, margin=0.1)

    def test_distance_inside_is_negative(self):
        d = self.rect.distance_to_boundary(5.0, 5.0)
        assert d < 0.0

    def test_distance_outside(self):
        d = self.rect.distance_to_boundary(7.0, 5.0)
        assert d == pytest.approx(1.0, abs=1e-9)

    def test_ray_hit(self):
        # Ray from (0, 5) pointing right (+x direction)
        t = self.rect.ray_intersection(0.0, 5.0, 1.0, 0.0)
        assert t is not None
        assert t == pytest.approx(4.0, abs=1e-6)   # hits left wall at x=4

    def test_ray_miss(self):
        # Ray from (0, 0) pointing straight up, misses rect at (5,5)
        t = self.rect.ray_intersection(0.0, 0.0, 0.0, 1.0)
        assert t is None

    def test_bbox(self):
        x0, y0, x1, y1 = self.rect.bbox()
        assert x0 == pytest.approx(4.0)
        assert y0 == pytest.approx(4.0)
        assert x1 == pytest.approx(6.0)
        assert y1 == pytest.approx(6.0)


class TestCircleObstacle:
    def setup_method(self):
        self.circle = CircleObstacle(cx=5.0, cy=5.0, radius=2.0)

    def test_contains_centre(self):
        assert self.circle.contains(5.0, 5.0)

    def test_contains_on_edge(self):
        assert self.circle.contains(7.0, 5.0)

    def test_outside(self):
        assert not self.circle.contains(7.01, 5.0)

    def test_distance_inside(self):
        d = self.circle.distance_to_boundary(5.0, 5.0)
        assert d == pytest.approx(-2.0)

    def test_distance_outside(self):
        d = self.circle.distance_to_boundary(8.0, 5.0)
        assert d == pytest.approx(1.0, abs=1e-9)

    def test_ray_hit(self):
        t = self.circle.ray_intersection(0.0, 5.0, 1.0, 0.0)
        assert t is not None
        assert t == pytest.approx(3.0, abs=1e-6)   # enters at x=3

    def test_ray_miss(self):
        t = self.circle.ray_intersection(0.0, 0.0, 1.0, 0.0)
        assert t is None


class TestObstacleMap:
    def setup_method(self):
        self.omap = ObstacleMap(world_width=20.0, world_height=20.0)
        self.omap.add(RectObstacle(cx=10.0, cy=10.0, half_w=1.0, half_h=1.0))
        self.omap.add(CircleObstacle(cx=5.0, cy=5.0, radius=1.0))

    def test_len(self):
        assert len(self.omap) == 2

    def test_point_in_obstacle(self):
        assert self.omap.point_in_obstacle(10.0, 10.0)

    def test_point_outside_all(self):
        assert not self.omap.point_in_obstacle(15.0, 15.0)

    def test_wall_boundary(self):
        # World boundary is treated as obstacle
        assert self.omap.point_in_obstacle(0.0, 10.0, margin=0.1)

    def test_ray_cast_hits_rect(self):
        # Ray from (0, 10) pointing right — should hit rect at cx=10
        dist = self.omap.cast_ray(0.0, 10.0, 0.0, max_range=20.0)
        assert dist == pytest.approx(9.0, abs=0.05)

    def test_ray_cast_hits_wall(self):
        # Ray from (2, 10) pointing right — rect is at (10, 10) half_w=1, so edge at x=9.
        # Right wall is at x=20.  Ray should hit rect edge at dist=7 from origin.
        dist = self.omap.cast_ray(2.0, 10.0, 0.0, max_range=50.0)
        assert dist == pytest.approx(7.0, abs=0.1), f"got {dist}"

    def test_min_distance(self):
        # Point at (5, 4) is just outside the circle (radius=1, boundary at y=4)
        d = self.omap.min_distance_to_obstacle(5.0, 4.0)
        assert d == pytest.approx(0.0, abs=0.01)
