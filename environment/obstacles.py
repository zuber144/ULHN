"""
environment/obstacles.py
========================
Obstacle primitives and obstacle map for the ULHN 2D simulator.

Supported shapes:
    RectObstacle   – Axis-aligned rectangle
    CircleObstacle – Circle / disk

All obstacles expose a common interface:
    contains(x, y, margin=0.0) -> bool
    distance_to_boundary(x, y) -> float
    bbox()  -> (x_min, y_min, x_max, y_max)
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


class Obstacle(abc.ABC):
    """Abstract base class for all obstacle shapes."""

    @abc.abstractmethod
    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        """
        Return True if point (x, y) is inside or within *margin* of the
        obstacle boundary.
        """

    @abc.abstractmethod
    def distance_to_boundary(self, x: float, y: float) -> float:
        """
        Return the shortest distance from point (x, y) to the obstacle
        boundary.  Negative values indicate the point is inside.
        """

    @abc.abstractmethod
    def bbox(self) -> Tuple[float, float, float, float]:
        """Return axis-aligned bounding box (x_min, y_min, x_max, y_max)."""

    @abc.abstractmethod
    def ray_intersection(
        self, ox: float, oy: float, dx: float, dy: float
    ) -> Optional[float]:
        """
        Return the distance along the ray (ox, oy) + t*(dx, dy) to the first
        intersection with this obstacle, or None if the ray misses.

        The direction vector (dx, dy) must be a unit vector.
        """


# ---------------------------------------------------------------------------
# Rectangular obstacle
# ---------------------------------------------------------------------------

@dataclass
class RectObstacle(Obstacle):
    """
    Axis-aligned rectangle defined by its centre and half-extents.

    Parameters
    ----------
    cx, cy : float
        Centre position (m).
    half_w, half_h : float
        Half-widths in x and y (m).
    """

    cx: float
    cy: float
    half_w: float
    half_h: float

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (
            abs(x - self.cx) <= self.half_w + margin
            and abs(y - self.cy) <= self.half_h + margin
        )

    def distance_to_boundary(self, x: float, y: float) -> float:
        """
        Signed distance:  negative = inside,  positive = outside.
        Uses the standard SDF for an axis-aligned box.
        """
        dx = abs(x - self.cx) - self.half_w
        dy = abs(y - self.cy) - self.half_h
        outside = math.sqrt(max(dx, 0.0) ** 2 + max(dy, 0.0) ** 2)
        inside = min(max(dx, dy), 0.0)
        return outside + inside

    def bbox(self) -> Tuple[float, float, float, float]:
        return (
            self.cx - self.half_w,
            self.cy - self.half_h,
            self.cx + self.half_w,
            self.cy + self.half_h,
        )

    def ray_intersection(
        self, ox: float, oy: float, dx: float, dy: float
    ) -> Optional[float]:
        """Slab method for AABB–ray intersection."""
        x_min, y_min, x_max, y_max = self.bbox()

        t_min = -math.inf
        t_max = math.inf

        # X slab
        if abs(dx) > 1e-12:
            tx1 = (x_min - ox) / dx
            tx2 = (x_max - ox) / dx
            t_min = max(t_min, min(tx1, tx2))
            t_max = min(t_max, max(tx1, tx2))
        else:
            if ox < x_min or ox > x_max:
                return None

        # Y slab
        if abs(dy) > 1e-12:
            ty1 = (y_min - oy) / dy
            ty2 = (y_max - oy) / dy
            t_min = max(t_min, min(ty1, ty2))
            t_max = min(t_max, max(ty1, ty2))
        else:
            if oy < y_min or oy > y_max:
                return None

        if t_max < t_min or t_max < 0.0:
            return None

        t_hit = t_min if t_min >= 0.0 else t_max
        return t_hit if t_hit >= 0.0 else None

    def __repr__(self) -> str:
        return (
            f"RectObstacle(cx={self.cx:.2f}, cy={self.cy:.2f}, "
            f"w={2*self.half_w:.2f}, h={2*self.half_h:.2f})"
        )


# ---------------------------------------------------------------------------
# Circular obstacle
# ---------------------------------------------------------------------------

@dataclass
class CircleObstacle(Obstacle):
    """
    Circular obstacle defined by centre and radius.

    Parameters
    ----------
    cx, cy : float
        Centre position (m).
    radius : float
        Radius (m).
    """

    cx: float
    cy: float
    radius: float

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return math.hypot(x - self.cx, y - self.cy) <= self.radius + margin

    def distance_to_boundary(self, x: float, y: float) -> float:
        """Positive outside, negative inside."""
        return math.hypot(x - self.cx, y - self.cy) - self.radius

    def bbox(self) -> Tuple[float, float, float, float]:
        return (
            self.cx - self.radius,
            self.cy - self.radius,
            self.cx + self.radius,
            self.cy + self.radius,
        )

    def ray_intersection(
        self, ox: float, oy: float, dx: float, dy: float
    ) -> Optional[float]:
        """Analytic ray–circle intersection."""
        fx = ox - self.cx
        fy = oy - self.cy
        a = dx * dx + dy * dy          # = 1 if direction is unit vector
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - self.radius ** 2

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return None

        sqrt_d = math.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)

        if t2 < 0.0:
            return None  # both intersections behind the ray origin
        if t1 >= 0.0:
            return t1    # ray starts outside the circle
        return t2        # ray starts inside the circle

    def __repr__(self) -> str:
        return (
            f"CircleObstacle(cx={self.cx:.2f}, cy={self.cy:.2f}, "
            f"r={self.radius:.2f})"
        )


# ---------------------------------------------------------------------------
# Obstacle map
# ---------------------------------------------------------------------------

class ObstacleMap:
    """
    Container for all obstacles in the simulation environment.

    Also models the world boundary as four implicit wall segments so that
    the robot cannot leave the world.
    """

    def __init__(
        self,
        world_width: float,
        world_height: float,
        obstacles: Optional[List[Obstacle]] = None,
    ) -> None:
        self.world_width = world_width
        self.world_height = world_height
        self._obstacles: List[Obstacle] = list(obstacles or [])

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, obstacle: Obstacle) -> None:
        self._obstacles.append(obstacle)

    def clear(self) -> None:
        self._obstacles.clear()

    @property
    def obstacles(self) -> List[Obstacle]:
        """Read-only view of the obstacle list."""
        return list(self._obstacles)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def point_in_obstacle(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Return True if (x, y) is inside any obstacle (or within margin)."""
        # World boundary check
        if (
            x < margin
            or y < margin
            or x > self.world_width - margin
            or y > self.world_height - margin
        ):
            return True
        return any(obs.contains(x, y, margin=margin) for obs in self._obstacles)

    def min_distance_to_obstacle(self, x: float, y: float) -> float:
        """
        Return the distance from (x, y) to the nearest obstacle boundary.
        The world boundary is treated as an obstacle.
        """
        wall_dist = min(
            x,
            y,
            self.world_width - x,
            self.world_height - y,
        )
        if not self._obstacles:
            return wall_dist
        obs_dist = min(
            obs.distance_to_boundary(x, y) for obs in self._obstacles
        )
        return min(wall_dist, obs_dist)

    def cast_ray(
        self,
        ox: float,
        oy: float,
        angle: float,
        max_range: float,
    ) -> float:
        """
        Cast a ray from (ox, oy) in direction *angle* (radians, world frame).

        Returns the distance to the first intersection with any obstacle or
        the world boundary, capped at *max_range*.
        """
        dx = math.cos(angle)
        dy = math.sin(angle)

        # World boundary intersections
        t_wall = max_range
        if dx > 1e-12:
            t_wall = min(t_wall, (self.world_width - ox) / dx)
        elif dx < -1e-12:
            t_wall = min(t_wall, -ox / dx)
        if dy > 1e-12:
            t_wall = min(t_wall, (self.world_height - oy) / dy)
        elif dy < -1e-12:
            t_wall = min(t_wall, -oy / dy)

        t_min = t_wall

        for obs in self._obstacles:
            t = obs.ray_intersection(ox, oy, dx, dy)
            if t is not None and 0.0 <= t < t_min:
                t_min = t

        return min(t_min, max_range)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._obstacles)

    def __repr__(self) -> str:
        return (
            f"ObstacleMap(world={self.world_width}×{self.world_height}, "
            f"n_obstacles={len(self._obstacles)})"
        )
