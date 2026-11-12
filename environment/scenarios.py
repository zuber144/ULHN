"""
environment/scenarios.py
========================
Procedural scenario generator for the ULHN 2D simulator.

Generates:
    - Random obstacle layouts (rectangles + circles)
    - Random robot start pose
    - Random goal position

Constraints:
    - Robot must not spawn inside or overlapping any obstacle
    - Goal must not spawn inside or overlapping any obstacle
    - Robot and goal must be at least *min_goal_distance* apart
    - All placements use a configurable random seed for reproducibility
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from environment.obstacles import (
    CircleObstacle,
    Obstacle,
    ObstacleMap,
    RectObstacle,
)

logger = logging.getLogger(__name__)

# Maximum placement retries before giving up
_MAX_RETRIES = 200


@dataclass
class ScenarioConfig:
    """All parameters that control scenario generation."""

    world_width: float = 20.0
    world_height: float = 20.0
    num_static_min: int = 5
    num_static_max: int = 12
    rect_width_range: Tuple[float, float] = (0.5, 3.0)
    rect_height_range: Tuple[float, float] = (0.5, 3.0)
    circle_radius_range: Tuple[float, float] = (0.3, 1.5)
    rect_fraction: float = 0.6     # Fraction of obstacles that are rectangular
    margin: float = 0.5            # Clearance for robot / goal placement
    robot_radius: float = 0.3
    goal_radius: float = 0.5
    min_goal_distance: float = 3.0


class ScenarioGenerator:
    """
    Generates reproducible random scenarios.

    Usage::

        gen = ScenarioGenerator(config, seed=42)
        obs_map, start_x, start_y, start_theta, goal_x, goal_y = gen.generate()
    """

    def __init__(
        self,
        config: Optional[ScenarioConfig] = None,
        seed: int = 42,
    ) -> None:
        self.config = config or ScenarioConfig()
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

    def reseed(self, seed: int) -> None:
        """Re-initialise the RNG with a new seed."""
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
    ) -> Tuple[ObstacleMap, float, float, float, float, float]:
        """
        Generate a complete scenario.

        Returns
        -------
        obstacle_map : ObstacleMap
        start_x, start_y, start_theta : float
            Robot starting pose.
        goal_x, goal_y : float
            Goal position.
        """
        cfg = self.config

        obstacle_map = ObstacleMap(
            world_width=cfg.world_width,
            world_height=cfg.world_height,
        )

        # 1. Generate obstacles
        n_obs = int(self._rng.integers(cfg.num_static_min, cfg.num_static_max + 1))
        obstacles = self._generate_obstacles(n_obs, obstacle_map, cfg)
        for obs in obstacles:
            obstacle_map.add(obs)

        # 2. Generate robot start pose
        start_x, start_y = self._sample_free_position(
            obstacle_map,
            margin=cfg.robot_radius + cfg.margin,
        )
        start_theta = float(self._rng.uniform(-math.pi, math.pi))

        # 3. Generate goal position
        for attempt in range(_MAX_RETRIES):
            goal_x, goal_y = self._sample_free_position(
                obstacle_map,
                margin=cfg.goal_radius + cfg.margin,
            )
            dist = math.hypot(goal_x - start_x, goal_y - start_y)
            if dist >= cfg.min_goal_distance:
                break
        else:
            logger.warning(
                "Could not place goal far enough from start after %d retries; "
                "using last position anyway.",
                _MAX_RETRIES,
            )

        logger.debug(
            "Scenario: %d obstacles, start=(%.1f,%.1f,%.1f°), goal=(%.1f,%.1f)",
            len(obstacles),
            start_x, start_y, math.degrees(start_theta),
            goal_x, goal_y,
        )

        return obstacle_map, start_x, start_y, start_theta, goal_x, goal_y

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_obstacles(
        self,
        n: int,
        obstacle_map: ObstacleMap,
        cfg: ScenarioConfig,
    ) -> List[Obstacle]:
        """
        Generate *n* non-overlapping obstacles placed inside the world.
        """
        obstacles: List[Obstacle] = []
        padding = 1.0   # Minimum distance from world boundary

        for _ in range(n):
            if self._rng.random() < cfg.rect_fraction:
                obs = self._try_place_rect(obstacles, cfg, padding)
            else:
                obs = self._try_place_circle(obstacles, cfg, padding)
            if obs is not None:
                obstacles.append(obs)

        return obstacles

    def _try_place_rect(
        self,
        existing: List[Obstacle],
        cfg: ScenarioConfig,
        padding: float,
    ) -> Optional[RectObstacle]:
        w = float(self._rng.uniform(*cfg.rect_width_range))
        h = float(self._rng.uniform(*cfg.rect_height_range))
        half_w = w / 2.0
        half_h = h / 2.0

        for _ in range(_MAX_RETRIES):
            cx = float(self._rng.uniform(padding + half_w,
                                          cfg.world_width - padding - half_w))
            cy = float(self._rng.uniform(padding + half_h,
                                          cfg.world_height - padding - half_h))
            candidate = RectObstacle(cx=cx, cy=cy, half_w=half_w, half_h=half_h)
            if not self._overlaps_any(candidate, existing):
                return candidate
        return None

    def _try_place_circle(
        self,
        existing: List[Obstacle],
        cfg: ScenarioConfig,
        padding: float,
    ) -> Optional[CircleObstacle]:
        r = float(self._rng.uniform(*cfg.circle_radius_range))

        for _ in range(_MAX_RETRIES):
            cx = float(self._rng.uniform(padding + r, cfg.world_width - padding - r))
            cy = float(self._rng.uniform(padding + r, cfg.world_height - padding - r))
            candidate = CircleObstacle(cx=cx, cy=cy, radius=r)
            if not self._overlaps_any(candidate, existing):
                return candidate
        return None

    @staticmethod
    def _overlaps_any(candidate: Obstacle, existing: List[Obstacle]) -> bool:
        """
        Naive pairwise overlap check using bounding boxes.
        Good enough for small numbers of obstacles.
        """
        cx1, cy1, cx2, cy2 = candidate.bbox()
        for obs in existing:
            ox1, oy1, ox2, oy2 = obs.bbox()
            # AABB overlap with a small gap
            gap = 0.3
            if (cx1 - gap < ox2 and cx2 + gap > ox1 and
                    cy1 - gap < oy2 and cy2 + gap > oy1):
                return True
        return False

    def _sample_free_position(
        self,
        obstacle_map: ObstacleMap,
        margin: float,
    ) -> Tuple[float, float]:
        """Sample a random (x, y) not in any obstacle."""
        cfg = self.config
        for _ in range(_MAX_RETRIES):
            x = float(self._rng.uniform(margin, cfg.world_width - margin))
            y = float(self._rng.uniform(margin, cfg.world_height - margin))
            if not obstacle_map.point_in_obstacle(x, y, margin=margin):
                return x, y
        # Fallback: return world centre (rarely needed)
        logger.warning("Could not sample a free position; using world centre.")
        return cfg.world_width / 2.0, cfg.world_height / 2.0
