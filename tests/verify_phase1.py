"""
tests/verify_phase1.py
Quick standalone verification script (no pytest, no gymnasium needed).
Run: python tests/verify_phase1.py
"""
import sys
import math
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util

def _load_module(name: str, path: str):
    """Load a module file and register it by name so dataclass __module__ works."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

root = Path(__file__).parent.parent / "environment"

robot_mod    = _load_module("environment.robot",     str(root / "robot.py"))
obstacle_mod = _load_module("environment.obstacles", str(root / "obstacles.py"))
physics_mod  = _load_module("environment.physics",   str(root / "physics.py"))
lidar_mod    = _load_module("environment.lidar",     str(root / "lidar.py"))
scenario_mod = _load_module("environment.scenarios", str(root / "scenarios.py"))
rewards_mod  = _load_module("environment.rewards",   str(root / "rewards.py"))

import numpy as np

PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []

def check(name, condition, detail=""):
    if condition:
        print("{} {}".format(PASS, name))
    else:
        print("{} {}  {}".format(FAIL, name, detail))
        errors.append(name)

# ---- Robot ----------------------------------------------------------------
Robot = robot_mod.Robot
r = Robot(radius=0.3, max_linear_velocity=1.0, max_angular_velocity=1.5)

r.reset(0.0, 0.0, 0.0)
s = r.step(1.0, 0.0, dt=0.1)
check("Robot straight motion x~0.1", abs(s.x - 0.1) < 1e-9, "got {}".format(s.x))

r.reset(0.0, 0.0, 0.0)
s = r.step(999.0, 999.0, dt=0.1)
check("Robot velocity clipping v", abs(s.v) <= 1.0 + 1e-9)
check("Robot velocity clipping omega", abs(s.omega) <= 1.5 + 1e-9)

r.reset(0.0, 0.0, 0.0)
check("Robot distance_to 3-4-5", abs(r.distance_to(3.0, 4.0) - 5.0) < 1e-9)

r.reset(0.0, 0.0, 0.0)
check("Robot angle_to ahead~0", abs(r.angle_to(1.0, 0.0)) < 1e-9)

# ---- Obstacles ------------------------------------------------------------
RectObstacle   = obstacle_mod.RectObstacle
CircleObstacle = obstacle_mod.CircleObstacle
ObstacleMap    = obstacle_mod.ObstacleMap

rect = RectObstacle(5.0, 5.0, 1.0, 1.0)
check("Rect contains centre", rect.contains(5.0, 5.0))
check("Rect outside", not rect.contains(7.0, 5.0))
check("Rect margin", rect.contains(6.05, 5.0, margin=0.1))

t = rect.ray_intersection(0.0, 5.0, 1.0, 0.0)
check("Rect ray hit t~4", t is not None and abs(t - 4.0) < 1e-5, "t={}".format(t))

t_miss = rect.ray_intersection(0.0, 0.0, 0.0, 1.0)
check("Rect ray miss", t_miss is None, "got {}".format(t_miss))

circ = CircleObstacle(5.0, 5.0, 2.0)
check("Circle contains centre", circ.contains(5.0, 5.0))
check("Circle outside", not circ.contains(7.1, 5.0))

t2 = circ.ray_intersection(0.0, 5.0, 1.0, 0.0)
check("Circle ray hit t~3", t2 is not None and abs(t2 - 3.0) < 1e-5, "t2={}".format(t2))

# ---- ObstacleMap ----------------------------------------------------------
omap = ObstacleMap(20.0, 20.0)
omap.add(rect)
d = omap.cast_ray(0.0, 5.0, 0.0, 20.0)
check("cast_ray hits rect at d~4", abs(d - 4.0) < 1e-5, "d={}".format(d))

check("Wall boundary detection", omap.point_in_obstacle(0.05, 10.0, margin=0.1))
check("Free space not obstacle", not omap.point_in_obstacle(15.0, 15.0))

# ---- LiDAR ----------------------------------------------------------------
LiDAR       = lidar_mod.LiDAR
LiDARConfig = lidar_mod.LiDARConfig

lidar = LiDAR(LiDARConfig(num_rays=36, fov_deg=360.0, max_range=10.0,
                           noise_enabled=False))
omap2 = ObstacleMap(20.0, 20.0)
scan = lidar.scan(10.0, 10.0, 0.0, omap2)
check("LiDAR output shape (36,)", scan.shape == (36,), "got {}".format(scan.shape))
check("LiDAR output dtype float32", scan.dtype == np.float32)
check("LiDAR normalised [0,1]", np.all(scan >= 0.0) and np.all(scan <= 1.0))

corrupted = LiDAR.corrupt_sector(scan, 0, 5, 0.0)
check("LiDAR corrupt sector", np.all(corrupted[:5] == 0.0))

noisy = LiDAR.add_gaussian_noise(scan, std=10.0)
check("LiDAR noise clamped [0,1]", np.all(noisy >= 0.0) and np.all(noisy <= 1.0))

# ---- Scenario generator ---------------------------------------------------
ScenarioGenerator = scenario_mod.ScenarioGenerator
ScenarioConfig    = scenario_mod.ScenarioConfig

cfg = ScenarioConfig(world_width=20.0, world_height=20.0,
                     num_static_min=3, num_static_max=6,
                     robot_radius=0.3, goal_radius=0.5,
                     min_goal_distance=3.0)
gen = ScenarioGenerator(cfg, seed=42)
obs_map, sx, sy, stheta, gx, gy = gen.generate()
check("Scenario: obstacles generated", len(obs_map) >= 3, "got {}".format(len(obs_map)))
check("Scenario: robot not in obstacle",
      not obs_map.point_in_obstacle(sx, sy, margin=0.3))
check("Scenario: goal not in obstacle",
      not obs_map.point_in_obstacle(gx, gy, margin=0.5))
check("Scenario: start-goal dist >= 3",
      math.hypot(gx-sx, gy-sy) >= 3.0,
      "dist={:.2f}".format(math.hypot(gx-sx, gy-sy)))

# ---- Reproducibility -------------------------------------------------------
gen2 = ScenarioGenerator(cfg, seed=42)
obs_map2, sx2, sy2, _, gx2, gy2 = gen2.generate()
check("Reproducibility: same start", abs(sx-sx2) < 1e-9 and abs(sy-sy2) < 1e-9)
check("Reproducibility: same goal",  abs(gx-gx2) < 1e-9 and abs(gy-gy2) < 1e-9)

# ---- Physics engine -------------------------------------------------------
PhysicsEngine = physics_mod.PhysicsEngine
r2 = Robot()
r2.reset(sx, sy, stheta)
physics = PhysicsEngine(robot=r2, obstacle_map=obs_map, dt=0.1, goal_radius=0.5)
_, collision, goal = physics.step(0.0, 0.0, gx, gy)
check("Physics: no collision at start", not collision, "collided on spawn")

# ---- Rewards --------------------------------------------------------------
RewardFunction = rewards_mod.RewardFunction
rf = RewardFunction()
rf.reset(initial_distance=5.0)
rew = rf.compute(current_distance=4.5, min_obstacle_distance=2.0,
                 prev_v=0.0, prev_omega=0.0, current_v=0.5, current_omega=0.0,
                 collision=False, goal_reached=False)
check("Reward: progress gives positive", rew > 0, "rew={:.3f}".format(rew))

rew_collision = rf.compute(current_distance=4.5, min_obstacle_distance=0.0,
                            prev_v=0.5, prev_omega=0.0, current_v=0.5, current_omega=0.0,
                            collision=True, goal_reached=False)
check("Reward: collision gives large negative", rew_collision < -40,
      "rew={:.3f}".format(rew_collision))

rew_goal = rf.compute(current_distance=0.1, min_obstacle_distance=2.0,
                       prev_v=0.5, prev_omega=0.0, current_v=0.5, current_omega=0.0,
                       collision=False, goal_reached=True)
check("Reward: goal gives large positive", rew_goal >= 100.0,
      "rew={:.3f}".format(rew_goal))

# ---- Summary --------------------------------------------------------------
total = 27
print()
if errors:
    print("FAILURES ({}/{}): {}".format(len(errors), total, ", ".join(errors)))
    sys.exit(1)
else:
    print("All {} checks PASSED".format(total))
