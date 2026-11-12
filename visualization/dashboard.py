"""
visualization/dashboard.py
===========================
Streamlit dashboard for ULHN.

Run with:
    streamlit run visualization/dashboard.py

Features:
    - Real-time 2D world view
    - LiDAR polar plot
    - Anomaly score gauge
    - Safety state indicator
    - Scenario selector
    - Noise / disturbance toggles
    - Live metrics panel
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

# Add project root to path so imports work when run from any directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Handle namespace collision with third-party PyPI 'watchdog' package
# which Streamlit imports during server initialization.
try:
    import watchdog
    _local_wd = str(_ROOT / "watchdog")
    if hasattr(watchdog, "__path__") and _local_wd not in watchdog.__path__:
        watchdog.__path__.insert(0, _local_wd)
except Exception:
    pass

import numpy as np
import streamlit as st
import yaml

# ---- Page config (must be first Streamlit call) ---
st.set_page_config(
    page_title="ULHN — Uncertainty-aware Learning Navigation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- CSS overrides for dark premium look ----
st.markdown("""
<style>
    .stApp { background-color: #0a0a1a; color: #c8c8ff; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a3e, #0d0d2a);
        border: 1px solid #3344aa;
        border-radius: 12px;
        padding: 16px;
        margin: 6px 0;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #66aaff; }
    .metric-label { font-size: 0.8rem; color: #8888cc; margin-top: 4px; }
    .status-normal   { color: #44ff88; font-weight: bold; }
    .status-slowdown { color: #ffcc44; font-weight: bold; }
    .status-stop     { color: #ff4444; font-weight: bold; }
    .status-reorient { color: #ff8844; font-weight: bold; }
    .status-resume   { color: #44ffcc; font-weight: bold; }
    .ood-normal   { color: #44ff88; }
    .ood-active   { color: #ff4444; font-weight: bold; }
    h1, h2, h3 { color: #99aaff; }
    .stButton>button {
        background: linear-gradient(135deg, #2233aa, #113399);
        color: #fff; border-radius: 8px; border: 1px solid #4455cc;
    }
</style>
""", unsafe_allow_html=True)


# ---- Load config ----
@st.cache_resource
def load_config():
    cfg_path = _ROOT / "config.yaml"
    if cfg_path.exists():
        with cfg_path.open() as f:
            return yaml.safe_load(f)
    return {}


CONFIG = load_config()


# ---- Lazy model loader ----
@st.cache_resource
def try_load_ppo():
    try:
        from stable_baselines3 import PPO
        model_path = _ROOT / "models" / "ppo" / "ppo_final.zip"
        if model_path.exists():
            return PPO.load(str(model_path))
    except Exception as e:
        st.warning(f"PPO model not loaded: {e}")
    return None


@st.cache_resource
def try_load_watchdog():
    try:
        # Ensure project's watchdog package is accessible
        import watchdog
        _local_wd = str(_ROOT / "watchdog")
        if hasattr(watchdog, "__path__") and _local_wd not in watchdog.__path__:
            watchdog.__path__.insert(0, _local_wd)

        try:
            from watchdog.autoencoder import load_autoencoder
            from watchdog.threshold import load_threshold_stats
            from watchdog.anomaly_detector import AnomalyDetector
        except ImportError:
            import importlib.util
            def _load_submodule(subname, relpath):
                if subname in sys.modules:
                    return sys.modules[subname]
                spec = importlib.util.spec_from_file_location(subname, _ROOT / relpath)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[subname] = mod
                spec.loader.exec_module(mod)
                return mod

            mod_ae = _load_submodule("watchdog.autoencoder", "watchdog/autoencoder.py")
            mod_th = _load_submodule("watchdog.threshold", "watchdog/threshold.py")
            mod_ad = _load_submodule("watchdog.anomaly_detector", "watchdog/anomaly_detector.py")
            load_autoencoder = mod_ae.load_autoencoder
            load_threshold_stats = mod_th.load_threshold_stats
            AnomalyDetector = mod_ad.AnomalyDetector

        ae_path = _ROOT / "models" / "autoencoder" / "autoencoder.pt"
        th_path = _ROOT / "models" / "autoencoder" / "threshold_stats.json"
        if ae_path.exists() and th_path.exists():
            model = load_autoencoder(str(ae_path))
            stats = load_threshold_stats(str(th_path))
            percentile = CONFIG.get("watchdog", {}).get("threshold_percentile", 95.0)
            return AnomalyDetector.from_stats(model, stats, percentile=percentile)
        else:
            st.warning(f"Watchdog model files not found: {ae_path.name} / {th_path.name}")
    except Exception as e:
        st.warning(f"Watchdog not loaded: {e}")
    return None


@st.cache_resource
def try_load_safety(threshold: float):
    try:
        from safety.safety_controller import SafetyController
        from safety.safety_rules import load_safety_config
        lidar_max = CONFIG.get("lidar", {}).get("max_range", 10.0)
        scfg = load_safety_config(CONFIG.get("safety", {}))
        return SafetyController(scfg, threshold=threshold, lidar_max_range=lidar_max)
    except Exception as e:
        st.warning(f"Safety controller not loaded: {e}")
    return None


# ---- Session state initialization ----
def init_session():
    defaults = {
        "running": False,
        "env": None,
        "obs": None,
        "info": {},
        "step": 0,
        "total_reward": 0.0,
        "episode": 0,
        "trajectory": [],
        "anomaly_scores": [],
        "safety_states": [],
        "lidar_obs": np.ones(36, dtype=np.float32),
        "anomaly_score": 0.0,
        "safety_state": "NORMAL",
        "is_ood": False,
        "collision": False,
        "goal_reached": False,
        "disturbance_active": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ---- Sidebar ----
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    scenario = st.selectbox(
        "Scenario",
        ["normal", "gaussian_noise", "missing_readings",
         "corrupt_sector", "partial_blockage"],
        index=0,
    )

    enable_noise       = st.toggle("Enable Sensor Noise", value=False)
    enable_disturbance = st.toggle("Enable Disturbance", value=True)
    disturbance_start  = st.slider("Disturbance Start Step", 0, 300, 100)
    show_lidar_rays    = st.toggle("Show LiDAR Rays", value=True)
    speed_multiplier   = st.slider("Simulation Speed", 0.1, 3.0, 1.0, 0.1)
    seed               = st.number_input("Episode Seed", value=42, min_value=0)

    st.markdown("---")
    st.markdown("### Models")
    ppo_loaded  = st.empty()
    wd_loaded   = st.empty()
    safe_loaded = st.empty()


# ---- Header ----
st.markdown("""
<div style="text-align:center; padding: 20px 0 10px 0;">
  <h1 style="font-size:2.4rem; background: linear-gradient(90deg,#66aaff,#aa66ff);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    🤖 ULHN Simulator
  </h1>
  <p style="color:#8888cc; font-size:1rem;">
    Uncertainty-aware Learning Navigation
  </p>
</div>
""", unsafe_allow_html=True)


# ---- Load models ----
ppo = try_load_ppo()
watchdog = try_load_watchdog()
threshold = watchdog.threshold if watchdog else 0.083
safety = try_load_safety(threshold)

ppo_loaded.markdown(f"PPO: {'✅ Loaded' if ppo else '❌ Not trained'}")
wd_loaded.markdown(f"Watchdog: {'✅ Loaded' if watchdog else '❌ Not trained'}")
safe_loaded.markdown(f"Safety: {'✅ Ready' if safety else '❌ Unavailable'}")


# ---- Control buttons ----
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
with col_btn1:
    if st.button("▶ Start", use_container_width=True):
        st.session_state.running = True
        st.session_state.env = None   # Force reset

with col_btn2:
    if st.button("⏹ Reset", use_container_width=True):
        st.session_state.running = False
        st.session_state.env = None
        st.session_state.step = 0
        st.session_state.total_reward = 0.0
        st.session_state.trajectory = []
        st.session_state.anomaly_scores = []
        st.session_state.safety_states = []
        st.session_state.collision = False
        st.session_state.goal_reached = False


# ---- Main layout ----
col_world, col_status = st.columns([3, 1])

with col_status:
    st.markdown("### System Status")
    status_placeholder = st.empty()
    metric_placeholder = st.empty()

with col_world:
    world_placeholder = st.empty()

lidar_placeholder = st.empty()
anomaly_placeholder = st.empty()

# ---- Simulation loop ----
if st.session_state.running and ppo is not None:
    from environment.ulhn_env import ULHNEnv
    from experiments.disturbances import (
        gaussian_noise as gn_fn, missing_readings as mr_fn,
        corrupt_sector as cs_fn, partial_blockage as pb_fn,
    )

    disturb_fns = {
        "gaussian_noise": gn_fn,
        "missing_readings": mr_fn,
        "corrupt_sector": lambda obs, **kw: cs_fn(obs, start_ray=0, width_rays=9),
        "partial_blockage": lambda obs, **kw: pb_fn(obs, centre_ray=0, half_width=6),
        "normal": lambda obs, **kw: obs,
    }
    disturb_fn = disturb_fns.get(scenario, disturb_fns["normal"])

    # Create / reset environment
    if st.session_state.env is None:
        cfg = dict(CONFIG)
        if enable_noise:
            cfg.setdefault("lidar", {})["noise_enabled"] = True
        env = ULHNEnv(config=cfg, render_mode=None)
        obs, info = env.reset(seed=int(seed))
        st.session_state.env = env
        st.session_state.obs = obs
        st.session_state.info = info
        st.session_state.step = 0
        st.session_state.total_reward = 0.0
        st.session_state.trajectory = [(info["x"], info["y"])]
        st.session_state.anomaly_scores = []
        st.session_state.safety_states = []
        if watchdog:
            watchdog.reset()
        if safety:
            safety.reset()

    env = st.session_state.env
    obs = st.session_state.obs
    step = st.session_state.step

    if not (st.session_state.collision or st.session_state.goal_reached):
        # Apply disturbance
        num_rays = env.lidar.num_rays
        lidar_obs = obs[:num_rays].copy()
        dist_active = enable_disturbance and step >= disturbance_start
        if dist_active and scenario != "normal":
            lidar_obs = disturb_fn(lidar_obs)
            obs_for_agent = obs.copy()
            obs_for_agent[:num_rays] = lidar_obs
        else:
            obs_for_agent = obs

        # PPO action
        action, _ = ppo.predict(obs_for_agent, deterministic=True)
        ppo_v = float(action[0]) * env.robot.max_linear_velocity
        ppo_omega = float(action[1]) * env.robot.max_angular_velocity

        # Watchdog
        anomaly_score = 0.0
        is_ood = False
        if watchdog:
            anomaly_score = watchdog.update(lidar_obs)
            is_ood = watchdog.is_ood()

        # Safety
        safety_state = "NORMAL"
        final_v, final_omega = ppo_v, ppo_omega
        if safety:
            decision = safety.step(ppo_v, ppo_omega, lidar_obs, anomaly_score)
            final_v    = decision.v_cmd
            final_omega = decision.omega_cmd
            safety_state = decision.state.name

        # Step env
        act = np.array([
            np.clip(final_v / env.robot.max_linear_velocity, -1, 1),
            np.clip(final_omega / env.robot.max_angular_velocity, -1, 1),
        ], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(act)

        st.session_state.obs = obs
        st.session_state.info = info
        st.session_state.step += 1
        st.session_state.total_reward += reward
        st.session_state.lidar_obs = lidar_obs
        st.session_state.anomaly_score = anomaly_score
        st.session_state.safety_state = safety_state
        st.session_state.is_ood = is_ood
        st.session_state.disturbance_active = dist_active
        st.session_state.trajectory.append((info["x"], info["y"]))
        st.session_state.anomaly_scores.append(anomaly_score)
        st.session_state.safety_states.append(safety_state)

        if info.get("collision", False):
            st.session_state.collision = True
        if info.get("goal_reached", False):
            st.session_state.goal_reached = True

# ---- Render status ----
ss = st.session_state
info = ss.info
state_color = {
    "NORMAL": "status-normal", "SLOWDOWN": "status-slowdown",
    "STOP": "status-stop", "REORIENT": "status-reorient", "RESUME": "status-resume",
}.get(ss.safety_state, "status-normal")

ood_class = "ood-active" if ss.is_ood else "ood-normal"
dist_badge = "🟠 ACTIVE" if ss.disturbance_active else "⚪ OFF"

with col_status:
    status_placeholder.markdown(f"""
<div class="metric-card">
  <div style="font-size:1.1rem; color:#ccf; margin-bottom:8px;">🤖 PPO</div>
  <div style="color:#44ff88; font-weight:bold;">ACTIVE</div>
</div>
<div class="metric-card">
  <div style="font-size:1.1rem; color:#ccf; margin-bottom:8px;">👁 Watchdog</div>
  <div class="{ood_class}">{'OOD DETECTED' if ss.is_ood else 'NORMAL'}</div>
  <div style="font-size:0.75rem; color:#8888cc; margin-top:4px;">
    Score: {ss.anomaly_score:.4f}<br>
    Threshold: {threshold:.4f}
  </div>
</div>
<div class="metric-card">
  <div style="font-size:1.1rem; color:#ccf; margin-bottom:8px;">🛡 Safety</div>
  <div class="{state_color}">{ss.safety_state}</div>
</div>
<div class="metric-card">
  <div style="font-size:0.9rem; color:#ccf; margin-bottom:4px;">📡 Disturbance</div>
  <div>{dist_badge}</div>
</div>
""", unsafe_allow_html=True)

    metric_placeholder.markdown(f"""
<div class="metric-card">
  <div class="metric-value">{ss.step}</div>
  <div class="metric-label">Steps</div>
</div>
<div class="metric-card">
  <div class="metric-value">{ss.total_reward:.1f}</div>
  <div class="metric-label">Cumulative Reward</div>
</div>
<div class="metric-card">
  <div class="metric-value">{info.get('goal_distance', 0):.2f}m</div>
  <div class="metric-label">Distance to Goal</div>
</div>
""", unsafe_allow_html=True)


# ---- World view ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def render_world():
    env = ss.env
    if env is None or env.obstacle_map is None:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0f0f1e")
        ax.set_facecolor("#1a1a2e")
        ax.text(0.5, 0.5, "Press START to begin simulation",
                ha="center", va="center", transform=ax.transAxes,
                color="#8888cc", fontsize=13)
        return fig

    from environment.obstacles import RectObstacle, CircleObstacle
    import math as _math

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0f0f1e")
    ax.set_facecolor("#1a1a2e")
    w, h = env.world_width, env.world_height
    ax.set_xlim(-0.3, w + 0.3)
    ax.set_ylim(-0.3, h + 0.3)
    ax.set_aspect("equal")

    # World border
    rect = patches.Rectangle((0, 0), w, h, lw=2, edgecolor="#4488ff",
                               facecolor="none", ls="--")
    ax.add_patch(rect)

    # Obstacles
    for obs in env.obstacle_map.obstacles:
        if isinstance(obs, RectObstacle):
            p = patches.Rectangle((obs.cx - obs.half_w, obs.cy - obs.half_h),
                                   2*obs.half_w, 2*obs.half_h,
                                   edgecolor="#ff6644", facecolor="#3d1a0e", lw=1.5, zorder=3)
            ax.add_patch(p)
        elif isinstance(obs, CircleObstacle):
            c = plt.Circle((obs.cx, obs.cy), obs.radius,
                            edgecolor="#ff6644", facecolor="#3d1a0e", lw=1.5, zorder=3)
            ax.add_patch(c)

    # Trajectory
    traj = ss.trajectory
    if len(traj) > 1:
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        ax.plot(xs, ys, color="#4488ff", lw=1.5, alpha=0.7, zorder=4)
        ax.plot(xs[0], ys[0], "o", markersize=8, color="#ffaa00", zorder=7, label="Start")

    # LiDAR rays
    if show_lidar_rays and env.obstacle_map is not None:
        rx, ry, rtheta = env.robot.x, env.robot.y, env.robot.theta
        fov_rad = _math.radians(env.lidar.config.fov_deg)
        n = env.lidar.num_rays
        offsets = np.linspace(-fov_rad/2, fov_rad/2, n,
                              endpoint=(env.lidar.config.fov_deg < 360.0))
        lidar = ss.lidar_obs
        ray_color = "#ff4444" if ss.is_ood else "#00ffff"
        for i, off in enumerate(offsets):
            angle = rtheta + off
            dist  = float(lidar[i]) * env.lidar.max_range
            ex = rx + dist * _math.cos(angle)
            ey = ry + dist * _math.sin(angle)
            ax.plot([rx, ex], [ry, ey], color=ray_color, alpha=0.2, lw=0.5, zorder=2)

    # Goal
    gx, gy = env.goal_position
    goal_circle = plt.Circle((gx, gy), env.goal_radius,
                              color="#00ff88", alpha=0.35, zorder=5)
    ax.add_patch(goal_circle)
    ax.plot(gx, gy, "*", markersize=14, color="#00ff88", zorder=6, label="Goal")

    # Robot
    rx, ry, rtheta = env.robot.x, env.robot.y, env.robot.theta
    robot_color = "#ff4444" if ss.collision else ("#ffaa00" if ss.is_ood else "#4488ff")
    body = plt.Circle((rx, ry), env.robot.radius, color=robot_color, alpha=0.9, zorder=7)
    ax.add_patch(body)
    arrow_len = env.robot.radius * 1.8
    ax.annotate("", xy=(rx + arrow_len * _math.cos(rtheta),
                        ry + arrow_len * _math.sin(rtheta)),
                xytext=(rx, ry),
                arrowprops=dict(arrowstyle="->", color="#fff", lw=1.8), zorder=8)

    event_text = ""
    if ss.collision:
        event_text = "💥 COLLISION"
    elif ss.goal_reached:
        event_text = "🎯 GOAL REACHED"
    elif ss.disturbance_active:
        event_text = f"⚡ {scenario.upper()}"

    ax.set_title(
        f"ULHN  |  Step {ss.step}  |  Ep {ss.episode}  {event_text}",
        color="#ccf", fontsize=11,
    )
    ax.tick_params(colors="#aaa")
    ax.legend(facecolor="#222244", labelcolor="#ccf", fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


with col_world:
    world_placeholder.pyplot(render_world(), use_container_width=True)


# ---- LiDAR view ----
from visualization.lidar_view import lidar_polar_figure

with lidar_placeholder.container():
    st.markdown("### 📡 LiDAR Scan")
    lc1, lc2 = st.columns(2)
    with lc1:
        fig_polar = lidar_polar_figure(
            ss.lidar_obs,
            num_rays=CONFIG.get("lidar", {}).get("num_rays", 36),
            anomaly_score=ss.anomaly_score,
            threshold=threshold,
        )
        st.pyplot(fig_polar, use_container_width=True)
        plt.close(fig_polar)

    with lc2:
        if ss.anomaly_scores:
            fig_anom, ax_anom = plt.subplots(figsize=(6, 3), facecolor="#0f0f1e")
            ax_anom.set_facecolor("#1a1a2e")
            steps_list = list(range(len(ss.anomaly_scores)))
            ax_anom.plot(steps_list, ss.anomaly_scores, color="#00ccff", lw=1.5)
            ax_anom.axhline(threshold, color="#ff4444", lw=2, ls="--", label="Threshold")
            ax_anom.fill_between(steps_list, ss.anomaly_scores, threshold,
                                 where=[s > threshold for s in ss.anomaly_scores],
                                 color="#ff4444", alpha=0.2)
            if ss.disturbance_active:
                dist_s = CONFIG.get("experiment", {}).get("demo_disturbance_start_step", disturbance_start)
                ax_anom.axvline(disturbance_start, color="#ff8844", ls=":", lw=1.5)
            ax_anom.set_xlabel("Step", color="#aaa")
            ax_anom.set_ylabel("Anomaly Score", color="#aaa")
            ax_anom.set_title("Anomaly Score vs Threshold", color="#ccf", fontsize=11)
            ax_anom.legend(facecolor="#222244", labelcolor="#ccf")
            ax_anom.tick_params(colors="#aaa")
            fig_anom.tight_layout()
            st.pyplot(fig_anom, use_container_width=True)
            plt.close(fig_anom)


# ---- Metrics bar ----
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
success_count = sum(1 for s in ss.safety_states if s == "NORMAL")
ood_count = sum(1 for a in ss.anomaly_scores if a > threshold)
m1.metric("Steps",          ss.step)
m2.metric("Cumulative Reward", f"{ss.total_reward:.2f}")
m3.metric("OOD Detections", ood_count)
m4.metric("Safety State",   ss.safety_state)


# ---- Auto-rerun ----
if ss.running and not (ss.collision or ss.goal_reached):
    time.sleep(max(0.02, 0.1 / speed_multiplier))
    st.rerun()
elif ss.running and (ss.collision or ss.goal_reached):
    st.success("Episode finished! Press Reset to start a new episode.")
