"""
watchdog/__init__.py
"""
import sys
from pathlib import Path

# Extend __path__ to include third-party PyPI 'watchdog' package if installed,
# preventing namespace collisions with libraries like Streamlit that import watchdog.observers/events.
try:
    import site
    _site_dirs = []
    if hasattr(site, "getsitepackages"):
        _site_dirs.extend(site.getsitepackages())
    if hasattr(site, "getusersitepackages"):
        _site_dirs.append(site.getusersitepackages())
    for sp in _site_dirs:
        pypi_wd = Path(sp) / "watchdog"
        if pypi_wd.is_dir() and str(pypi_wd) not in __path__:
            __path__.append(str(pypi_wd))
except Exception:
    pass

from watchdog.autoencoder import LiDARAutoencoder, save_autoencoder, load_autoencoder
from watchdog.anomaly_detector import AnomalyDetector
from watchdog.threshold import ThresholdStats, compute_threshold_stats

__all__ = [
    "LiDARAutoencoder",
    "save_autoencoder",
    "load_autoencoder",
    "AnomalyDetector",
    "ThresholdStats",
    "compute_threshold_stats",
]

