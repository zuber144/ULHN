"""
evaluation/statistics.py
========================
Statistical helper functions for evaluation.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


def confidence_interval_95(data: List[float]) -> Tuple[float, float]:
    """Compute 95% bootstrap confidence interval."""
    arr = np.array(data)
    n = len(arr)
    if n < 2:
        return float(arr[0]) if n == 1 else (0.0, 0.0)
    bootstraps = [np.mean(np.random.choice(arr, size=n, replace=True))
                  for _ in range(1000)]
    lo = float(np.percentile(bootstraps, 2.5))
    hi = float(np.percentile(bootstraps, 97.5))
    return lo, hi


def wilcoxon_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Wilcoxon signed-rank test for paired comparisons."""
    from scipy import stats
    stat, p_value = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p_value)
