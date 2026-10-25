"""Scoring for probability forecasts, always measured against naive baselines.

A forecaster that says "up" 60% of the time looks accurate in a market that rises 60% of the
time, so accuracy alone is meaningless. Everything here is reported next to the baseline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Metrics:
    n: int
    accuracy: float
    majority_accuracy: float  # accuracy of always predicting the more common outcome
    brier: float
    brier_baseline: float  # Brier of the base-rate forecast
    brier_skill: float  # 1 - brier/brier_baseline; > 0 means better than the base rate
    z_score: float  # accuracy vs majority baseline, with overlap-adjusted sample size
    avg_ret_when_up: float
    avg_ret_when_down: float
    verdict: str


def evaluate(
    p: np.ndarray, y: np.ndarray, fwd_ret: np.ndarray, p_base: np.ndarray, horizon: int
) -> Metrics:
    """Score probability forecasts ``p`` of the binary outcome ``y``.

    Args:
        p: forecast P(up) per sample.   y: realised outcome (1 = up).
        fwd_ret: realised forward return.   p_base: the naive base-rate forecast per sample.
        horizon: days ahead. Consecutive samples overlap by ``horizon - 1`` days, so the
            effective sample size is roughly ``n / horizon``; the z-score accounts for this.
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    fwd_ret, p_base = np.asarray(fwd_ret, float), np.asarray(p_base, float)
    mask = np.isfinite(p) & np.isfinite(y) & np.isfinite(fwd_ret) & np.isfinite(p_base)
    p, y, fwd_ret, p_base = p[mask], y[mask], fwd_ret[mask], p_base[mask]
    n = len(p)
    if n == 0:
        logger.debug("evaluate(): no usable samples after dropping NaNs")
        return Metrics(0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan,
                       math.nan, "insufficient data")

    pred_up = p > 0.5
    accuracy = float(np.mean(pred_up == (y > 0.5)))
    up_rate = float(y.mean())
    majority = max(up_rate, 1 - up_rate)
    brier = float(np.mean((p - y) ** 2))
    brier_base = float(np.mean((p_base - y) ** 2))
    # Brier skill score: 0 = no better than the base-rate forecast, 1 = perfect, <0 = worse.
    skill = 1 - brier / brier_base if brier_base > 0 else math.nan
    # Samples that are `horizon` days apart or less share price moves, so they are not
    # independent. Dividing n by the horizon is a rough effective sample size; using the raw n
    # would make the z-score look far more convincing than it is.
    n_eff = max(n / horizon, 1.0)
    # Standard error of an accuracy figure if the forecaster were just guessing the majority.
    sd = math.sqrt(max(majority * (1 - majority), 1e-9) / n_eff)
    z = (accuracy - majority) / sd
    up_ret = float(fwd_ret[pred_up].mean()) if pred_up.any() else math.nan
    down_ret = float(fwd_ret[~pred_up].mean()) if (~pred_up).any() else math.nan

    # 1.64 is the one-sided 5% threshold. We also require positive skill so a forecaster
    # cannot pass on accuracy alone while its probabilities are badly calibrated.
    if n_eff < 30:
        verdict = "too few independent samples to judge"
    elif z > 1.64 and skill > 0:
        verdict = "possible edge (still verify out-of-sample)"
    else:
        verdict = "no demonstrable edge over naive baselines"
    return Metrics(n, accuracy, majority, brier, brier_base, skill, float(z), up_ret, down_ret,
                   verdict)
