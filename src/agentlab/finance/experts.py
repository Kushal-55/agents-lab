"""Forecasting experts and the weighted ensemble that learns which ones to trust.

Each expert independently outputs P(price is higher after `horizon` days). The ensemble
combines them with weights learned online from real outcomes, using the Hedge
(multiplicative weights) algorithm with fixed-share mixing:

    w_i  <-  w_i * exp(-eta * loss_i)        loss = Brier score (p - outcome)^2
    w    <-  (1 - alpha) * normalise(w) + alpha / N

Experts that keep being right gain weight and ones that keep being wrong lose it. The ``alpha``
term puts a floor under every weight, so an expert that has been wrong for a while can still
recover if the market changes character.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agentlab.finance.model import LogisticModel

logger = logging.getLogger(__name__)

RULE_EXPERTS = ("trend", "momentum", "mean_reversion")
EXPERT_NAMES = (*RULE_EXPERTS, "base_rate", "ml_logreg")
PROB_FLOOR, PROB_CEIL = 0.05, 0.95  # never claim certainty
MAX_TILT = 0.10  # rule experts move at most +/-0.10 away from 0.5 (markets are noisy)


def _tilt(z: pd.Series) -> pd.Series:
    """Squash a signal strength ``z`` into a probability within MAX_TILT of 0.5 (tanh saturates)."""
    return (0.5 + MAX_TILT * np.tanh(z.fillna(0.0))).clip(PROB_FLOOR, PROB_CEIL)


def rule_expert_probs(feats: pd.DataFrame) -> pd.DataFrame:
    """Hand-written signals, scaled by recent volatility so a 'big' move is relative.

    * trend           - price above its 50-day average suggests continuation.
    * momentum        - strong 20-day return suggests continuation.
    * mean_reversion  - overbought RSI / upper Bollinger band suggests a pullback (and v.v.).
    """
    # Typical size of a 20-day move given recent daily volatility. Dividing by it turns raw
    # returns into "how many normal moves is this", which is comparable across calm/wild periods.
    scale = (feats["vol_20"] * math.sqrt(20)).replace(0, np.nan)
    out = pd.DataFrame(index=feats.index)
    out["trend"] = _tilt(feats["trend_gap"] / scale)
    out["momentum"] = _tilt(feats["ret_20"] / scale)
    # RSI is centred on 50 and scaled so +-25 maps to +-1, matching the Bollinger position range.
    # The minus sign flips it: overbought -> expect a pullback.
    out["mean_reversion"] = _tilt(-((feats["rsi"] - 50) / 25 + feats["bb_pos"]) / 2)
    return out


def base_rate_series(y: pd.Series, horizon: int) -> pd.Series:
    """Fraction of 'up' outcomes among labels that were already known at each date."""
    known = y.shift(horizon)  # the label of day t-h is revealed on day t
    # Running mean of the labels revealed so far; 0.5 until the first label shows up.
    return known.expanding().mean().fillna(0.5).clip(PROB_FLOOR, PROB_CEIL)


def walk_forward_ml(
    feats: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    *,
    min_train: int = 250,
    retrain_every: int = 21,
    max_train: int = 1500,
) -> pd.Series:
    """Out-of-sample probabilities from a logistic model retrained on an expanding window.

    The model used for day t is fitted only on days i with ``i + horizon <= t`` - exactly the
    labels that would have been observable on day t. NaN where there is not enough history.
    """
    x = feats.to_numpy(dtype=float)
    labels = y.to_numpy(dtype=float)
    valid = np.isfinite(x).all(axis=1)
    out = np.full(len(feats), np.nan)
    first_possible = int(np.argmax(valid)) if valid.any() else len(feats)
    retrains = 0
    # Every `retrain_every` days, refit on everything knowable at t0 and use that model for the
    # next block of days. Refitting daily would give nearly the same answers at 21x the cost.
    for t0 in range(first_possible + min_train + horizon, len(feats), retrain_every):
        # Training rows are those whose label has been revealed by t0: row i needs
        # i + horizon <= t0, hence the slice up to (and including) t0 - horizon.
        known = slice(0, t0 - horizon + 1)
        idx = np.flatnonzero(valid[known] & np.isfinite(labels[known]))
        idx = idx[-max_train:]  # cap the window so the cost per refit stays bounded
        if len(idx) < min_train:
            continue
        model = LogisticModel().fit(x[idx], labels[idx])
        retrains += 1
        block = np.arange(t0, min(t0 + retrain_every, len(feats)))
        block = block[valid[block]]
        out[block] = np.clip(model.predict_proba(x[block]), PROB_FLOOR, PROB_CEIL)
    logger.debug("walk_forward_ml: %d refits over %d rows (horizon %d)", retrains, len(feats),
                 horizon)
    return pd.Series(out, index=feats.index, name="ml_logreg")


@dataclass
class Ensemble:
    """Online-learned weighted average of expert probabilities."""

    weights: dict[str, float] = field(
        default_factory=lambda: {n: 1.0 / len(EXPERT_NAMES) for n in EXPERT_NAMES}
    )
    eta: float = 3.0
    alpha: float = 0.01

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if total <= 0 or any(w < 0 for w in self.weights.values()):
            raise ValueError("Weights must be non-negative with a positive sum.")
        self.weights = {k: v / total for k, v in self.weights.items()}

    def predict(self, probs: dict[str, float]) -> float:
        """Weighted mean over the experts that produced a forecast."""
        avail = {k: p for k, p in probs.items() if k in self.weights and np.isfinite(p)}
        if not avail:
            return 0.5
        total = sum(self.weights[k] for k in avail)
        p = sum(self.weights[k] * v for k, v in avail.items()) / total
        return float(np.clip(p, PROB_FLOOR, PROB_CEIL))

    def update(self, probs: dict[str, float], outcome_up: bool | int | float) -> None:
        """Learn from one revealed outcome. Experts without a forecast count as neutral (0.5)."""
        y = float(outcome_up)
        new = {}
        for name, w in self.weights.items():
            p = probs.get(name, 0.5)
            p = 0.5 if not np.isfinite(p) else p
            # Hedge step: shrink the weight by exp(-eta * Brier loss). A confident wrong forecast
            # (loss near 1) is punished far more than a hedged one (p near 0.5, loss 0.25).
            new[name] = w * math.exp(-self.eta * (p - y) ** 2)
        total = sum(new.values())
        n = len(new)
        # Renormalise, then mix in a little uniform weight (fixed share) as the floor.
        self.weights = {k: (1 - self.alpha) * v / total + self.alpha / n for k, v in new.items()}
        logger.debug("Ensemble update (outcome=%d): %s", int(y),
                     {k: round(v, 3) for k, v in self.weights.items()})
