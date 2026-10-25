"""Walk-forward replay: run the learning loop over past prices without peeking ahead.

On each simulated day t the engine
  1. learns from the outcome that has just become observable (the forecast made on day
     t - horizon, whose result is only known now),
  2. forecasts using only information available on day t.
That is what the live agent does in real time, so one function serves two purposes: a
backtest that doesn't cheat, and a way to warm-start the ensemble weights from history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from agentlab.finance.evaluation import Metrics, evaluate
from agentlab.finance.experts import (
    EXPERT_NAMES,
    Ensemble,
    base_rate_series,
    rule_expert_probs,
    walk_forward_ml,
)
from agentlab.finance.features import build_features, direction_label, forward_return

logger = logging.getLogger(__name__)

MIN_PRICES = 400  # enough for indicator warm-up + training + an evaluation window


@dataclass
class ReplayResult:
    horizon: int
    probs: pd.DataFrame  # per-day expert probabilities + 'ensemble' (rows from the replay start)
    weights_history: pd.DataFrame
    final_weights: dict[str, float]
    metrics: dict[str, Metrics]  # per expert and 'ensemble', on days with a known outcome
    n_days: int


def replay(
    prices: pd.DataFrame,
    horizon: int,
    *,
    min_train: int = 250,
    retrain_every: int = 21,
    eta: float = 3.0,
    alpha: float = 0.01,
) -> ReplayResult:
    if len(prices) < MIN_PRICES:
        raise ValueError(f"Need at least {MIN_PRICES} daily prices, got {len(prices)}.")
    feats = build_features(prices)
    fwd = forward_return(prices, horizon)
    y = direction_label(fwd)

    probs = rule_expert_probs(feats)
    probs["base_rate"] = base_rate_series(y, horizon)
    probs["ml_logreg"] = walk_forward_ml(feats, y, horizon, min_train=min_train,
                                         retrain_every=retrain_every)
    probs = probs[list(EXPERT_NAMES)]

    # Start the replay on the first day every expert has a forecast. The logistic model needs
    # `min_train` labelled rows first, so it is always the last one to come online.
    ready = probs["ml_logreg"].notna().to_numpy()
    if not ready.any():
        raise ValueError("Not enough clean history to train the model; fetch a longer period.")
    start = int(np.argmax(ready))
    logger.debug("Replay: %d prices, horizon %d, first usable day index %d", len(prices), horizon,
                 start)

    ensemble = Ensemble(eta=eta, alpha=alpha)
    y_arr = y.to_numpy()
    p_matrix = probs.to_numpy()
    p_ens = np.full(len(prices), np.nan)
    weights_rows: list[list[float]] = []
    for t in range(start, len(prices)):
        # Index of the forecast whose horizon ends today. Its outcome (price at t vs price at
        # t - horizon) is now known, so the ensemble may learn from it, but nothing later.
        learned = t - horizon
        if learned >= start and np.isfinite(y_arr[learned]):
            ensemble.update(dict(zip(EXPERT_NAMES, p_matrix[learned], strict=True)), y_arr[learned])
        p_ens[t] = ensemble.predict(dict(zip(EXPERT_NAMES, p_matrix[t], strict=True)))
        weights_rows.append([ensemble.weights[n] for n in EXPERT_NAMES])

    probs = probs.copy()
    probs["ensemble"] = p_ens
    probs = probs.iloc[start:]
    weights_history = pd.DataFrame(weights_rows, index=probs.index, columns=list(EXPERT_NAMES))

    outcome = y.iloc[start:].to_numpy()
    fwd_arr = fwd.iloc[start:].to_numpy()
    p_base = probs["base_rate"].to_numpy()
    metrics = {
        name: evaluate(probs[name].to_numpy(), outcome, fwd_arr, p_base, horizon)
        for name in (*EXPERT_NAMES, "ensemble")
    }
    logger.info("Replay: %d days from %s, ensemble accuracy %.3f", len(probs),
                probs.index[0].date(), metrics["ensemble"].accuracy)
    return ReplayResult(horizon, probs, weights_history, dict(ensemble.weights), metrics,
                        len(probs))
