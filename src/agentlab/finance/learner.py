"""The forecaster: predict, journal, wait, score, adapt.

1. ``forecast``: every expert gives P(up in `horizon` days). The ensemble blends them with its
   current learned weights and the result is written to the journal straight away.
2. Time passes and real prices arrive.
3. ``resolve``: for each journalled forecast whose horizon has elapsed, compare with what
   happened, update the weights (Hedge) and occasionally write a short lesson.

The first time a (ticker, horizon) pair is seen there is no history of live forecasts, so the
weights are warm-started by replaying the past price history through the same loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from agentlab.finance.backtest import ReplayResult, replay
from agentlab.finance.data import PriceProvider, validate_ticker
from agentlab.finance.evaluation import Metrics
from agentlab.finance.experts import EXPERT_NAMES, Ensemble, rule_expert_probs
from agentlab.finance.features import (
    FEATURE_COLUMNS,
    build_features,
    direction_label,
    forward_return,
)
from agentlab.finance.journal import Journal, PredictionRecord
from agentlab.finance.model import LogisticModel

logger = logging.getLogger(__name__)

MIN_HORIZON, MAX_HORIZON = 1, 60
LESSON_EVERY = 5
LEAN_THRESHOLD = 0.05  # |p - 0.5| below this is reported as "no clear edge"


@dataclass(frozen=True)
class Forecast:
    ticker: str
    horizon: int
    as_of: str
    entry_price: float
    p_up: float
    stance: str
    expert_probs: dict[str, float]
    weights: dict[str, float]
    evidence: Metrics | None  # walk-forward track record (None if skipped for speed)
    drivers: list[tuple[str, float]]  # (feature, z-score) most unusual today
    journal_id: int | None
    newly_recorded: bool
    warm_started: bool


@dataclass
class ResolveSummary:
    resolved: int = 0
    still_pending: int = 0
    skipped: int = 0
    lessons: list[str] = field(default_factory=list)


def _check_horizon(horizon: int) -> int:
    if not isinstance(horizon, int) or not MIN_HORIZON <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be an integer between {MIN_HORIZON} and {MAX_HORIZON}")
    return horizon


def stance_for(p_up: float) -> str:
    if p_up >= 0.5 + LEAN_THRESHOLD:
        return "LEAN UP"
    if p_up <= 0.5 - LEAN_THRESHOLD:
        return "LEAN DOWN"
    return "NO CLEAR EDGE"


class Forecaster:
    def __init__(self, provider: PriceProvider, journal: Journal, *, eta: float = 3.0,
                 alpha: float = 0.01, min_train: int = 250) -> None:
        self._provider, self._journal = provider, journal
        self._eta, self._alpha, self._min_train = eta, alpha, min_train

    # ---- public API ---------------------------------------------------------------------
    def backtest(self, ticker: str, horizon: int = 5, period: str = "5y") -> ReplayResult:
        ticker, horizon = validate_ticker(ticker), _check_horizon(horizon)
        prices = self._provider.fetch(ticker, period)
        return replay(prices, horizon, min_train=self._min_train, eta=self._eta,
                      alpha=self._alpha)

    def forecast(self, ticker: str, horizon: int = 5, period: str = "5y",
                 record: bool = True, with_evidence: bool = True) -> Forecast:
        ticker, horizon = validate_ticker(ticker), _check_horizon(horizon)
        prices = self._provider.fetch(ticker, period)

        stored = self._journal.get_weights(ticker, horizon)
        warm_started = stored is None
        history: ReplayResult | None = None
        if warm_started or with_evidence:  # replay is the expensive step; skip when possible
            history = replay(prices, horizon, min_train=self._min_train, eta=self._eta,
                             alpha=self._alpha)
        if warm_started:
            self._journal.set_weights(ticker, horizon, history.final_weights)
            logger.info("Warm-started weights for %s/%dd from %d days of history", ticker,
                        horizon, history.n_days)
        # Live weights from the journal win; otherwise use the ones the replay just produced.
        weights = stored or history.final_weights

        feats = build_features(prices)
        fwd = forward_return(prices, horizon)
        y = direction_label(fwd)
        latest = feats.iloc[[-1]]
        if latest.isna().any(axis=None):
            raise ValueError(f"{ticker}: latest features are incomplete; need more history.")

        probs: dict[str, float] = rule_expert_probs(latest).iloc[0].to_dict()
        # y is NaN for the last `horizon` days (outcome not known yet), so dropna() leaves only
        # labels that are genuinely observable today.
        probs["base_rate"] = float(np.clip(y.dropna().mean(), 0.05, 0.95))
        train = feats.join(y.rename("y")).dropna()  # rows whose outcome is already known
        if len(train) >= self._min_train:
            model = LogisticModel().fit(train[FEATURE_COLUMNS].to_numpy(), train["y"].to_numpy())
            probs["ml_logreg"] = float(np.clip(model.predict_proba(latest.to_numpy())[0], 0.05,
                                               0.95))
        else:
            logger.info("%s: only %d labelled rows, skipping the ML expert", ticker, len(train))
        probs = {k: float(v) for k, v in probs.items() if np.isfinite(v)}

        ensemble = Ensemble(weights=dict(weights), eta=self._eta, alpha=self._alpha)
        p_up = ensemble.predict(probs)

        # What is unusual about today? z-score of each feature vs its own history.
        hist = feats.dropna()
        z = ((latest.iloc[0] - hist.mean()) / hist.std().replace(0, np.nan)).dropna()
        drivers = sorted(z.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]

        as_of = prices.index[-1].date().isoformat()
        entry = float(prices["Close"].iloc[-1])
        journal_id, created = (None, False)
        if record:
            journal_id, created = self._journal.record_prediction(
                ticker, horizon, as_of, p_up, probs, ensemble.weights, entry
            )
        result = Forecast(
            ticker=ticker, horizon=horizon, as_of=as_of, entry_price=entry, p_up=p_up,
            stance=stance_for(p_up), expert_probs=probs, weights=dict(ensemble.weights),
            evidence=history.metrics["ensemble"] if history else None,
            drivers=[(k, float(v)) for k, v in drivers],
            journal_id=journal_id, newly_recorded=created, warm_started=warm_started,
        )
        logger.info("Forecast %s/%dd as_of=%s p_up=%.3f stance=%s recorded=%s", ticker, horizon,
                    as_of, p_up, result.stance, created)
        return result

    def resolve(self, period: str = "5y") -> ResolveSummary:
        """Score every journalled forecast whose horizon has elapsed, and learn from it."""
        summary = ResolveSummary()
        pending = self._journal.predictions(resolved=False)
        by_ticker: dict[str, list[PredictionRecord]] = {}
        for rec in pending:
            by_ticker.setdefault(rec.ticker, []).append(rec)

        for ticker, records in by_ticker.items():
            try:
                prices = self._provider.fetch(ticker, period)
            except Exception as exc:  # noqa: BLE001 - one bad ticker must not block the rest
                logger.warning("Cannot resolve %s: %s", ticker, exc)
                summary.skipped += len(records)
                continue
            close = prices["Close"]
            resolved_now: dict[int, list[PredictionRecord]] = {}
            for rec in sorted(records, key=lambda r: r.as_of):  # learn in the order things happened
                # Locate the forecast date in the fresh price history. searchsorted gives the
                # first index >= the date, so we also check it is an exact match.
                pos = close.index.searchsorted(pd.Timestamp(rec.as_of))
                if pos >= len(close) or close.index[pos] != pd.Timestamp(rec.as_of):
                    logger.warning("%s %s not found in fresh data; skipping", ticker, rec.as_of)
                    summary.skipped += 1
                    continue
                # The outcome is the close `horizon` trading days after as_of. If that bar does
                # not exist yet, the forecast has to wait.
                if pos + rec.horizon >= len(close):
                    logger.debug("#%d %s/%dd not mature yet", rec.id, ticker, rec.horizon)
                    summary.still_pending += 1
                    continue
                entry, exit_ = float(close.iloc[pos]), float(close.iloc[pos + rec.horizon])
                outcome = exit_ > entry
                # Start from the latest stored weights (they already include earlier lessons),
                # falling back to the weights recorded with the forecast.
                weights = self._journal.get_weights(ticker, rec.horizon) or rec.weights
                ensemble = Ensemble(weights=weights, eta=self._eta, alpha=self._alpha)
                ensemble.update(rec.expert_probs, outcome)
                self._journal.apply_resolution(rec.id, outcome, exit_, ticker, rec.horizon,
                                               ensemble.weights)
                summary.resolved += 1
                resolved_now.setdefault(rec.horizon, []).append(replace(rec, outcome_up=outcome))
            for horizon, new in resolved_now.items():
                done = self._journal.predictions(resolved=True, ticker=ticker)
                done = [r for r in done if r.horizon == horizon]
                # Write a lesson each time the scored total crosses a multiple of LESSON_EVERY
                # (compare the bucket before and after this batch). It summarises the latest 10.
                if (len(done) // LESSON_EVERY) > ((len(done) - len(new)) // LESSON_EVERY):
                    lesson = self._make_lesson(ticker, horizon, done[-2 * LESSON_EVERY:])
                    self._journal.add_lesson(ticker, horizon, lesson)
                    summary.lessons.append(f"{ticker} {horizon}d: {lesson}")
        logger.info("Resolved %d, pending %d, skipped %d", summary.resolved,
                    summary.still_pending, summary.skipped)
        return summary

    # ---- helpers ------------------------------------------------------------------------
    def _make_lesson(self, ticker: str, horizon: int, recs: list[PredictionRecord]) -> str:
        hits = {name: [] for name in EXPERT_NAMES}
        ens_hits = []
        for r in recs:
            actual = bool(r.outcome_up)
            ens_hits.append((r.p_up > 0.5) == actual)
            for name, p in r.expert_probs.items():
                if name in hits:
                    hits[name].append((p > 0.5) == actual)
        rates = {n: float(np.mean(v)) for n, v in hits.items() if v}
        best = max(rates, key=rates.get)
        worst = min(rates, key=rates.get)
        weights = self._journal.get_weights(ticker, horizon) or {}
        top = max(weights, key=weights.get) if weights else "n/a"
        return (
            f"Scored {len(recs)} forecast(s): ensemble direction hit-rate "
            f"{np.mean(ens_hits):.0%}. Best expert {best} ({rates[best]:.0%}), worst {worst} "
            f"({rates[worst]:.0%}). Highest weight now: {top}."
        )

