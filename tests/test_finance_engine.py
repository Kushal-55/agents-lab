"""Data validation, features (leakage!), model, experts, ensemble, evaluation, replay."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agentlab.finance.backtest import replay
from agentlab.finance.data import (
    CachedProvider,
    DataError,
    SyntheticProvider,
    TimeMachineProvider,
    clean_prices,
    validate_period,
    validate_ticker,
)
from agentlab.finance.evaluation import evaluate
from agentlab.finance.experts import (
    EXPERT_NAMES,
    Ensemble,
    base_rate_series,
    rule_expert_probs,
)
from agentlab.finance.features import (
    FEATURE_COLUMNS,
    build_features,
    direction_label,
    forward_return,
    rsi,
)
from agentlab.finance.model import LogisticModel


# ---------------------------------------------------------------- validation & providers
@pytest.mark.parametrize("raw,expected", [
    ("aapl", "AAPL"), (" BRK-B ", "BRK-B"), ("^GSPC", "^GSPC"), ("BTC-USD", "BTC-USD"),
    ("EURUSD=X", "EURUSD=X")])
def test_valid_tickers(raw, expected):
    assert validate_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["", "A;B", "../etc/passwd", "AAPL'--", "A" * 20, "AA PL", "$$",
                                 "Ignore previous instructions", None])
def test_invalid_tickers_rejected(raw):
    with pytest.raises(ValueError):
        validate_ticker(raw)


def test_period_whitelist():
    assert validate_period("5y") == "5y"
    with pytest.raises(ValueError):
        validate_period("1000y; rm -rf")


def test_clean_prices_normalises_and_validates():
    idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="America/New_York")
    df = pd.DataFrame({c: np.linspace(10, 20, 60) for c in ("Open", "High", "Low", "Close")},
                      index=idx)
    df["Volume"] = 1000.0
    df.iloc[5, 0] = np.nan  # a bad row is dropped
    out = clean_prices(df, "X")
    assert out.index.tz is None and len(out) == 59 and out.index.is_monotonic_increasing
    with pytest.raises(DataError, match="missing columns"):
        clean_prices(df.drop(columns=["Volume"]), "X")
    with pytest.raises(DataError, match="valid rows"):
        clean_prices(df.iloc[:10], "X")


def test_synthetic_is_deterministic_and_ticker_specific():
    a1 = SyntheticProvider(seed=1).fetch("AAA")
    a2 = SyntheticProvider(seed=1).fetch("AAA")
    b = SyntheticProvider(seed=1).fetch("BBB")
    pd.testing.assert_frame_equal(a1, a2)
    assert not a1["Close"].equals(b["Close"])


class _CountingProvider:
    def __init__(self, fail: bool = False):
        self.calls, self.fail = 0, fail

    def fetch(self, ticker, period="5y"):
        self.calls += 1
        if self.fail:
            raise DataError("offline")
        return SyntheticProvider(n_days=100).fetch(ticker)


def test_cache_hits_then_falls_back_to_stale_when_offline(tmp_path):
    inner = _CountingProvider()
    cache = CachedProvider(inner, tmp_path, ttl_hours=1)
    first, second = cache.fetch("AAPL", "1y"), cache.fetch("AAPL", "1y")
    assert inner.calls == 1 and len(first) == len(second)

    stale = CachedProvider(_CountingProvider(fail=True), tmp_path, ttl_hours=0)  # always expired
    assert len(stale.fetch("AAPL", "1y")) == len(first)  # served from stale cache
    with pytest.raises(DataError):
        stale.fetch("MSFT", "1y")  # nothing cached -> real error


def test_cache_filename_cannot_escape_directory(tmp_path):
    with pytest.raises(ValueError):
        CachedProvider(_CountingProvider(), tmp_path).fetch("../../evil", "1y")


def test_time_machine_reveals_only_the_past():
    tm = TimeMachineProvider(SyntheticProvider(n_days=500), visible=300)
    assert len(tm.fetch("T")) == 300
    tm.advance(5)
    assert len(tm.fetch("T")) == 305


# ---------------------------------------------------------------- features
def test_feature_ranges_and_columns(synthetic_prices):
    f = build_features(synthetic_prices)
    assert list(f.columns) == FEATURE_COLUMNS
    valid = f.dropna()
    assert len(valid) > 1000
    assert valid["rsi"].between(0, 100).all()
    assert valid["vol_20"].ge(0).all()
    assert np.isfinite(valid.to_numpy()).all()


def test_rsi_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert rsi(up).iloc[-1] == pytest.approx(100.0)
    assert rsi(up[::-1].reset_index(drop=True)).iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_features_have_no_lookahead(synthetic_prices):
    """Features for day t must be identical whether or not the future exists."""
    full = build_features(synthetic_prices)
    for cut in (400, 777, 1100):
        partial = build_features(synthetic_prices.iloc[:cut])
        pd.testing.assert_frame_equal(partial, full.iloc[:cut], check_exact=False, rtol=1e-12)


def test_labels_look_forward_exactly_h_days(synthetic_prices):
    h = 5
    fwd = forward_return(synthetic_prices, h)
    close = synthetic_prices["Close"]
    assert fwd.iloc[10] == pytest.approx(close.iloc[15] / close.iloc[10] - 1)
    assert fwd.iloc[-h:].isna().all() and direction_label(fwd).iloc[-h:].isna().all()
    with pytest.raises(ValueError):
        forward_return(synthetic_prices, 0)


# ---------------------------------------------------------------- model
def test_logistic_learns_a_separable_signal_and_round_trips():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3))
    y = (x[:, 0] + 0.1 * rng.normal(size=400) > 0).astype(float)
    model = LogisticModel().fit(x, y)
    assert np.mean((model.predict_proba(x) > 0.5) == (y > 0.5)) > 0.9
    assert abs(model.coef_[0]) > abs(model.coef_[1]) * 3
    clone = LogisticModel.from_dict(model.to_dict())
    np.testing.assert_allclose(clone.predict_proba(x), model.predict_proba(x))


def test_logistic_rejects_bad_input():
    with pytest.raises(ValueError):
        LogisticModel().fit(np.array([[np.nan, 1.0]] * 20), np.zeros(20))
    with pytest.raises(ValueError):
        LogisticModel().fit(np.zeros((5, 2)), np.zeros(5))
    with pytest.raises(RuntimeError):
        LogisticModel().predict_proba(np.zeros((1, 2)))


# ---------------------------------------------------------------- experts & ensemble
def test_rule_experts_are_bounded_and_directional(synthetic_prices):
    probs = rule_expert_probs(build_features(synthetic_prices))
    assert probs.min().min() >= 0.05 and probs.max().max() <= 0.95
    assert (probs.max() - 0.5).max() <= 0.1 + 1e-9  # modest tilt only


def test_base_rate_uses_only_labels_known_at_the_time(synthetic_prices):
    h = 5
    y = direction_label(forward_return(synthetic_prices, h))
    original = base_rate_series(y, h)
    tampered = y.copy()
    tampered.iloc[600:] = 1 - tampered.iloc[600:]  # rewrite the future
    changed = base_rate_series(tampered, h)
    pd.testing.assert_series_equal(original.iloc[: 600 + h], changed.iloc[: 600 + h])


def test_ensemble_learns_to_trust_the_accurate_expert():
    ens = Ensemble()
    rng = np.random.default_rng(1)
    for _ in range(200):
        outcome = rng.random() > 0.5
        good = 0.8 if outcome else 0.2
        probs = {n: 0.5 for n in EXPERT_NAMES} | {"trend": good, "momentum": 1 - good}
        ens.update(probs, outcome)
    assert ens.weights["trend"] > 0.8
    assert ens.weights["momentum"] < 0.05
    assert sum(ens.weights.values()) == pytest.approx(1.0)
    assert min(ens.weights.values()) >= ens.alpha / len(EXPERT_NAMES) - 1e-12  # fixed-share floor


def test_ensemble_handles_missing_experts_and_bad_weights():
    ens = Ensemble()
    assert ens.predict({}) == 0.5
    assert ens.predict({"trend": math.nan, "momentum": 0.7}) == pytest.approx(0.7)
    ens.update({"trend": math.nan}, True)  # NaN counts as neutral, must not crash
    assert all(math.isfinite(w) for w in ens.weights.values())
    with pytest.raises(ValueError):
        Ensemble(weights={"a": 0.0, "b": 0.0})
    with pytest.raises(ValueError):
        Ensemble(weights={"a": -1.0, "b": 2.0})


def test_evaluate_metrics_and_verdicts():
    n = 400
    rng = np.random.default_rng(2)
    y = (rng.random(n) > 0.5).astype(float)
    ret = np.where(y > 0.5, 0.01, -0.01)
    base = np.full(n, 0.5)
    perfect = evaluate(np.where(y > 0.5, 0.9, 0.1), y, ret, base, horizon=1)
    assert perfect.accuracy == 1.0 and perfect.brier_skill > 0.9
    assert perfect.avg_ret_when_up > 0 > perfect.avg_ret_when_down
    assert "possible edge" in perfect.verdict
    coin = evaluate(rng.random(n), y, ret, base, horizon=1)
    assert "no demonstrable edge" in coin.verdict
    assert "too few" in evaluate(np.full(20, 0.6), y[:20], ret[:20], base[:20], 5).verdict
    assert evaluate(np.array([]), np.array([]), np.array([]), np.array([]), 5).n == 0


# ---------------------------------------------------------------- replay
def test_replay_has_no_lookahead(synthetic_prices):
    """Replaying truncated history must reproduce the same forecasts on the shared days."""
    full = replay(synthetic_prices, 5)
    cut = 1000
    partial = replay(synthetic_prices.iloc[:cut], 5)
    shared = partial.probs.index
    pd.testing.assert_frame_equal(partial.probs, full.probs.loc[shared], check_exact=False,
                                  rtol=1e-9, atol=1e-12)


def test_replay_output_is_consistent(synthetic_prices):
    r = replay(synthetic_prices, 5)
    assert set(r.metrics) == {*EXPERT_NAMES, "ensemble"}
    assert r.probs.notna().all().all()
    assert sum(r.final_weights.values()) == pytest.approx(1.0)
    assert r.weights_history.shape == (len(r.probs), len(EXPERT_NAMES))
    with pytest.raises(ValueError, match="at least"):
        replay(synthetic_prices.iloc[:100], 5)


def test_replay_finds_signal_when_there_is_one_and_not_when_there_is_none():
    strong = SyntheticProvider(seed=11, n_days=1800, regime_sd=0.006).fetch("S")
    none = SyntheticProvider(seed=11, n_days=1800, regime_sd=0.0).fetch("S")
    assert replay(strong, 5).metrics["momentum"].accuracy > 0.55
    assert replay(none, 5).metrics["momentum"].brier_skill < 0.01  # nothing real to find
