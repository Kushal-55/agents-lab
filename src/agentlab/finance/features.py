"""Technical indicators and the feature matrix used by the experts.

The one rule that matters here: the feature row for day *t* may only use data up to and
including day *t*. Labels (what we try to predict) look forward and are built separately.
``test_features_have_no_lookahead`` in tests/test_finance_engine.py checks this by recomputing
the features on truncated data and comparing.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_20",
    "trend_gap",
    "sma_cross",
    "rsi",
    "macd_hist",
    "vol_20",
    "bb_pos",
    "volume_ratio",
]


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index in [0, 100]."""
    delta = close.diff()
    # Wilder smoothing is an EMA with alpha = 1/window (not the usual 2/(window+1)).
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # No losses at all in the window => fully overbought.
    return out.where(loss != 0, 100.0).where(gain.notna() & loss.notna())


def macd_histogram(close: pd.Series) -> pd.Series:
    macd_line = ema(close, 12) - ema(close, 26)
    signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    return macd_line - signal


def bollinger_position(close: pd.Series, window: int = 20) -> pd.Series:
    """Where price sits in the Bollinger band: -1 = lower band, 0 = middle, +1 = upper band."""
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std()
    return (close - mid) / (2 * std.replace(0, np.nan))


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute the feature matrix (one row per date; early rows are NaN while warming up)."""
    close, volume = prices["Close"], prices["Volume"]
    daily = close.pct_change()
    # Every column below is a trailing (backward-looking) calculation. Prices and volume are
    # divided by their own averages so the features are scale-free across tickers.
    feats = pd.DataFrame(index=prices.index)
    feats["ret_1"] = daily
    feats["ret_5"] = close.pct_change(5)
    feats["ret_20"] = close.pct_change(20)
    feats["trend_gap"] = close / sma(close, 50) - 1
    feats["sma_cross"] = sma(close, 10) / sma(close, 50) - 1
    feats["rsi"] = rsi(close)
    feats["macd_hist"] = macd_histogram(close) / close
    feats["vol_20"] = daily.rolling(20, min_periods=20).std()
    feats["bb_pos"] = bollinger_position(close)
    feats["volume_ratio"] = volume / volume.rolling(20, min_periods=20).mean() - 1
    feats = feats[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    complete = int(feats.notna().all(axis=1).sum())  # rows past the indicator warm-up period
    logger.debug("Built %d feature rows (%d complete)", len(feats), complete)
    return feats


def forward_return(prices: pd.DataFrame, horizon: int) -> pd.Series:
    """Return from day t's close to the close ``horizon`` trading days later (NaN at the end)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    close = prices["Close"]
    return close.shift(-horizon) / close - 1


def direction_label(fwd_ret: pd.Series) -> pd.Series:
    """1.0 if the forward return is positive, 0.0 otherwise, NaN where unknown."""
    return (fwd_ret > 0).astype(float).where(fwd_ret.notna())
