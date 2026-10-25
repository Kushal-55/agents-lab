"""Turn price frames into short text summaries (used by the Level 8 tools).

These are pure functions: DataFrame in, string out, no network and no state, which keeps them
easy to test.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from agentlab.finance.features import build_features


def summarize_prices(ticker: str, prices: pd.DataFrame) -> str:
    close = prices["Close"]
    last = float(close.iloc[-1])

    def change(days: int) -> str:
        if len(close) <= days:
            return "n/a"
        return f"{last / float(close.iloc[-1 - days]) - 1:+.1%}"

    window = close.iloc[-252:]  # roughly one trading year
    daily = close.pct_change().dropna()
    # Annualise daily volatility with sqrt(252 trading days).
    vol = float(daily.iloc[-20:].std() * math.sqrt(252)) if len(daily) >= 20 else math.nan
    high, low = float(window.max()), float(window.min())
    return (
        f"{ticker} close ${last:,.2f} on {prices.index[-1].date()}\n"
        f"Change: 1d {change(1)}, 5d {change(5)}, 1m {change(21)}, 3m {change(63)}, "
        f"1y {change(252)}\n"
        f"52-week range ${low:,.2f} - ${high:,.2f} (now {last / high - 1:+.1%} from the high)\n"
        f"Realised volatility (20d, annualised): {vol:.1%}\n"
        f"Average volume (20d): {float(prices['Volume'].iloc[-20:].mean()):,.0f}"
    )


def technical_snapshot(ticker: str, prices: pd.DataFrame) -> str:
    feats = build_features(prices).iloc[-1]
    if feats.isna().any():
        return f"{ticker}: not enough history for a full technical read."
    rsi = feats["rsi"]
    rsi_note = "overbought (>70)" if rsi > 70 else "oversold (<30)" if rsi < 30 else "neutral"
    macd_note = "positive (bullish momentum)" if feats["macd_hist"] > 0 else \
        "negative (bearish momentum)"
    trend_note = "above" if feats["trend_gap"] > 0 else "below"
    cross_note = "10-day average above 50-day (uptrend)" if feats["sma_cross"] > 0 else \
        "10-day average below 50-day (downtrend)"
    bb = feats["bb_pos"]
    bb_note = "near upper band" if bb > 0.8 else "near lower band" if bb < -0.8 else "mid-band"
    return (
        f"{ticker} technical snapshot ({prices.index[-1].date()}):\n"
        f"- RSI(14) {rsi:.0f}: {rsi_note}\n"
        f"- MACD histogram {macd_note}\n"
        f"- Price is {abs(feats['trend_gap']):.1%} {trend_note} its 50-day average; {cross_note}\n"
        f"- Bollinger position {bb:+.2f}: {bb_note}\n"
        f"- 20-day volatility {feats['vol_20']:.2%} per day; volume vs 20d avg "
        f"{feats['volume_ratio']:+.0%}\n"
        "Indicators describe the past; they are weak predictors on their own."
    )


def sparkline(values: np.ndarray, width: int = 40) -> str:
    """Tiny text chart for terminals."""
    if len(values) == 0:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    idx = np.linspace(0, len(values) - 1, min(width, len(values))).astype(int)
    v = np.asarray(values, float)[idx]
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return blocks[3] * len(v)
    return "".join(blocks[int((x - lo) / (hi - lo) * (len(blocks) - 1))] for x in v)
