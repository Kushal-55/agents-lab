"""Plain-text formatting for forecasts, backtests and the live track record.

Used by both the CLI and the LLM tools, so the numbers the user sees on the command line are
the same ones the chat agent quotes. This module only formats; it does no calculation.
"""

from __future__ import annotations

import math

import numpy as np

from agentlab.finance import DISCLAIMER
from agentlab.finance.backtest import ReplayResult
from agentlab.finance.evaluation import Metrics
from agentlab.finance.experts import EXPERT_NAMES
from agentlab.finance.journal import Journal
from agentlab.finance.learner import Forecast


def _pct(x: float, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def format_metrics(m: Metrics) -> str:
    if m.n == 0:
        return "no resolved samples yet"
    return (
        f"accuracy {_pct(m.accuracy)} vs {_pct(m.majority_accuracy)} for always guessing the "
        f"common outcome; Brier skill {m.brier_skill:+.3f} (>0 beats the base rate); "
        f"z={m.z_score:+.2f}; n={m.n}. Verdict: {m.verdict}."
    )


def format_forecast(f: Forecast) -> str:
    lines = [
        f"{f.ticker}: {f.horizon}-trading-day direction forecast "
        f"(data as of {f.as_of}, close ${f.entry_price:,.2f})",
        f"P(higher in {f.horizon} days) = {_pct(f.p_up)}  ->  {f.stance}",
        "Expert views [P(up), learned weight]:",
    ]
    for name in EXPERT_NAMES:
        if name in f.expert_probs:
            lines.append(f"  - {name:15s} {f.expert_probs[name]:.2f}  weight {f.weights[name]:.0%}")
    if f.drivers:
        unusual = ", ".join(f"{k} {v:+.1f} sigma" for k, v in f.drivers)
        lines.append(f"Most unusual readings today vs history: {unusual}")
    if f.evidence is not None:
        lines.append(f"Walk-forward track record on this ticker: {format_metrics(f.evidence)}")
        if f.evidence.verdict.startswith(("no demonstrable", "too few")):
            base = f.expert_probs.get("base_rate")
            note = (f" (about the historical share of up-moves, {_pct(base)})"
                    if base is not None else "")
            lines.append("Note: the track record shows no reliable edge, so treat the "
                         f"probability above as roughly the base rate{note}, not as a signal.")
    if f.warm_started:
        lines.append("(First forecast for this ticker/horizon: weights were warm-started "
                     "from historical replay.)")
    if f.journal_id is not None:
        state = "recorded" if f.newly_recorded else "already recorded for this date"
        lines.append(f"Journal entry #{f.journal_id} {state}; it will be scored once "
                     f"{f.horizon} more trading days of prices exist.")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def format_backtest(ticker: str, result: ReplayResult) -> str:
    start, end = result.probs.index[0].date(), result.probs.index[-1].date()
    lines = [
        f"{ticker}: walk-forward replay, {result.horizon}-day horizon, {result.n_days} days "
        f"({start} to {end}). Every forecast used only data available that day.",
        f"{'expert':16s}{'accuracy':>9}{'naive':>8}{'skill':>8}{'z':>7}   verdict",
    ]
    for name in (*EXPERT_NAMES, "ensemble"):
        m = result.metrics[name]
        lines.append(f"{name:16s}{_pct(m.accuracy):>9}{_pct(m.majority_accuracy):>8}"
                     f"{m.brier_skill:>+8.3f}{m.z_score:>+7.2f}   {m.verdict}")
    weights = ", ".join(f"{k} {v:.0%}" for k, v in result.final_weights.items())
    lines.append(f"Final learned weights: {weights}")
    lines.append("skill = 1 - Brier/Brier(base rate); naive = always guess the common outcome.")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def format_performance(journal: Journal, ticker: str | None = None) -> str:
    done = journal.predictions(resolved=True, ticker=ticker)
    pending = journal.predictions(resolved=False, ticker=ticker)
    scope = ticker or "all tickers"
    lines = [f"Live track record ({scope}): {len(done)} scored, {len(pending)} pending."]
    if not done:
        lines.append("Nothing scored yet: forecasts are scored only after their horizon "
                     "has really elapsed. Run `resolve` later.")
        return "\n".join(lines)

    y = np.array([float(r.outcome_up) for r in done])
    p = np.array([r.p_up for r in done])
    acc = float(np.mean((p > 0.5) == (y > 0.5)))
    brier = float(np.mean((p - y) ** 2))
    lines.append(f"Ensemble direction accuracy {_pct(acc)}; Brier {brier:.3f} "
                 f"(a constant 50/50 forecast scores 0.250).")
    for name in EXPERT_NAMES:
        hits = [(r.expert_probs[name] > 0.5) == r.outcome_up for r in done
                if name in r.expert_probs]
        if hits:
            lines.append(f"  - {name:15s} accuracy {_pct(float(np.mean(hits)))} (n={len(hits)})")
    if len(done) < 30:
        lines.append("Fewer than 30 scored forecasts: treat these percentages as noise.")
    return "\n".join(lines)
