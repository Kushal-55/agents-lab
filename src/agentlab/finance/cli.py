"""Command line interface for the self-learning market agent (Level 9).

Offline / key-free commands:  forecast, backtest, resolve, report, lessons, demo
LLM-powered command:          chat   (needs a free LLM, see .env.example)
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from agentlab.config import ConfigError, Settings, load_settings
from agentlab.finance import DISCLAIMER
from agentlab.finance.analysis import sparkline
from agentlab.finance.assembly import build_analyst_agent, make_journal, make_provider
from agentlab.finance.data import DataError, SyntheticProvider, TimeMachineProvider
from agentlab.finance.experts import EXPERT_NAMES
from agentlab.finance.journal import Journal
from agentlab.finance.learner import Forecaster
from agentlab.finance.report import format_backtest, format_forecast, format_performance
from agentlab.llm import LLMClient, LLMError
from agentlab.log import setup_logging
from agentlab.repl import print_tool_call, run_repl

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-agent",
        description="Self-learning market forecasting agent. " + DISCLAIMER,
    )
    parser.add_argument("--synthetic", action="store_true",
                        help="use seeded FAKE prices (offline; separate journal)")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING (default INFO)")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, helptext in (("forecast", "forecast direction and record it in the journal"),
                           ("backtest", "walk-forward backtest of every expert")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("ticker")
        p.add_argument("--horizon", type=int, default=5, help="trading days ahead (1-60)")
    sub.add_parser("resolve", help="score matured forecasts and let the agent learn")
    for name in ("report", "lessons"):
        p = sub.add_parser(name, help=f"show the {name}")
        p.add_argument("ticker", nargs="?")
    demo = sub.add_parser("demo", help="offline demo of the full learning loop on fake data")
    demo.add_argument("--steps", type=int, default=120)
    sub.add_parser("chat", help="talk to the LLM analyst agent")
    return parser


def run_demo(settings: Settings, steps: int = 120, horizon: int = 5) -> str:
    """Predict -> wait -> score -> learn, daily, on seeded fake prices with a hidden trend.

    The agent starts with NO prior knowledge (equal expert weights) so you can watch it
    learn from experience alone.
    """
    steps = max(1, min(steps, 250))
    logger.info("Running demo: %d steps, horizon %d", steps, horizon)
    # The time machine only reveals the first `visible` rows, and advance() reveals one more
    # each loop, which simulates a new trading day arriving without waiting for real time.
    provider = TimeMachineProvider(SyntheticProvider(regime_sd=0.004), visible=900)
    with tempfile.TemporaryDirectory(prefix="agentlab-demo-") as tmp:
        journal = Journal(Path(tmp) / "demo.sqlite3")
        forecaster = Forecaster(provider, journal)
        start_weights = {n: 1 / len(EXPERT_NAMES) for n in EXPERT_NAMES}
        journal.set_weights("DEMO", horizon, start_weights)  # cold start: skip warm-start
        for _ in range(steps):
            forecaster.forecast("DEMO", horizon, with_evidence=False)
            provider.advance(1)
            forecaster.resolve()
        end_weights = journal.get_weights("DEMO", horizon) or {}
        out = [
            f"DEMO: {steps} daily cycles of forecast -> wait {horizon} days -> score -> learn,",
            "on FAKE prices containing a hidden slow trend. Starting weights are equal.",
            "",
            f"{'expert':16s}{'start weight':>14}{'end weight':>12}",
        ]
        for name, w0 in start_weights.items():
            out.append(f"{name:16s}{w0:>14.0%}{end_weights.get(name, 0):>12.0%}")
        out += ["", format_performance(journal, "DEMO"), "", *journal.lessons("DEMO", limit=2)]
        out += ["", "Price path shown to the agent: " + sparkline(
            provider.fetch("DEMO").Close.to_numpy()), DISCLAIMER]
    return "\n".join(out)


def _chat(settings: Settings, synthetic: bool) -> int:
    try:
        settings = load_settings()  # now the LLM key is required
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    provider = make_provider(settings, synthetic)
    agent = build_analyst_agent(
        LLMClient(settings), settings, provider, journal=make_journal(settings, synthetic),
        on_tool_call=print_tool_call,
    )
    banner = (f"Self-learning market analyst ({settings.provider}/{settings.model}).\n"
              f"{DISCLAIMER}")
    return run_repl(agent, banner)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(require_key=False)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    setup_logging(args.log_level or settings.log_level, settings.data_dir / "agentlab.log")
    logger.info("Command=%s synthetic=%s", args.command, args.synthetic)

    try:
        if args.command == "chat":
            return _chat(settings, args.synthetic)
        if args.command == "demo":
            print(run_demo(settings, args.steps))
            return 0
        provider = make_provider(settings, args.synthetic)
        journal = make_journal(settings, args.synthetic)
        forecaster = Forecaster(provider, journal)
        if args.synthetic:
            print("[synthetic mode: prices are FAKE, journal is separate]\n")
        if args.command == "forecast":
            print(format_forecast(forecaster.forecast(args.ticker, args.horizon)))
        elif args.command == "backtest":
            print(format_backtest(args.ticker.upper(),
                                  forecaster.backtest(args.ticker, args.horizon)))
        elif args.command == "resolve":
            summary = forecaster.resolve()
            print(f"Resolved {summary.resolved}; still waiting {summary.still_pending}; "
                  f"skipped {summary.skipped}.")
            for lesson in summary.lessons:
                print(f" - {lesson}")
        elif args.command == "report":
            print(format_performance(journal, args.ticker.upper() if args.ticker else None))
        elif args.command == "lessons":
            rows = journal.lessons(args.ticker.upper() if args.ticker else None, limit=10)
            print("\n".join(rows) if rows else "No lessons yet.")
        return 0
    except (ValueError, DataError, LLMError) as exc:
        # These are expected, user-fixable problems (bad ticker, no data, LLM unreachable).
        # Show a short message; the traceback is only kept at DEBUG level.
        logger.warning("Command %s failed: %s", args.command, exc)
        logger.debug("Traceback for failed command", exc_info=True)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
