"""LEVEL 8 - A finance analyst agent with live market data.

Concept:   Everything so far, plus real (free) data: prices, technical indicators and news
           from Yahoo Finance via yfinance, cached on disk, plus the RAG notes and calculators.
Concepts to learn: wrapping a data API as tools, caching + stale-cache fallback, ticker validation,
           and treating web text (news) as UNTRUSTED input (prompt-injection defence).
Limit:     It reports facts; it doesn't forecast or learn from its mistakes -> Level 9.

Run:   python levels/08_finance_analyst.py            (add --synthetic for offline fake data)
Try:   "Give me a snapshot of MSFT and tell me what the RSI means."
"""

from __future__ import annotations

import argparse
import logging

from agentlab.bootstrap import start
from agentlab.finance.assembly import build_analyst_agent, make_provider
from agentlab.repl import print_tool_call, run_repl

logger = logging.getLogger("level08")


def main() -> int:
    parser = argparse.ArgumentParser(description="Finance analyst agent (Level 8)")
    parser.add_argument("--synthetic", action="store_true", help="use fake offline prices")
    args = parser.parse_args()

    settings, llm = start()
    logger.info("Starting level 8 (synthetic prices: %s)", args.synthetic)
    agent = build_analyst_agent(llm, settings, make_provider(settings, args.synthetic),
                                on_tool_call=print_tool_call)
    return run_repl(agent, "Finance analyst: prices, technicals, news, knowledge base. "
                           "Educational only, not financial advice.")


if __name__ == "__main__":
    raise SystemExit(main())
