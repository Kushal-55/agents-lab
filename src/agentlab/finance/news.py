"""News headline fetching with defensive parsing.

Headlines come from the open internet, so they are untrusted: a headline could contain text
written to instruct the model (prompt injection). Control characters are stripped, lengths are
capped, and the tool that returns them labels the block as quoted data.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agentlab.finance.data import DataError, validate_ticker

logger = logging.getLogger(__name__)

MAX_HEADLINES = 8
MAX_HEADLINE_CHARS = 160
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _clean(text: Any) -> str:
    return _CONTROL.sub(" ", str(text)).strip()[:MAX_HEADLINE_CHARS]


def parse_news_items(items: list[dict[str, Any]] | None) -> list[tuple[str, str]]:
    """Return ``(title, source)`` pairs from yfinance news in either of its known layouts."""
    out: list[tuple[str, str]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        # Newer yfinance versions nest the fields under "content"; older ones are flat.
        content = item.get("content") if isinstance(item.get("content"), dict) else item
        title = content.get("title")
        if not title:
            continue
        provider = content.get("provider")
        source = (provider.get("displayName") if isinstance(provider, dict) else None) \
            or content.get("publisher") or "unknown"
        out.append((_clean(title), _clean(source)))
        if len(out) >= MAX_HEADLINES:
            break
    return out


def fetch_headlines(ticker: str) -> list[tuple[str, str]]:
    import yfinance as yf

    ticker = validate_ticker(ticker)
    try:
        headlines = parse_news_items(yf.Ticker(ticker).news)
        logger.info("Fetched %d headline(s) for %s", len(headlines), ticker)
        return headlines
    except Exception as exc:  # noqa: BLE001 - yfinance raises many types
        logger.warning("News fetch failed for %s: %s", ticker, type(exc).__name__)
        raise DataError(f"Could not fetch news for {ticker}.") from exc
