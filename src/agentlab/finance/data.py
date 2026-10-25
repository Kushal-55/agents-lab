"""Market data providers (all free, no API key).

* ``YahooProvider``    - real daily prices via the ``yfinance`` package.
* ``CachedProvider``   - wraps any provider with an on-disk CSV cache (be polite to free APIs).
* ``SyntheticProvider``- seeded fake prices for tests and offline demos. NOT real market data.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
_TICKER_RE = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,11}$")
_VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"}


class DataError(RuntimeError):
    """Market data could not be obtained or failed validation."""


def validate_ticker(raw: str) -> str:
    """Normalise and validate a ticker symbol (blocks path/SQL/prompt injection via the symbol)."""
    ticker = (raw or "").strip().upper()
    if not _TICKER_RE.match(ticker):
        raise ValueError(f"Invalid ticker symbol: {raw!r}")
    return ticker


def validate_period(period: str) -> str:
    if period not in _VALID_PERIODS:
        raise ValueError(f"period must be one of {sorted(_VALID_PERIODS)}")
    return period


class PriceProvider(Protocol):
    def fetch(self, ticker: str, period: str = "5y") -> pd.DataFrame: ...


def clean_prices(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate and normalise an OHLCV frame: tz-naive sorted unique dates, no NaN/<=0 prices."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{ticker}: data is missing columns {missing}")
    out = df.loc[:, list(REQUIRED_COLUMNS)].astype(float)
    # Make every date timezone-naive midnight so frames from different sources line up and
    # comparisons with plain dates (e.g. journal as_of values) work.
    index = pd.to_datetime(out.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    out.index = index.normalize()
    out.index.name = "Date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna()
    # Zero or negative prices are data errors; they would break returns and logs downstream.
    out = out[(out[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    dropped = len(df) - len(out)
    if dropped:
        logger.debug("%s: dropped %d invalid/duplicate row(s) while cleaning", ticker, dropped)
    if len(out) < 30:
        raise DataError(f"{ticker}: only {len(out)} valid rows returned (need at least 30)")
    return out


class YahooProvider:
    """Daily OHLCV from Yahoo Finance (split/dividend adjusted)."""

    def __init__(self, retries: int = 1, backoff_s: float = 1.0) -> None:
        self._retries = retries
        self._backoff_s = backoff_s

    def fetch(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        import yfinance as yf  # imported lazily so tests/offline use never need it

        ticker, period = validate_ticker(ticker), validate_period(period)
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                raw = yf.Ticker(ticker).history(period=period, auto_adjust=True)
                if raw is not None and not raw.empty:
                    logger.info("Fetched %d rows for %s from Yahoo Finance", len(raw), ticker)
                    return clean_prices(raw, ticker)
                last_error = DataError(f"{ticker}: no data returned (invalid symbol?)")
            except DataError:
                raise
            except Exception as exc:  # noqa: BLE001 - yfinance raises many different types
                last_error = exc
                logger.warning("Yahoo fetch attempt %d failed for %s: %s", attempt + 1, ticker,
                               type(exc).__name__)
            if attempt < self._retries:
                time.sleep(self._backoff_s * (2**attempt))
        raise DataError(f"Could not fetch data for {ticker}: {last_error}") from last_error


class CachedProvider:
    """Cache another provider's results as CSV files; fall back to stale cache if offline."""

    def __init__(self, inner: PriceProvider, cache_dir: Path, ttl_hours: float = 6.0) -> None:
        self._inner = inner
        self._dir = cache_dir
        self._ttl_s = ttl_hours * 3600

    def fetch(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        ticker, period = validate_ticker(ticker), validate_period(period)
        # The validated ticker is the only user-influenced part of the filename.
        safe_name = re.sub(r"[^A-Z0-9]", "_", ticker)
        path = self._dir / f"{safe_name}_{period}.csv"
        fresh = path.exists() and (time.time() - path.stat().st_mtime) < self._ttl_s
        if fresh:
            logger.debug("Cache hit for %s", ticker)
            return self._read(path, ticker)
        logger.info("Cache miss for %s/%s, fetching from the source", ticker, period)
        try:
            df = self._inner.fetch(ticker, period)
        except DataError:
            if path.exists():
                logger.warning("Using STALE cached data for %s (live fetch failed)", ticker)
                return self._read(path, ticker)
            raise
        self._dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        return df

    @staticmethod
    def _read(path: Path, ticker: str) -> pd.DataFrame:
        return clean_prices(pd.read_csv(path, index_col=0, parse_dates=True), ticker)


class SyntheticProvider:
    """Seeded fake prices for tests/demos. NOT real market data.

    Daily returns = slowly-changing latent drift (a persistent 'regime', so trend-following can
    genuinely work when ``regime_sd > 0``) + random noise. Set ``regime_sd=0`` for a pure
    random walk where no strategy has an edge.
    """

    def __init__(self, seed: int = 7, n_days: int = 1500, regime_sd: float = 0.0012,
                 regime_phi: float = 0.97, drift: float = 0.0003, vol: float = 0.012) -> None:
        self._seed, self._n = seed, n_days
        self._regime_sd, self._phi = regime_sd, regime_phi
        self._drift, self._vol = drift, vol

    def fetch(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        ticker = validate_ticker(ticker)
        # Different tickers get different (but reproducible) paths.
        rng = np.random.default_rng(self._seed + sum(ord(c) for c in ticker))
        shocks = rng.normal(0, self._vol, self._n)
        innovations = rng.normal(0, self._regime_sd * np.sqrt(1 - self._phi**2), self._n)
        regime = np.zeros(self._n)
        for i in range(1, self._n):
            regime[i] = self._phi * regime[i - 1] + innovations[i]
        close = 100 * np.exp(np.cumsum(self._drift + regime + shocks))
        spread = np.abs(rng.normal(0, self._vol / 2, self._n))
        dates = pd.bdate_range(end="2024-12-31", periods=self._n)
        frame = pd.DataFrame(
            {
                "Open": np.r_[close[0], close[:-1]],
                "High": close * (1 + spread),
                "Low": close * (1 - spread),
                "Close": close,
                "Volume": rng.integers(1_000_000, 5_000_000, self._n).astype(float),
            },
            index=dates,
        )
        return clean_prices(frame, ticker)


class TimeMachineProvider:
    """Reveals only the first ``visible`` rows of another provider's data.

    ``advance()`` simulates trading days passing, so the full predict -> wait -> score -> learn
    loop can be demonstrated and tested in seconds instead of waiting weeks.
    """

    def __init__(self, inner: PriceProvider, visible: int) -> None:
        if visible < 1:
            raise ValueError("visible must be >= 1")
        self._inner, self.visible = inner, visible

    def advance(self, days: int = 1) -> None:
        self.visible += days

    def fetch(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        return self._inner.fetch(ticker, period).iloc[: self.visible]
