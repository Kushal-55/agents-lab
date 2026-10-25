"""Prediction journal backed by SQLite.

A forecast is stored before its outcome is known and scored later against the real price, so
the track record can't be edited after the fact. The journal holds three tables:

* ``predictions``     one row per (ticker, horizon, as_of date), later filled in with the outcome
* ``expert_weights``  the current learned ensemble weights per (ticker, horizon)
* ``lessons``         short text summaries written every few scored forecasts

All SQL uses bound parameters; the only f-string is built from fixed fragments.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agentlab.db import connect

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    as_of TEXT NOT NULL,
    created_at TEXT NOT NULL,
    p_up REAL NOT NULL,
    expert_probs TEXT NOT NULL,
    weights TEXT NOT NULL,
    entry_price REAL NOT NULL,
    outcome_up INTEGER,
    exit_price REAL,
    resolved_at TEXT,
    UNIQUE (ticker, horizon, as_of)
);
CREATE TABLE IF NOT EXISTS expert_weights (
    ticker TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    weights TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (ticker, horizon)
);
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    text TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class PredictionRecord:
    id: int
    ticker: str
    horizon: int
    as_of: str
    created_at: str
    p_up: float
    expert_probs: dict[str, float]
    weights: dict[str, float]
    entry_price: float
    outcome_up: bool | None = None
    exit_price: float | None = None
    resolved_at: str | None = None

    @property
    def resolved(self) -> bool:
        return self.outcome_up is not None


class Journal:
    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        db_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)  # owner-only data dir
        with connect(self._path) as conn:
            conn.executescript(_SCHEMA)

    # ---- weights -------------------------------------------------------------------------
    def get_weights(self, ticker: str, horizon: int) -> dict[str, float] | None:
        with connect(self._path) as conn:
            row = conn.execute(
                "SELECT weights FROM expert_weights WHERE ticker = ? AND horizon = ?",
                (ticker, horizon),
            ).fetchone()
        return json.loads(row["weights"]) if row else None

    def set_weights(self, ticker: str, horizon: int, weights: dict[str, float]) -> None:
        with connect(self._path) as conn:
            self._upsert_weights(conn, ticker, horizon, weights)
        logger.debug("Stored weights for %s/%dd: %s", ticker, horizon, weights)

    @staticmethod
    def _upsert_weights(conn, ticker: str, horizon: int, weights: dict[str, float]) -> None:  # noqa: ANN001
        conn.execute(
            "INSERT INTO expert_weights (ticker, horizon, weights, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(ticker, horizon) DO UPDATE SET "
            "weights = excluded.weights, updated_at = excluded.updated_at",
            (ticker, horizon, json.dumps(weights), _now()),
        )

    # ---- predictions ---------------------------------------------------------------------
    def record_prediction(
        self,
        ticker: str,
        horizon: int,
        as_of: str,
        p_up: float,
        expert_probs: dict[str, float],
        weights: dict[str, float],
        entry_price: float,
    ) -> tuple[int, bool]:
        """Insert a forecast. Returns ``(id, created)``; a repeat for the same day is a no-op."""
        with connect(self._path) as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO predictions (ticker, horizon, as_of, created_at, p_up, "
                "expert_probs, weights, entry_price) VALUES (?,?,?,?,?,?,?,?)",
                (ticker, horizon, as_of, _now(), p_up, json.dumps(expert_probs),
                 json.dumps(weights), entry_price),
            )
            # INSERT OR IGNORE hits the UNIQUE (ticker, horizon, as_of) constraint when the same
            # forecast is requested twice on one day; rowcount tells us whether we inserted.
            created = cur.rowcount == 1
            row = conn.execute(
                "SELECT id FROM predictions WHERE ticker = ? AND horizon = ? AND as_of = ?",
                (ticker, horizon, as_of),
            ).fetchone()
        if created:
            logger.info("Journal: recorded #%d %s/%dd as_of=%s p_up=%.3f", row["id"], ticker,
                        horizon, as_of, p_up)
        else:
            logger.info("Journal: %s/%dd for %s already recorded (#%d), not duplicating", ticker,
                        horizon, as_of, row["id"])
        return int(row["id"]), created

    def apply_resolution(
        self,
        prediction_id: int,
        outcome_up: bool,
        exit_price: float,
        ticker: str,
        horizon: int,
        new_weights: dict[str, float],
    ) -> None:
        """Mark a prediction resolved and save the updated weights in one transaction.

        Both writes commit together or not at all, so a crash can never leave a forecast
        scored without the weights having learned from it (or the reverse).
        """
        with connect(self._path) as conn:
            # "outcome_up IS NULL" makes this idempotent: a forecast can only be learned from once.
            cur = conn.execute(
                "UPDATE predictions SET outcome_up = ?, exit_price = ?, resolved_at = ? "
                "WHERE id = ? AND outcome_up IS NULL",
                (int(outcome_up), exit_price, _now(), prediction_id),
            )
            if cur.rowcount != 1:
                logger.warning("Prediction %d was already resolved; skipping", prediction_id)
                return
            self._upsert_weights(conn, ticker, horizon, new_weights)
        logger.info("Journal: resolved #%d %s/%dd outcome=%s exit_price=%.2f", prediction_id,
                    ticker, horizon, "UP" if outcome_up else "NOT UP", exit_price)

    def predictions(
        self, *, resolved: bool | None = None, ticker: str | None = None
    ) -> list[PredictionRecord]:
        clauses, params = [], []
        if resolved is True:
            clauses.append("outcome_up IS NOT NULL")
        elif resolved is False:
            clauses.append("outcome_up IS NULL")
        if ticker:
            clauses.append("ticker = ?")
            params.append(ticker)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""  # fixed fragments only
        with connect(self._path) as conn:
            rows = conn.execute(
                f"SELECT * FROM predictions {where} ORDER BY as_of, id",  # noqa: S608
                params,
            ).fetchall()
        return [
            PredictionRecord(
                id=r["id"], ticker=r["ticker"], horizon=r["horizon"], as_of=r["as_of"],
                created_at=r["created_at"], p_up=r["p_up"],
                expert_probs=json.loads(r["expert_probs"]), weights=json.loads(r["weights"]),
                entry_price=r["entry_price"],
                outcome_up=None if r["outcome_up"] is None else bool(r["outcome_up"]),
                exit_price=r["exit_price"], resolved_at=r["resolved_at"],
            )
            for r in rows
        ]

    # ---- lessons -------------------------------------------------------------------------
    def add_lesson(self, ticker: str, horizon: int, text: str) -> None:
        with connect(self._path) as conn:
            conn.execute(
                "INSERT INTO lessons (ticker, horizon, created_at, text) VALUES (?,?,?,?)",
                (ticker, horizon, _now(), text),
            )
        logger.info("Journal: lesson added for %s/%dd", ticker, horizon)

    def lessons(self, ticker: str | None = None, limit: int = 5) -> list[str]:
        query = "SELECT created_at, ticker, horizon, text FROM lessons"
        params: list = []
        if ticker:
            query += " WHERE ticker = ?"
            params.append(ticker)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with connect(self._path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            f"[{r['created_at'][:10]}] {r['ticker']} {r['horizon']}d: {r['text']}" for r in rows
        ]
