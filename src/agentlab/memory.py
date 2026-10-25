"""Short-term (conversation window) and long-term (persistent facts) memory."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentlab.db import connect
from agentlab.log import contains_secret
from agentlab.retrieval import Bm25Index

logger = logging.getLogger(__name__)

MAX_FACT_CHARS = 500
MAX_FACTS = 1000


class ConversationMemory:
    """Rolling window over conversation *turns*.

    A turn starts at a user message and includes the assistant/tool messages that follow.
    Trimming whole turns keeps every assistant ``tool_calls`` message paired with its
    ``tool`` replies, which providers reject if split.
    """

    def __init__(self, max_turns: int = 10) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self._max_turns = max_turns
        self._turns: list[list[dict[str, Any]]] = []

    def start_turn(self, user_message: dict[str, Any]) -> list[dict[str, Any]]:
        """Begin a new turn and return its (mutable) message list."""
        self._turns.append([user_message])
        if len(self._turns) > self._max_turns:
            dropped = len(self._turns) - self._max_turns
            self._turns = self._turns[dropped:]
            logger.debug("Dropped %d old conversation turn(s)", dropped)
        return self._turns[-1]

    def messages(self) -> list[dict[str, Any]]:
        return [m for turn in self._turns for m in turn]

    def discard_last_turn(self) -> None:
        """Remove an unfinished turn (e.g. after an LLM error) so history stays consistent."""
        if self._turns:
            self._turns.pop()

    def clear(self) -> None:
        self._turns.clear()


class FactStore:
    """Persistent long-term memory: short facts in SQLite, searchable with BM25.

    Facts live on local disk only (in the git-ignored runtime directory). Text that looks
    like a credential is refused so secrets never reach the database.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)  # owner-only data dir
        with connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS facts ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " content TEXT NOT NULL UNIQUE,"
                " created_at TEXT NOT NULL)"
            )

    def add(self, content: str) -> bool:
        """Store a fact. Returns False if it was a duplicate. Raises ValueError if invalid."""
        content = " ".join(content.split())
        if not content:
            raise ValueError("Fact is empty.")
        if len(content) > MAX_FACT_CHARS:
            raise ValueError(f"Fact is longer than {MAX_FACT_CHARS} characters.")
        if contains_secret(content):
            raise ValueError("Refusing to store text that looks like a secret or API key.")
        with connect(self._db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            if count >= MAX_FACTS:
                raise ValueError("Memory is full.")
            cur = conn.execute(
                "INSERT OR IGNORE INTO facts (content, created_at) VALUES (?, ?)",
                (content, datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
            inserted = cur.rowcount == 1
        logger.info("Fact %s", "stored" if inserted else "already known")
        return inserted

    def all(self) -> list[str]:
        with connect(self._db_path) as conn:
            return [r["content"] for r in conn.execute("SELECT content FROM facts ORDER BY id")]

    def recent(self, k: int = 10) -> list[str]:
        with connect(self._db_path) as conn:
            rows = conn.execute("SELECT content FROM facts ORDER BY id DESC LIMIT ?", (k,))
            return [r["content"] for r in rows]

    def search(self, query: str, k: int = 5) -> list[str]:
        facts = self.all()
        if not facts:
            return []
        # The store is capped at MAX_FACTS, so rebuilding a small BM25 index per search is
        # cheap and avoids keeping a second copy of the data in sync.
        index = Bm25Index()
        for i, fact in enumerate(facts):
            index.add(str(i), fact, source="memory")
        hits = index.search(query, k=k)
        # Log counts only: the query and the facts are the user's personal notes.
        logger.debug("Memory search matched %d of %d fact(s)", len(hits), len(facts))
        return [facts[int(hit.doc_id)] for hit in hits]
