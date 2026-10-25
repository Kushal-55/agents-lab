"""Small SQLite helper: one connection per operation, always committed or rolled back, then closed.

Short-lived connections avoid sharing a connection between threads and make it impossible to
leave a transaction open by accident. Both the journal and the fact store use this.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


def _create_private_file(db_path: Path | str) -> None:
    """Create an empty database file readable only by the current user (mode 600).

    The journal and memory databases hold the user's forecasts and personal notes. SQLite would
    create the file with the default umask (usually world-readable), so create it ourselves
    first. An empty file is a valid new SQLite database.
    """
    if str(db_path) == ":memory:":
        return
    path = Path(db_path)
    if not path.exists():
        os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))


@contextmanager
def connect(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    """Open a connection; commit on success, roll back on error, always close.

    Use parameterised queries (``?`` placeholders) with this connection; never build SQL
    from strings that include model or user input.
    """
    _create_private_file(db_path)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        # Undo any partial writes so the database never holds half of a logical operation.
        logger.warning("Database error on %s, rolling back", db_path)
        conn.rollback()
        raise
    finally:
        conn.close()
