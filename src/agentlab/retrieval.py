"""Keyword retrieval with BM25, written from scratch.

BM25 is the ranking function behind many search engines. It scores a chunk higher when it
contains rare query words many times, with a penalty for very long chunks. For a small local
knowledge base it works well, needs no embedding model and costs nothing to run.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 1_000_000
ALLOWED_SUFFIXES = {".md", ".txt"}
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    (
        "a an and are as at be by for from has have in is it its of on or that the to was were "
        "what when which who why will with how do does i you we they this these those"
    ).split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float
    text: str
    source: str


class Bm25Index:
    """In-memory BM25 index over short text chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1, self._b = k1, b
        self._docs: list[tuple[str, str, str, Counter[str], int]] = []
        self._df: Counter[str] = Counter()
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, doc_id: str, text: str, source: str = "") -> None:
        tokens = tokenize(text)
        counts = Counter(tokens)
        self._docs.append((doc_id, text, source, counts, len(tokens)))
        self._df.update(counts.keys())
        self._total_len += len(tokens)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q_tokens = set(tokenize(query))
        if not self._docs or not q_tokens:
            return []
        n = len(self._docs)
        avg_len = self._total_len / n or 1.0
        scored: list[Hit] = []
        for doc_id, text, source, counts, length in self._docs:
            score = 0.0
            for term in q_tokens:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                # idf: terms found in few chunks are more informative than common ones.
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                # Term-frequency saturation (k1) and length normalisation (b).
                denom = tf + self._k1 * (1 - self._b + self._b * length / avg_len)
                score += idf * tf * (self._k1 + 1) / denom
            if score > 0:
                scored.append(Hit(doc_id, score, text, source))
        scored.sort(key=lambda h: h.score, reverse=True)
        logger.debug("BM25 search: %d candidate chunk(s)", len(scored))  # not the query text
        return scored[:k]


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    """Split on blank lines, then pack paragraphs into chunks up to ``max_chars``."""
    chunks: list[str] = []
    current = ""
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def build_index_from_dir(directory: Path) -> Bm25Index:
    """Index every ``.md``/``.txt`` file directly inside ``directory``.

    Safety: symlinks, files that resolve outside the directory, oversized files and
    non-text suffixes are skipped, so the agent can never be steered to read arbitrary files.
    """
    index = Bm25Index()
    root = directory.resolve()
    if not root.is_dir():
        logger.warning("Knowledge directory %s does not exist", root)
        return index
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in ALLOWED_SUFFIXES or path.is_symlink():
            continue
        resolved = path.resolve()
        if resolved.parent != root or not resolved.is_file():
            logger.warning("Skipping %s (outside knowledge directory)", path.name)
            continue
        if resolved.stat().st_size > MAX_FILE_BYTES:
            logger.warning("Skipping %s (larger than %d bytes)", path.name, MAX_FILE_BYTES)
            continue
        text = resolved.read_text(encoding="utf-8", errors="replace")
        for i, chunk in enumerate(chunk_text(text)):
            index.add(f"{path.name}#{i}", chunk, source=path.name)
    logger.info("Indexed %d chunks from %s", len(index), root)
    return index
