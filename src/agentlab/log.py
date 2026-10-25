"""Logging setup with automatic secret redaction.

Library modules use ``logging.getLogger(__name__)``. Entry points call
:func:`setup_logging` once. Anything that looks like an API key, token, password or private
key is scrubbed before it reaches any handler, as a defence against accidental leaks.

Redaction happens in two places on purpose:

* ``RedactingFilter`` cleans the message of each record.
* ``RedactingFormatter`` cleans the final formatted text. Tracebacks are only added at
  format time, so a filter alone would let a secret inside an exception message through.

Log files are created readable only by the current user (mode 600).
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

REDACTED = "[REDACTED]"

# (pattern, replacement) pairs for common credential shapes. Patterns are deliberately
# conservative (minimum lengths, known prefixes) to avoid redacting ordinary text.
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), REDACTED),  # OpenAI / OpenRouter / Anthropic style
    (re.compile(r"gsk_[A-Za-z0-9]{16,}"), REDACTED),  # Groq
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), REDACTED),  # Google
    (re.compile(r"AKIA[0-9A-Z]{12,}"), REDACTED),  # AWS access key id
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"), REDACTED),  # GitHub
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), REDACTED),  # GitHub fine-grained
    (re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"), REDACTED),  # Slack
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{12,}"), REDACTED),  # Authorization headers
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"),
     REDACTED),  # JWT
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?"
                r"(?:-----END [A-Z ]*PRIVATE KEY-----|$)"),
     REDACTED),  # PEM private key block (through the END line, or to the end of the text)
    # A username:password pair embedded in a URL before the host. Keep the scheme and host.
    (re.compile(r"(?<=://)[^/\s:@]+:[^/\s:@]+(?=@)"), REDACTED),
]

# name=value assignments for obviously sensitive names. The name is kept, the value hidden.
# Kept separate from the shape patterns above because it also matches harmless source code
# such as ``api_key = os.getenv(...)``, so source-code scans must not use it.
_ASSIGNMENT_PATTERN = (
    re.compile(r"(?i)\b(api[_-]?key|secret[_-]?(?:access[_-]?)?key|password|passwd|token)"
               r"(\s*[=:]\s*)['\"]?[^\s'\",;]{8,}"),
    rf"\1\2{REDACTED}",
)


def redact(text: str) -> str:
    """Replace anything that looks like a credential with a placeholder."""
    for pattern, replacement in (*_SECRET_PATTERNS, _ASSIGNMENT_PATTERN):
        text = pattern.sub(replacement, text)
    return text


def contains_secret(text: str, *, include_assignments: bool = True) -> bool:
    """True if ``text`` looks like it contains a credential.

    Set ``include_assignments=False`` to check only for key-shaped strings (used when scanning
    source files, where ``password = some_variable`` is normal).
    """
    patterns = [p for p, _ in _SECRET_PATTERNS]
    if include_assignments:
        patterns.append(_ASSIGNMENT_PATTERN[0])
    return any(p.search(text) for p in patterns)


class RedactingFilter(logging.Filter):
    """Scrub credentials from log records (message and formatted args)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None  # already merged into msg
        return True


class RedactingFormatter(logging.Formatter):
    """Scrub credentials from the final text, including exception tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


class PrivateRotatingFileHandler(RotatingFileHandler):
    """Rotating log file that is only readable by its owner.

    ``_open`` runs for the first file and again after every rollover, so each new file gets
    the restrictive mode (a plain ``chmod`` after construction would miss rotated files).
    """

    def _open(self):  # noqa: ANN202 - matches the stdlib signature
        stream = super()._open()
        # Some filesystems have no permission bits; that is not worth failing logging over.
        with contextlib.suppress(OSError):
            os.chmod(self.baseFilename, 0o600)
        return stream


def setup_logging(level: str = "WARNING", log_file: Path | None = None) -> None:
    """Configure root logging once (idempotent).

    ``level`` applies to the console (stderr). A log file, if given, always records INFO and
    above (or DEBUG if ``level`` is DEBUG) so there is a full audit trail without console noise.
    """
    root = logging.getLogger()
    console_level = logging.getLevelName(level.upper())
    if not isinstance(console_level, int):
        console_level = logging.WARNING
    file_level = min(console_level, logging.INFO)
    if getattr(root, "_agentlab_configured", False):
        return
    root.setLevel(min(console_level, file_level))

    fmt = RedactingFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    handlers: list[tuple[logging.Handler, int]] = [(logging.StreamHandler(), console_level)]
    if log_file is not None:
        log_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        handlers.append((PrivateRotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3),
                         file_level))

    for handler, handler_level in handlers:
        handler.setLevel(handler_level)
        handler.setFormatter(fmt)
        handler.addFilter(RedactingFilter())
        root.addHandler(handler)

    # Chatty HTTP libraries can echo request details; keep them at WARNING.
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # yfinance logs its own ERRORs for bad tickers; we raise a cleaner error ourselves.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    root._agentlab_configured = True  # type: ignore[attr-defined]
