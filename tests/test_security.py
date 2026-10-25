"""Security checks: secret redaction, private file permissions, log hygiene, and a repo scan.

The repo scan is the regression guard for this file's purpose: if a real key is ever pasted
into source, docs or config, `pytest` fails before it can be committed or shared.
"""

from __future__ import annotations

import logging
import os
import stat
import sys
from pathlib import Path

import pytest
from conftest import FakeLLM, text_reply, tool_reply
from fake_secrets import (
    FAKE_AWS_KEY_ID,
    FAKE_BEARER,
    FAKE_GITHUB_TOKEN,
    FAKE_GROQ_KEY,
    FAKE_JWT,
    FAKE_OPENAI_KEY,
    FAKE_OPENROUTER_KEY,
    FAKE_PRIVATE_KEY,
)

from agentlab.agent import Agent
from agentlab.builtin_tools import make_memory_tools
from agentlab.db import connect
from agentlab.log import PrivateRotatingFileHandler, RedactingFormatter, contains_secret, redact
from agentlab.memory import FactStore
from agentlab.tools import ToolRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")


# ---------------------------------------------------------------- redaction
@pytest.mark.parametrize("secret", [
    FAKE_OPENAI_KEY, FAKE_OPENROUTER_KEY, FAKE_GROQ_KEY, FAKE_AWS_KEY_ID, FAKE_BEARER,
    FAKE_GITHUB_TOKEN, FAKE_JWT, FAKE_PRIVATE_KEY,
])
def test_every_known_key_shape_is_redacted(secret):
    assert contains_secret(secret)
    assert secret not in redact(f"prefix {secret} suffix")


def test_url_credentials_and_assignments_are_redacted():
    # Built from parts so the repo scan below doesn't flag this test data.
    out = redact("connect to postgres://" + "admin:hunter2pass" + "@db.example.com/app")
    assert "hunter2pass" not in out and "db.example.com" in out  # host kept, password gone
    out = redact('config: api_key="abcd1234efgh5678" and password: correcthorsebattery')
    assert "abcd1234efgh5678" not in out and "correcthorsebattery" not in out
    assert "api_key" in out  # the name stays so the log line is still readable


@pytest.mark.parametrize("text", [
    "The RSI is 70", "token budget exceeded", "password manager: bitwarden",
    "I want to retire at 60 with $2M", "AAPL closed at 189.50",
])
def test_ordinary_text_is_not_redacted(text):
    assert redact(text) == text and not contains_secret(text)


def test_formatter_redacts_secrets_inside_tracebacks():
    """A filter alone misses these: the traceback text is only added at format time."""
    try:
        raise RuntimeError(f"upstream rejected key {FAKE_OPENROUTER_KEY}")
    except RuntimeError:
        record = logging.LogRecord("t", logging.ERROR, "f", 1, "tool crashed", None, sys.exc_info())
    formatted = RedactingFormatter("%(message)s").format(record)
    assert "tool crashed" in formatted and "RuntimeError" in formatted
    assert "0123456789abcdef" not in formatted


# ---------------------------------------------------------------- file permissions
@posix_only
def test_new_databases_are_owner_only(tmp_path):
    path = tmp_path / "data" / "journal.sqlite3"
    path.parent.mkdir()
    with connect(path) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


@posix_only
def test_fact_store_creates_private_directory_and_file(tmp_path):
    db = tmp_path / "runtime" / "memory.sqlite3"
    FactStore(db)
    assert stat.S_IMODE(os.stat(db).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(db.parent).st_mode) == 0o700


@posix_only
def test_log_file_is_owner_only_including_after_rotation(tmp_path):
    log_path = tmp_path / "agentlab.log"
    handler = PrivateRotatingFileHandler(log_path, maxBytes=200, backupCount=1)
    logger = logging.getLogger("test_private_rotation")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        for i in range(20):  # enough output to force at least one rollover
            logger.info("line %d %s", i, "x" * 40)
    finally:
        logger.removeHandler(handler)
        handler.close()
    assert (tmp_path / "agentlab.log.1").exists(), "rollover did not happen"
    assert stat.S_IMODE(os.stat(log_path).st_mode) == 0o600


# ---------------------------------------------------------------- log hygiene
def test_tool_arguments_are_not_written_to_the_log(caplog):
    """`remember` arguments are personal notes; only the tool name and size may be logged."""
    private_note = "my sister's surprise party is on the 14th"
    store_dir = Path(os.environ["AGENTLAB_DATA_DIR"])
    registry = ToolRegistry(make_memory_tools(FactStore(store_dir / "m.sqlite3")))
    llm = FakeLLM([tool_reply("remember", f'{{"fact": "{private_note}"}}'), text_reply("Noted.")])
    with caplog.at_level(logging.DEBUG):
        Agent(llm, registry, "sys").run("remember this")
    assert "surprise party" not in caplog.text
    assert "remember" in caplog.text  # the tool name is still there for debugging


def test_refused_secret_fact_is_not_logged_or_stored(caplog, tmp_path):
    registry = ToolRegistry(make_memory_tools(FactStore(tmp_path / "m.sqlite3")))
    with caplog.at_level(logging.DEBUG):
        out = registry.execute("remember", f'{{"fact": "my key is {FAKE_OPENAI_KEY}"}}')
    assert out.startswith("Error:")
    assert FAKE_OPENAI_KEY not in caplog.text
    assert FactStore(tmp_path / "m.sqlite3").all() == []


# ---------------------------------------------------------------- repo scan
_SCANNED_SUFFIXES = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".json", ".yaml", ".yml",
                     ".example", ".sh"}
_SKIPPED_DIRS = {"venv", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "runtime",
                 "agentlab.egg-info", ".git"}


def _source_files():
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or _SKIPPED_DIRS & set(path.relative_to(PROJECT_ROOT).parts):
            continue
        # The real .env is intentionally excluded: it is git-ignored and holds the user's keys.
        if path.name == ".env":
            continue
        if path.suffix in _SCANNED_SUFFIXES or path.name.startswith(".env."):
            yield path


def test_no_key_shaped_strings_in_source_docs_or_config():
    offenders = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            if contains_secret(line, include_assignments=False):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{number}")
    assert not offenders, f"key-shaped strings found (not printing them): {offenders}"


def test_env_example_has_no_filled_in_secrets():
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    for line in example.splitlines():
        name, _, value = line.partition("=")
        if line.startswith("#") or not value:
            continue
        # Only non-secret settings may have a value in the example file.
        assert not any(word in name for word in ("KEY", "SECRET", "TOKEN", "PASSWORD")), name


def test_env_file_is_git_ignored():
    ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignore and ".env.*" in ignore and "!.env.example" in ignore
