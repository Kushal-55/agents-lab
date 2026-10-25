"""Config, logging redaction, safe math, tool registry."""

from __future__ import annotations

import logging

import pytest
from fake_secrets import FAKE_AWS_KEY_ID, FAKE_BEARER, FAKE_GROQ_KEY, FAKE_OPENROUTER_KEY

from agentlab.config import ConfigError, load_settings
from agentlab.log import RedactingFilter, contains_secret, redact
from agentlab.safe_math import UnsafeExpressionError, safe_eval
from agentlab.tools import Tool, ToolRegistry, object_schema, param


# ---------------------------------------------------------------- config
def test_default_provider_is_keyless_ollama():
    s = load_settings(load_env_file=False)
    assert s.provider == "ollama"
    assert s.base_url.startswith("http://localhost")


def test_remote_provider_requires_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        load_settings(load_env_file=False)
    assert load_settings(load_env_file=False, require_key=False).provider == "openrouter"


def test_api_key_never_appears_in_repr(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", FAKE_GROQ_KEY)
    s = load_settings(load_env_file=False)
    assert FAKE_GROQ_KEY not in repr(s)
    assert s.api_key == FAKE_GROQ_KEY


@pytest.mark.parametrize("name,value", [("LLM_PROVIDER", "nope"), ("LLM_TIMEOUT_S", "abc"),
                                        ("LLM_TEMPERATURE", "9")])
def test_invalid_config_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError):
        load_settings(load_env_file=False)


# ---------------------------------------------------------------- logging
def test_redact_scrubs_common_key_shapes():
    text = f"key={FAKE_OPENROUTER_KEY} and {FAKE_BEARER} and {FAKE_AWS_KEY_ID}"
    out = redact(text)
    assert "0123456789abcdef" not in out and "abcdefghijklmnop" not in out
    assert FAKE_AWS_KEY_ID not in out and out.count("[REDACTED]") == 3
    assert contains_secret(text) and not contains_secret("just a normal sentence")


def test_logging_filter_redacts_formatted_args():
    record = logging.LogRecord("t", logging.INFO, "f", 1, "token %s", (FAKE_OPENROUTER_KEY,), None)
    RedactingFilter().filter(record)
    assert "0123456789abcdef" not in record.getMessage()


# ---------------------------------------------------------------- safe math
@pytest.mark.parametrize("expr,expected", [
    ("2 + 3 * 4", 14), ("(1 + 2) ** 3", 27), ("sqrt(16) + abs(-2)", 6), ("10 // 3", 3),
    ("-5 + +2", -3), ("max(1, 9, 4)", 9), ("1200 * (1 + 0.05) ** 10", 1954.67),
])
def test_safe_eval_valid(expr, expected):
    assert safe_eval(expr) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')", "open('/etc/passwd')", "().__class__", "x + 1",
    "lambda: 1", "[1,2,3]", "1 if True else 2", "'a' * 5", "sqrt(x=4)", "", "   ", "2 +",
    "9 ** 9 ** 9 ** 9", "1 / 0", "(-8) ** 0.5", "log(-1)", "1" * 300, "+".join(["1"] * 100),
])
def test_safe_eval_rejects_unsafe_or_invalid(expr):
    with pytest.raises(UnsafeExpressionError):
        safe_eval(expr)


# ---------------------------------------------------------------- tool registry
def _registry() -> ToolRegistry:
    def add(a: int, b: int = 1) -> str:
        return str(a + b)

    def boom() -> str:
        raise RuntimeError("secret internal detail /Users/someone/path")

    def big() -> str:
        return "x" * 10_000

    return ToolRegistry([
        Tool("add", "add", object_schema({"a": param("integer", "a", minimum=0, maximum=100),
                                          "b": param("integer", "b")}, ["a"]), add),
        Tool("boom", "crash", object_schema({}, []), boom),
        Tool("big", "large output", object_schema({}, []), big),
    ])


def test_registry_executes_valid_call():
    assert _registry().execute("add", '{"a": 2, "b": 3}') == "5"


@pytest.mark.parametrize("args,fragment", [
    ('{"b": 1}', "Missing required"), ('{"a": "2"}', "must be of type integer"),
    ('{"a": true}', "must be of type integer"), ('{"a": 1, "z": 2}', "Unknown argument"),
    ('{"a": 1000}', "<= 100"), ('{"a": -1}', ">= 0"), ("not json", "not valid JSON"),
    ("[1, 2]", "must be a JSON object"), ("x" * 9000, "too large"),
])
def test_registry_rejects_bad_arguments(args, fragment):
    out = _registry().execute("add", args)
    assert out.startswith("Error:") and fragment in out


def test_registry_contains_tool_crashes_without_leaking_details():
    out = _registry().execute("boom", "{}")
    assert out == "Error: tool 'boom' failed unexpectedly."
    assert "/Users" not in out


def test_registry_unknown_tool_and_truncation():
    reg = _registry()
    assert "unknown tool" in reg.execute("nope", "{}")
    assert reg.execute("big", "").endswith("[truncated]") and len(reg.execute("big", "")) < 4100


def test_registry_rejects_duplicate_names():
    with pytest.raises(ValueError):
        ToolRegistry([_registry()._tools["add"], _registry()._tools["add"]])
