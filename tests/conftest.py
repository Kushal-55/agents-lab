"""Shared fixtures. Nothing here touches the network or the user's real data directory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from agentlab.finance.data import SyntheticProvider
from agentlab.llm import ChatResult, ToolCall


class FakeLLM:
    """A scripted ChatModel: returns queued replies and records what it was sent."""

    def __init__(self, replies: list[ChatResult | Callable[[list[dict[str, Any]]], ChatResult]]):
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, *, temperature=None, max_tokens=1024) -> ChatResult:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self._replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        reply = self._replies.pop(0)
        return reply(messages) if callable(reply) else reply


def text_reply(text: str) -> ChatResult:
    return ChatResult(content=text)


def tool_reply(name: str, arguments: str, call_id: str = "call_1") -> ChatResult:
    return ChatResult(content="", tool_calls=(ToolCall(call_id, name, arguments),))


@pytest.fixture
def synthetic_prices():
    return SyntheticProvider(seed=3, n_days=1200, regime_sd=0.004).fetch("TEST")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Keep tests independent of the developer's real .env and runtime data."""
    for var in ("LLM_PROVIDER", "LLM_MODEL", "OPENROUTER_API_KEY", "GROQ_API_KEY",
                "GEMINI_API_KEY", "LLM_TIMEOUT_S", "LLM_MAX_RETRIES", "LLM_TEMPERATURE",
                "LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AGENTLAB_DATA_DIR", str(tmp_path / "runtime"))
