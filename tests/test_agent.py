"""Agent loop, conversation memory, LLM client error mapping, retrieval, fact store."""

from __future__ import annotations

from types import SimpleNamespace

import openai
import pytest
from conftest import FakeLLM, text_reply, tool_reply
from fake_secrets import FAKE_OPENAI_KEY

from agentlab.agent import Agent
from agentlab.builtin_tools import calculator_tool, make_knowledge_tool, make_memory_tools
from agentlab.config import load_settings
from agentlab.llm import LLMClient, LLMError
from agentlab.memory import ConversationMemory, FactStore
from agentlab.retrieval import Bm25Index, build_index_from_dir, chunk_text
from agentlab.tools import ToolRegistry


# ---------------------------------------------------------------- agent loop
def test_agent_calls_tool_then_answers():
    llm = FakeLLM([tool_reply("calculator", '{"expression": "6*7"}'), text_reply("It is 42.")])
    seen = []
    agent = Agent(llm, ToolRegistry([calculator_tool]), "sys", on_tool_call=seen.append)
    result = agent.run("what is 6*7?")
    assert result.answer == "It is 42." and result.steps == 2
    assert result.tool_traces[0].result == "6*7 = 42" and len(seen) == 1
    # The tool result must be sent back to the model, linked to the right call id.
    tool_msg = llm.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_1"


def test_agent_feeds_tool_errors_back_instead_of_crashing():
    llm = FakeLLM([tool_reply("calculator", '{"expression": "__import__(1)"}'),
                   text_reply("Sorry, I could not compute that.")])
    result = Agent(llm, ToolRegistry([calculator_tool]), "sys").run("hack")
    assert result.tool_traces[0].result.startswith("Error:")
    assert "could not" in result.answer


def test_agent_stops_at_step_limit_and_still_answers():
    loop = [tool_reply("calculator", '{"expression": "1+1"}', f"c{i}") for i in range(3)]
    llm = FakeLLM([*loop, text_reply("Best effort answer.")])
    result = Agent(llm, ToolRegistry([calculator_tool]), "sys", max_steps=3).run("loop forever")
    assert result.hit_step_limit and result.answer == "Best effort answer."
    assert llm.calls[-1]["tools"] is None  # the forced final call offers no tools


def test_agent_validates_input_and_rolls_back_on_llm_failure():
    memory = ConversationMemory()

    class Failing:
        def chat(self, *a, **k):
            raise LLMError("down")

    agent = Agent(Failing(), ToolRegistry(), "sys", memory=memory)
    with pytest.raises(ValueError):
        agent.run("   ")
    with pytest.raises(ValueError):
        agent.run("x" * 5000)
    with pytest.raises(LLMError):
        agent.run("hello")
    assert memory.messages() == []  # no half-finished turn left behind


def test_conversation_memory_trims_whole_turns():
    mem = ConversationMemory(max_turns=2)
    for i in range(3):
        turn = mem.start_turn({"role": "user", "content": f"q{i}"})
        turn.append({"role": "assistant", "content": f"a{i}"})
    texts = [m["content"] for m in mem.messages()]
    assert texts == ["q1", "a1", "q2", "a2"]
    with pytest.raises(ValueError):
        ConversationMemory(0)


# ---------------------------------------------------------------- LLM client error mapping
def _client_raising(exc: Exception) -> LLMClient:
    client = LLMClient(load_settings(load_env_file=False))

    def create(**_):
        raise exc

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client


def _req():
    import httpx
    return httpx.Request("POST", "http://x")


@pytest.mark.parametrize("exc,fragment", [
    (openai.APIConnectionError(request=_req()), "Ollama"),
    (openai.RateLimitError("r", response=__import__("httpx").Response(429, request=_req()),
                           body=None), "Rate limited"),
    (openai.AuthenticationError("a", response=__import__("httpx").Response(401, request=_req()),
                                body=None), "Authentication"),
])
def test_llm_errors_are_user_friendly(exc, fragment):
    with pytest.raises(LLMError, match=fragment):
        _client_raising(exc).chat([{"role": "user", "content": "hi"}])


def test_llm_parses_tool_calls():
    client = LLMClient(load_settings(load_env_file=False))
    message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
        id="abc", type="function", function=SimpleNamespace(name="calculator", arguments="{}"))])
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2))
    client._client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **_: response)))
    result = client.chat([{"role": "user", "content": "x"}])
    assert result.tool_calls[0].name == "calculator" and result.prompt_tokens == 5
    assert result.as_message()["tool_calls"][0]["id"] == "abc"


# ---------------------------------------------------------------- retrieval
def test_bm25_ranks_relevant_chunk_first():
    idx = Bm25Index()
    idx.add("a", "RSI measures overbought and oversold momentum conditions", "ind.md")
    idx.add("b", "Diversification reduces unsystematic risk across assets", "risk.md")
    idx.add("c", "Mortgage amortisation spreads payments evenly over the term", "loan.md")
    assert idx.search("what is rsi overbought")[0].source == "ind.md"
    assert idx.search("diversification risk")[0].doc_id == "b"
    assert idx.search("zzzz qqqq") == [] and idx.search("") == []


def test_chunk_text_respects_limit():
    chunks = chunk_text("\n\n".join(["para " * 30] * 10), max_chars=400)
    assert all(len(c) <= 400 + 150 for c in chunks) and len(chunks) > 1


def test_index_builder_skips_unsafe_files(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "ok.md").write_text("Compound interest grows exponentially over time.")
    (kb / "script.py").write_text("compound interest secret")  # wrong suffix
    (kb / "huge.md").write_text("x" * 1_100_000)  # too large
    secret = tmp_path / "outside.md"
    secret.write_text("compound interest TOP SECRET")
    (kb / "link.md").symlink_to(secret)  # symlink escaping the directory
    index = build_index_from_dir(kb)
    hits = index.search("compound interest")
    assert [h.source for h in hits] == ["ok.md"]


def test_knowledge_tool_reports_sources(tmp_path):
    idx = Bm25Index()
    idx.add("x", "Sharpe ratio measures return per unit of risk", "risk.md")
    out = make_knowledge_tool(idx).func("sharpe ratio")
    assert "[source: risk.md]" in out
    assert "No relevant" in make_knowledge_tool(idx).func("astronomy")


# ---------------------------------------------------------------- long-term memory
def test_fact_store_persists_dedupes_and_searches(tmp_path):
    path = tmp_path / "mem.sqlite3"
    store = FactStore(path)
    assert store.add("The user is saving for a house down payment") is True
    assert store.add("The user is saving   for a house down payment") is False  # whitespace dup
    store.add("The user prefers low-cost index funds")
    reopened = FactStore(path)  # a new session sees the same memories
    assert len(reopened.all()) == 2
    assert "index funds" in reopened.search("which funds does the user prefer")[0]


@pytest.mark.parametrize("bad", ["", "   ", "x" * 501, f"my key is {FAKE_OPENAI_KEY}"])
def test_fact_store_rejects_invalid_or_secret_facts(tmp_path, bad):
    with pytest.raises(ValueError):
        FactStore(tmp_path / "m.sqlite3").add(bad)


def test_memory_tools_return_errors_not_exceptions(tmp_path):
    reg = ToolRegistry(make_memory_tools(FactStore(tmp_path / "m.sqlite3")))
    assert reg.execute("remember", '{"fact": "I like bonds"}') == "Saved."
    assert reg.execute("remember", f'{{"fact": "token {FAKE_OPENAI_KEY}"}}').startswith("Error:")
    assert "bonds" in reg.execute("recall", '{"query": "bonds"}')
    # A query with no keyword overlap still surfaces recent memories instead of "nothing".
    assert "I like bonds" in reg.execute("recall", '{"query": "goals"}')
    empty = ToolRegistry(make_memory_tools(FactStore(tmp_path / "empty.sqlite3")))
    assert empty.execute("recall", '{"query": "goals"}') == "Memory is empty."
