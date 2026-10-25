"""JSON extraction, REPL loop, multi-agent pipeline, and that every level script compiles."""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest
from conftest import FakeLLM, text_reply, tool_reply

from agentlab.agent import Agent
from agentlab.builtin_tools import calculator_tool
from agentlab.jsonutil import extract_json_object
from agentlab.multi_agent import make_plan, run_pipeline
from agentlab.repl import run_repl
from agentlab.tools import ToolRegistry

LEVELS = sorted((Path(__file__).resolve().parents[1] / "levels").glob("*.py"))


@pytest.mark.parametrize("text,expected", [
    ('{"tool": "calculator", "args": {"expression": "1+1"}}', "calculator"),
    ('Sure! ```json\n{"tool": "calculator"}\n``` hope that helps', "calculator"),
    ('Text with {not json} then {"tool": "a"}', "a"),
    ('{"tool": "x}y", "n": {"deep": 1}} trailing', "x}y"),
])
def test_extract_json_finds_objects_in_messy_text(text, expected):
    assert extract_json_object(text)["tool"] == expected


@pytest.mark.parametrize("text", ["", "no braces", "{broken", "[1,2,3]", "{'single': 'quotes'}"])
def test_extract_json_returns_none_when_absent(text):
    assert extract_json_object(text) is None


def test_planner_falls_back_when_output_is_unusable():
    assert make_plan(FakeLLM([text_reply("I refuse to use JSON")]), "Q?") == ["Q?"]
    assert make_plan(FakeLLM([text_reply('{"subtasks": "oops"}')]), "Q?") == ["Q?"]
    many = '{"subtasks": ["a", "b", "c", "d", "e"]}'
    assert make_plan(FakeLLM([text_reply(many)]), "Q?") == ["a", "b", "c"]


def test_pipeline_runs_roles_in_order_and_revises_on_critique():
    llm = FakeLLM([
        text_reply('{"subtasks": ["What is RSI?"]}'),          # planner
        tool_reply("calculator", '{"expression": "2+2"}'),     # researcher uses a tool
        text_reply("RSI is a momentum gauge."),                # researcher answer
        text_reply("Draft one."),                              # writer
        text_reply("The draft misses the 70 threshold."),      # critic objects
        text_reply("Final answer mentioning 70."),             # writer revises
    ])
    result = run_pipeline(llm, ToolRegistry([calculator_tool]), "Explain RSI")
    assert result.plan == ["What is RSI?"] and "momentum gauge" in result.findings[0]
    assert result.draft == "Draft one." and result.final == "Final answer mentioning 70."
    assert not llm._replies  # every scripted call was consumed, in order


def test_pipeline_accepts_draft_when_critic_says_ok():
    llm = FakeLLM([text_reply('{"subtasks": ["a"]}'), text_reply("found"), text_reply("Draft"),
                   text_reply("OK.")])
    assert run_pipeline(llm, ToolRegistry(), "Q").final == "Draft"


def test_pipeline_does_not_revise_when_critic_is_silent():
    llm = FakeLLM([text_reply('{"subtasks": ["a"]}'), text_reply("found"), text_reply("Draft"),
                   text_reply("")])
    result = run_pipeline(llm, ToolRegistry(), "Q")
    assert result.final == "Draft" and not llm._replies


def test_pipeline_rejects_bad_questions():
    for bad in ("", "  ", "x" * 2001):
        with pytest.raises(ValueError):
            run_pipeline(FakeLLM([]), ToolRegistry(), bad)


def test_repl_survives_errors_and_exits_cleanly(monkeypatch, capsys):
    class Boom:
        def chat(self, *a, **k):
            raise ValueError("bad input")

    agent = Agent(Boom(), ToolRegistry(), "sys")
    inputs = iter(["", "hello", "quit"])
    monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
    assert run_repl(agent, "banner") == 0
    out = capsys.readouterr().out
    assert "banner" in out and "[error] bad input" in out


def test_repl_handles_eof(monkeypatch):
    def eof(_=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert run_repl(Agent(FakeLLM([]), ToolRegistry(), "s"), "b") == 0


@pytest.mark.parametrize("path", LEVELS, ids=lambda p: p.name)
def test_every_level_script_compiles(path, tmp_path):
    py_compile.compile(str(path), cfile=str(tmp_path / "out.pyc"), doraise=True)


def test_there_are_nine_levels():
    assert [p.name[:2] for p in LEVELS] == [f"0{i}" for i in range(1, 10)]
