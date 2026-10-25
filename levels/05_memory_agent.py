"""LEVEL 5 - An agent with memory.

Concept:   Two kinds of memory.
             short-term: the rolling conversation window (ConversationMemory)
             long-term:  durable facts in SQLite the agent saves/recalls with tools (FactStore)
Concepts to learn: why memory is just "tools + storage", how to bound it, and privacy hygiene:
           facts stay on your disk (git-ignored) and secrets are refused.
Try:       Tell it a goal, quit, restart, and ask "what do you remember about me?"
Limit:     It only knows what you told it -> Level 6 gives it a knowledge base (RAG).

Run:  python levels/05_memory_agent.py
"""

from __future__ import annotations

import logging

from agentlab.agent import Agent
from agentlab.bootstrap import start
from agentlab.builtin_tools import calculator_tool, make_memory_tools
from agentlab.memory import ConversationMemory, FactStore
from agentlab.repl import print_tool_call, run_repl
from agentlab.tools import ToolRegistry

logger = logging.getLogger("level05")

SYSTEM = (
    "You are a personal finance coach with long-term memory. At the start of a conversation "
    "call `recall` to check what you know about the user. When the user shares a durable fact "
    "or goal, call `remember` with one short sentence. NEVER store passwords, account numbers "
    "or API keys. Make clear that you are not a licensed advisor."
)


def main() -> int:
    settings, llm = start()
    # Long-term memory lives in a SQLite file in data/runtime, so it survives restarts.
    store = FactStore(settings.data_dir / "memory.sqlite3")
    logger.info("Starting level 5 with %d stored fact(s)", len(store.all()))
    registry = ToolRegistry([calculator_tool, *make_memory_tools(store)])
    agent = Agent(llm, registry, SYSTEM, memory=ConversationMemory(10),
                  on_tool_call=print_tool_call)
    return run_repl(agent, f"Memory agent. {len(store.all())} fact(s) in long-term memory.")


if __name__ == "__main__":
    raise SystemExit(main())
