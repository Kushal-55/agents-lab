"""A small, safe interactive loop shared by the agent levels."""

from __future__ import annotations

import logging

from agentlab.agent import Agent, ToolTrace
from agentlab.llm import LLMError

logger = logging.getLogger(__name__)

EXIT_WORDS = {"quit", "exit", "q"}


def print_tool_call(trace: ToolTrace) -> None:
    """Show what the agent is doing, so the loop is visible while learning."""
    print(f"   [tool] {trace.name}({trace.arguments[:100]})")


def run_repl(agent: Agent, banner: str) -> int:
    """Read questions until the user quits. LLM/validation errors never kill the session."""
    print(banner)
    print("Type 'quit' to exit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in EXIT_WORDS:
            return 0
        if not question:
            continue
        logger.debug("User question (%d chars)", len(question))  # length only, not the content
        try:
            print(f"\nAgent: {agent.run(question).answer}")
        except (LLMError, ValueError) as exc:
            logger.info("Request failed: %s", exc)
            print(f"\n[error] {exc}")
