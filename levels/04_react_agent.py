"""LEVEL 4 - A real agent: native function calling in a ReAct loop.

Concept:   Reason -> Act -> Observe, repeated. The provider returns structured tool calls,
           so no fragile text parsing. The agent decides which tools to call, in what order.
Concepts to learn: tool schemas (JSON Schema), the bounded agent loop, argument validation,
           and why max_steps and output limits matter.
Limit:     Each run starts blank; it forgets between sessions -> Level 5 adds memory.

Run:  python levels/04_react_agent.py
Try:  "If I borrow 250000 at 6.5% for 30 years, what is the payment? And how much interest?"
"""

from __future__ import annotations

import logging

from agentlab.agent import Agent
from agentlab.bootstrap import start
from agentlab.builtin_tools import (
    calculator_tool,
    compound_growth_tool,
    datetime_tool,
    loan_payment_tool,
)
from agentlab.memory import ConversationMemory
from agentlab.repl import print_tool_call, run_repl
from agentlab.tools import ToolRegistry

logger = logging.getLogger("level04")

SYSTEM = (
    "You are a precise personal-finance assistant. Use tools for every calculation and for "
    "the current date; never do arithmetic in your head. Explain results briefly."
)


def main() -> int:
    settings, llm = start()
    registry = ToolRegistry([calculator_tool, datetime_tool, loan_payment_tool,
                             compound_growth_tool])
    logger.info("Starting level 4 with tools: %s", ", ".join(registry.names()))
    # max_steps bounds the reason/act loop so a confused model can't call tools forever;
    # ConversationMemory(10) keeps the last 10 turns as short-term context.
    agent = Agent(llm, registry, SYSTEM, max_steps=6, memory=ConversationMemory(10),
                  on_tool_call=print_tool_call)
    return run_repl(agent, f"ReAct agent ({settings.provider}/{settings.model}), "
                           f"tools: {', '.join(registry.names())}")


if __name__ == "__main__":
    raise SystemExit(main())
