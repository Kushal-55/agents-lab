"""The agent loop: reason -> call tools -> observe -> repeat until done.

This is the same pseudocode as every agent framework:

    while the model asks for tools (bounded by max_steps):
        run the tools, append the results to the conversation
    return the model's final text
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentlab.llm import ChatModel
from agentlab.memory import ConversationMemory
from agentlab.tools import ToolRegistry

logger = logging.getLogger(__name__)

MAX_USER_INPUT_CHARS = 4000
STEP_LIMIT_NOTICE = (
    "You have used all available tool steps. Answer now using only the information already "
    "gathered, and say clearly if anything is missing."
)


@dataclass(frozen=True)
class ToolTrace:
    name: str
    arguments: str
    result: str


@dataclass
class AgentResult:
    answer: str
    steps: int
    tool_traces: list[ToolTrace] = field(default_factory=list)
    hit_step_limit: bool = False


class Agent:
    def __init__(
        self,
        llm: ChatModel,
        registry: ToolRegistry,
        system_prompt: str,
        *,
        max_steps: int = 6,
        memory: ConversationMemory | None = None,
        max_tokens: int = 1024,
        on_tool_call: Callable[[ToolTrace], None] | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self._llm = llm
        self._registry = registry
        self._system = {"role": "system", "content": system_prompt}
        self._max_steps = max_steps
        self._memory = memory or ConversationMemory()
        self._max_tokens = max_tokens
        self._on_tool_call = on_tool_call

    def run(self, user_input: str) -> AgentResult:
        user_input = user_input.strip()
        if not user_input:
            raise ValueError("Input is empty.")
        if len(user_input) > MAX_USER_INPUT_CHARS:
            raise ValueError(f"Input is longer than {MAX_USER_INPUT_CHARS} characters.")

        turn = self._memory.start_turn({"role": "user", "content": user_input})
        try:
            return self._loop(turn)
        except Exception:
            self._memory.discard_last_turn()  # never keep a half-finished turn
            raise

    def _loop(self, turn: list[dict[str, Any]]) -> AgentResult:
        specs = self._registry.specs() or None
        result = AgentResult(answer="", steps=0)

        for step in range(1, self._max_steps + 1):
            result.steps = step
            reply = self._llm.chat(
                self._messages(), specs, max_tokens=self._max_tokens
            )
            turn.append(reply.as_message())

            if not reply.tool_calls:
                result.answer = reply.content.strip() or "(The model returned an empty answer.)"
                logger.info("Agent finished in %d step(s), %d tool call(s)", step,
                            len(result.tool_traces))
                return result

            for call in reply.tool_calls:
                output = self._registry.execute(call.name, call.arguments)
                trace = ToolTrace(call.name, call.arguments, output)
                result.tool_traces.append(trace)
                turn.append({"role": "tool", "tool_call_id": call.id, "content": output})
                # Arguments can contain what the user typed (e.g. a fact to remember), so only
                # the tool name and the size are written to the log file.
                logger.info("Tool call: %s (%d chars of arguments)", call.name, len(call.arguments))
                if self._on_tool_call:
                    self._on_tool_call(trace)

        # Step budget exhausted: ask for a final answer without offering tools.
        logger.warning("Agent hit the %d-step limit", self._max_steps)
        result.hit_step_limit = True
        turn.append({"role": "user", "content": STEP_LIMIT_NOTICE})
        final = self._llm.chat(self._messages(), None, max_tokens=self._max_tokens)
        turn.append(final.as_message())
        result.answer = final.content.strip() or "(Stopped: step limit reached.)"
        return result

    def _messages(self) -> list[dict[str, Any]]:
        return [self._system, *self._memory.messages()]
