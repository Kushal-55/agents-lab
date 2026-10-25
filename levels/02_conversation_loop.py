"""LEVEL 2 - A conversation loop.

Concept:   The model is stateless. "Memory" is just us re-sending the message history
           every turn. History is capped so cost and latency stay bounded.
Concepts to learn: roles (system/user/assistant), the loop pattern, and context-window trimming.
Limit:     It can only talk. It cannot calculate, look things up or act -> Level 3.

Run:  python levels/02_conversation_loop.py        (commands: /reset, quit)
"""

from __future__ import annotations

import logging

from agentlab.bootstrap import start
from agentlab.llm import LLMError

logger = logging.getLogger("level02")
SYSTEM = {"role": "system", "content": "You are a concise, friendly finance tutor."}
MAX_HISTORY_MESSAGES = 20  # keep the most recent messages only


def main() -> int:
    _, llm = start()
    history: list[dict[str, str]] = []
    print("Chat with a finance tutor. Commands: /reset, quit")
    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text.lower() in {"quit", "exit", "q"}:
            return 0
        if text == "/reset":
            history.clear()
            print("(history cleared)")
            continue
        if not text or len(text) > 4000:
            continue

        history.append({"role": "user", "content": text})
        history = history[-MAX_HISTORY_MESSAGES:]  # trim: oldest messages fall out
        try:
            reply = llm.chat([SYSTEM, *history])
        except LLMError as exc:
            history.pop()  # do not keep a user message that never got an answer
            print(f"[error] {exc}")
            continue
        history.append({"role": "assistant", "content": reply.content})
        print(f"\nTutor: {reply.content}")
        logger.info("tokens in/out: %d/%d", reply.prompt_tokens, reply.completion_tokens)


if __name__ == "__main__":
    raise SystemExit(main())
