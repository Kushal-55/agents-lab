"""LEVEL 1 - A single LLM call.

Concept:   An LLM is a function: text in, text out. No memory, no tools, no loop.
Concepts to learn: the request shape (model, messages, max_tokens) and how to handle failure.
Limit:     It forgets everything after each call -> Level 2 adds conversation.

Run:  python levels/01_llm_call.py "Explain compound interest in two sentences."
"""

from __future__ import annotations

import logging
import sys

import openai
from openai import OpenAI

from agentlab.bootstrap import start

logger = logging.getLogger("level01")
MAX_PROMPT_CHARS = 4000


def main() -> int:
    settings, _ = start()
    prompt = " ".join(sys.argv[1:]).strip() or input("Your question: ").strip()
    if not prompt or len(prompt) > MAX_PROMPT_CHARS:
        print(f"Please enter between 1 and {MAX_PROMPT_CHARS} characters.")
        return 1

    # Every free provider speaks the same OpenAI-compatible protocol, so one client fits all.
    client = OpenAI(base_url=settings.base_url, api_key=settings.api_key,
                    timeout=settings.timeout_s, max_retries=settings.max_retries)
    try:
        response = client.chat.completions.create(
            model=settings.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
    except openai.OpenAIError as exc:  # never print the raw error: it can echo request details
        logger.error("LLM call failed: %s", type(exc).__name__)
        print(f"The model call failed ({type(exc).__name__}). Check your .env and network.")
        return 1

    print(response.choices[0].message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
