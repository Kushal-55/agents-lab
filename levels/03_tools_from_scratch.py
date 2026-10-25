"""LEVEL 3 - Tool use, built by hand.

Concept:   The model cannot do exact maths or know today's date. So we teach it a protocol:
           "reply with JSON naming a tool", then WE run the tool and show it the result.
You learn: parsing model output, dispatching to real code, and why this is brittle
           (models add prose, break JSON, hallucinate tool names).
Security:  Never eval() model output. The calculator parses an AST with a strict whitelist.
Limit:     Fragile text protocol -> Level 4 uses the provider's native function calling.

Run:  python levels/03_tools_from_scratch.py
"""

from __future__ import annotations

import json
import logging

from agentlab.bootstrap import start
from agentlab.builtin_tools import calculator_tool, datetime_tool
from agentlab.jsonutil import extract_json_object
from agentlab.llm import LLMError
from agentlab.tools import ToolRegistry

logger = logging.getLogger("level03")

SYSTEM = """You are a helpful assistant with two tools:
  calculator        -> input: {"expression": "<arithmetic>"}
  current_datetime  -> input: {}
If you need a tool, reply with ONLY a JSON object like:
  {"tool": "calculator", "args": {"expression": "1200*1.05**10"}}
After you receive the tool result, answer the user in plain language.
If no tool is needed, just answer normally."""

MAX_ROUNDS = 3


def main() -> int:
    _, llm = start()
    tools = ToolRegistry([calculator_tool, datetime_tool])
    print("Tools-from-scratch assistant. Try: 'What is 1200 invested at 5% for 10 years?'")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"quit", "exit", "q"}:
            return 0
        if not question or len(question) > 4000:
            continue

        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        try:
            for _ in range(MAX_ROUNDS):
                text = llm.chat(messages).content
                call = extract_json_object(text)
                if not call or "tool" not in call:
                    print(f"\nAgent: {text}")
                    break
                name, args = str(call["tool"]), call.get("args", {})
                logger.info("Model asked for tool %r", name)
                # Our registry validates the arguments - the model's JSON is untrusted input.
                result = tools.execute(name, json.dumps(args if isinstance(args, dict) else {}))
                print(f"   [tool] {name} -> {result}")
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": f"TOOL RESULT: {result}"}]
            else:
                print("\nAgent: I could not finish within the tool-round limit.")
        except LLMError as exc:
            print(f"[error] {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
