"""LEVEL 7 - Multiple cooperating agents.

Concept:   Planner -> Researchers -> Writer -> Critic. One LLM, several roles. Each role has
           a narrow prompt and (for researchers) its own tools. The critic checks the writer's
           draft against the research notes and can force one revision.
Concepts to learn: decomposition, role prompts, reviewer loops, and why small focused agents are
           easier to debug than one giant prompt.
Limit:     Still only static knowledge, and cannot learn from results -> Levels 8 and 9.

Run:  python levels/07_multi_agent.py "Compare diversification and position sizing for a beginner."
"""

from __future__ import annotations

import logging
import sys

from agentlab.bootstrap import start
from agentlab.builtin_tools import calculator_tool, make_knowledge_tool
from agentlab.llm import LLMError
from agentlab.multi_agent import run_pipeline
from agentlab.retrieval import build_index_from_dir
from agentlab.tools import ToolRegistry

logger = logging.getLogger("level07")


def main() -> int:
    settings, llm = start()
    question = " ".join(sys.argv[1:]).strip() or input("Question: ").strip()
    tools = ToolRegistry([make_knowledge_tool(build_index_from_dir(settings.knowledge_dir)),
                          calculator_tool])
    logger.info("Starting level 7 pipeline (question is %d chars)", len(question))
    try:
        result = run_pipeline(llm, tools, question)
    except (LLMError, ValueError) as exc:
        logger.error("Pipeline failed: %s", exc)
        print(f"[error] {exc}")
        return 1

    print("PLAN:")
    for i, task in enumerate(result.plan, 1):
        print(f"  {i}. {task}")
    print(f"\nCRITIC: {result.critique}\n")
    print(f"FINAL ANSWER:\n{result.final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
