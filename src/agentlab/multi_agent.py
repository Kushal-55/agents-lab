"""Level 7: a small team of specialised agents.

    Planner -> Researcher(s) -> Writer -> Critic -> (Writer revises once)

Each role is the same LLM with a different prompt (and, for the researcher, different tools).
Splitting work this way keeps each prompt focused and makes failures easier to locate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agentlab.agent import Agent
from agentlab.jsonutil import extract_json_object
from agentlab.llm import ChatModel
from agentlab.memory import ConversationMemory
from agentlab.tools import ToolRegistry

logger = logging.getLogger(__name__)

MAX_SUBTASKS = 3

PLANNER_PROMPT = (
    "You are a planner. Break the user's question into at most 3 short, independent research "
    'sub-questions. Reply with ONLY JSON: {"subtasks": ["...", "..."]}'
)
RESEARCHER_PROMPT = (
    "You are a researcher. Answer the sub-question using the tools (knowledge base search, "
    "calculator). Be factual and brief, cite source file names, and say if the knowledge base "
    "has nothing relevant. Never invent numbers."
)
WRITER_PROMPT = (
    "You are a writer. Combine the research notes into one clear, well-structured answer for "
    "the original question. Use only facts present in the notes. Keep it under 250 words."
)
CRITIC_PROMPT = (
    "You are a strict reviewer. Compare the DRAFT against the NOTES. If every claim in the draft "
    "is supported by the notes and the question is answered, reply with exactly: OK\n"
    "Otherwise reply with a short list of specific problems to fix."
)


@dataclass
class PipelineResult:
    plan: list[str]
    findings: list[str] = field(default_factory=list)
    draft: str = ""
    critique: str = ""
    final: str = ""


def _ask(llm: ChatModel, system: str, user: str, max_tokens: int = 700) -> str:
    reply = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        None,
        max_tokens=max_tokens,
    )
    return reply.content.strip()


def make_plan(llm: ChatModel, question: str) -> list[str]:
    parsed = extract_json_object(_ask(llm, PLANNER_PROMPT, question, max_tokens=300)) or {}
    tasks = parsed.get("subtasks")
    if isinstance(tasks, list):
        cleaned = [str(t).strip()[:300] for t in tasks if str(t).strip()][:MAX_SUBTASKS]
        if cleaned:
            return cleaned
    logger.warning("Planner returned no usable plan; falling back to the original question")
    return [question]


def run_pipeline(llm: ChatModel, researcher_tools: ToolRegistry, question: str) -> PipelineResult:
    question = question.strip()
    if not question or len(question) > 2000:
        raise ValueError("Question must be between 1 and 2000 characters.")

    result = PipelineResult(plan=make_plan(llm, question))
    logger.info("Pipeline plan has %d sub-task(s)", len(result.plan))
    for task in result.plan:
        # A fresh single-turn researcher per sub-task, so one task's tool calls and notes
        # don't leak into the next one's context.
        researcher = Agent(llm, researcher_tools, RESEARCHER_PROMPT, max_steps=4,
                           memory=ConversationMemory(1))
        answer = researcher.run(task).answer
        result.findings.append(f"Q: {task}\nA: {answer}")
        logger.info("Researcher finished: %s", task[:80])

    notes = "\n\n".join(result.findings)
    writer_input = f"QUESTION: {question}\n\nNOTES:\n{notes}"
    result.draft = _ask(llm, WRITER_PROMPT, writer_input)
    result.critique = _ask(llm, CRITIC_PROMPT, f"{writer_input}\n\nDRAFT:\n{result.draft}", 300)
    # The critic is told to answer exactly "OK" when satisfied; tolerate case and a trailing dot.
    verdict = result.critique.strip().upper().rstrip(".")
    if not verdict or verdict == "OK":
        if not verdict:
            logger.warning("Critic returned nothing; accepting the draft as-is")
        result.final = result.draft
    else:
        logger.info("Critic requested a revision")
        result.final = _ask(
            llm, WRITER_PROMPT,
            f"{writer_input}\n\nPREVIOUS DRAFT:\n{result.draft}\n\nREVIEWER FEEDBACK:\n"
            f"{result.critique}\n\nWrite an improved final answer.",
        )
    return result
