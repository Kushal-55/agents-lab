"""LEVEL 6 - Retrieval-Augmented Generation (RAG).

Concept:   Instead of trusting the model's memory, retrieve relevant passages from your own
           documents and make it answer FROM them, with citations.
Concepts to learn: chunking, BM25 ranking (built from scratch - no paid embeddings), grounding
           prompts, and refusing to answer when the documents don't cover the question.
Docs:      data/knowledge/*.md - add your own notes there (.md / .txt only).
Limit:     Static documents only. No live market data -> Level 8.

Run:  python levels/06_rag_agent.py
Try:  "What does an RSI above 70 mean?"  then  "What is the capital of France?" (should refuse)
"""

from __future__ import annotations

import logging

from agentlab.agent import Agent
from agentlab.bootstrap import start
from agentlab.builtin_tools import make_knowledge_tool
from agentlab.memory import ConversationMemory
from agentlab.repl import print_tool_call, run_repl
from agentlab.retrieval import build_index_from_dir
from agentlab.tools import ToolRegistry

logger = logging.getLogger("level06")

SYSTEM = (
    "You answer finance questions ONLY from the knowledge base. Always call `search_knowledge` "
    "first. Quote or paraphrase the passages and cite the source file in brackets, e.g. "
    "[technical_indicators.md]. If the passages do not contain the answer, say you don't know "
    "from the knowledge base - do not guess."
)


def main() -> int:
    settings, llm = start()
    # Chunk and index every .md/.txt file in data/knowledge once, at start-up.
    index = build_index_from_dir(settings.knowledge_dir)
    if len(index) == 0:
        logger.error("No documents indexed from %s", settings.knowledge_dir)
        print(f"No documents found in {settings.knowledge_dir}")
        return 1
    logger.info("Starting level 6 with %d indexed chunk(s)", len(index))
    registry = ToolRegistry([make_knowledge_tool(index)])
    agent = Agent(llm, registry, SYSTEM, memory=ConversationMemory(6),
                  on_tool_call=print_tool_call)
    return run_repl(agent, f"RAG agent over {len(index)} chunks of finance notes.")


if __name__ == "__main__":
    raise SystemExit(main())
