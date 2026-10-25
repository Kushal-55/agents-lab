"""Tools shared across the learning levels: calculator, clock, finance calculators, memory
and knowledge-base search. Each is a ``Tool`` (name, description, JSON schema, function)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from agentlab.memory import FactStore
from agentlab.retrieval import Bm25Index
from agentlab.safe_math import UnsafeExpressionError, safe_eval
from agentlab.tools import Tool, ToolError, object_schema, param

logger = logging.getLogger(__name__)


def _calculate(expression: str) -> str:
    try:
        value = safe_eval(expression)
    except UnsafeExpressionError as exc:
        logger.info("Calculator rejected an expression (%d chars): %s", len(expression), exc)
        raise ToolError(str(exc)) from exc
    return f"{expression.strip()} = {value:,.10g}"


def _current_datetime() -> str:
    now = datetime.now(timezone.utc)
    return f"Current UTC date/time: {now:%A, %Y-%m-%d %H:%M:%S}"


calculator_tool = Tool(
    name="calculator",
    description=(
        "Evaluate an arithmetic expression exactly. Supports + - * / // % ** parentheses, "
        "sqrt, log, log10, exp, abs, round, min, max, pi, e. Use it for ALL maths."
    ),
    parameters=object_schema({"expression": param("string", "e.g. '1200 * (1 + 0.05) ** 10'")},
                             ["expression"]),
    func=_calculate,
)

datetime_tool = Tool(
    name="current_datetime",
    description="Get the current date and time in UTC.",
    parameters=object_schema({}, []),
    func=_current_datetime,
)


def _monthly_payment(principal: float, annual_rate_pct: float, years: int) -> str:
    months = years * 12
    if annual_rate_pct == 0:
        payment = principal / months  # no interest: just split the principal evenly
    else:
        # Standard amortisation formula: P * r(1+r)^n / ((1+r)^n - 1), with r the monthly rate.
        r = annual_rate_pct / 100 / 12
        payment = principal * r * (1 + r) ** months / ((1 + r) ** months - 1)
    total = payment * months
    return (
        f"Loan ${principal:,.2f} at {annual_rate_pct}% for {years} years: "
        f"monthly payment ${payment:,.2f}, total paid ${total:,.2f}, "
        f"total interest ${total - principal:,.2f}."
    )


def _compound_growth(
    principal: float, annual_rate_pct: float, years: int, monthly_contribution: float = 0.0
) -> str:
    r = annual_rate_pct / 100 / 12
    months = years * 12
    growth = principal * (1 + r) ** months
    # Future value of an annuity for the monthly contributions (plain sum when the rate is 0).
    contrib = monthly_contribution * months if r == 0 else (
        monthly_contribution * (((1 + r) ** months - 1) / r)
    )
    final = growth + contrib
    invested = principal + monthly_contribution * months
    return (
        f"After {years} years at {annual_rate_pct}%/yr (monthly compounding): "
        f"${final:,.2f} from ${invested:,.2f} invested "
        f"(growth ${final - invested:,.2f}). Assumes a constant return; real returns vary."
    )


loan_payment_tool = Tool(
    name="loan_payment",
    description="Fixed monthly payment for an amortised loan or mortgage.",
    parameters=object_schema(
        {
            "principal": param("number", "Loan amount in dollars", minimum=0.01, maximum=1e10),
            "annual_rate_pct": param("number", "Annual interest rate in percent", minimum=0,
                                     maximum=100),
            "years": param("integer", "Loan term in years", minimum=1, maximum=60),
        },
        ["principal", "annual_rate_pct", "years"],
    ),
    func=_monthly_payment,
)

compound_growth_tool = Tool(
    name="compound_growth",
    description="Future value of an investment with optional monthly contributions.",
    parameters=object_schema(
        {
            "principal": param("number", "Starting amount in dollars", minimum=0, maximum=1e10),
            "annual_rate_pct": param("number", "Assumed annual return in percent", minimum=-50,
                                     maximum=100),
            "years": param("integer", "Years to invest", minimum=1, maximum=80),
            "monthly_contribution": param("number", "Added each month", minimum=0, maximum=1e8),
        },
        ["principal", "annual_rate_pct", "years"],
    ),
    func=_compound_growth,
)


def make_memory_tools(store: FactStore) -> list[Tool]:
    """Tools letting the agent write to and read from persistent long-term memory."""

    def remember(fact: str) -> str:
        try:
            stored = store.add(fact)
        except ValueError as exc:
            # Includes the "looks like a secret" refusal; the fact itself is not logged.
            logger.info("remember() refused a fact: %s", exc)
            raise ToolError(str(exc)) from exc
        return "Saved." if stored else "I already knew that."

    def recall(query: str) -> str:
        facts = store.search(query, k=5)
        if facts:
            return "\n".join(f"- {f}" for f in facts)
        # Keyword search misses synonyms ("goals" vs "saving for a house"); the store is small,
        # so fall back to the most recent facts rather than claiming we know nothing.
        recent = store.recent(10)
        if recent:
            return "No keyword match; most recent memories:\n" + "\n".join(f"- {f}" for f in recent)
        return "Memory is empty."

    return [
        Tool(
            "remember",
            "Save a short, durable fact about the user or their goals for future sessions. "
            "Never save passwords, API keys or other secrets.",
            object_schema({"fact": param("string", "One self-contained sentence")}, ["fact"]),
            remember,
        ),
        Tool(
            "recall",
            "Search long-term memory for facts relevant to a query. Use broad keywords.",
            object_schema({"query": param("string", "What to look for")}, ["query"]),
            recall,
        ),
    ]


def make_knowledge_tool(index: Bm25Index, name: str = "search_knowledge") -> Tool:
    """A retrieval tool over a local document index (the 'R' in RAG)."""

    def search(query: str) -> str:
        hits = index.search(query, k=3)
        logger.info("Knowledge search (%d chars) returned %d hit(s)", len(query), len(hits))
        if not hits:
            return "No relevant passages found in the knowledge base."
        return "\n\n".join(f"[source: {h.source}]\n{h.text}" for h in hits)

    return Tool(
        name,
        "Search the local finance knowledge base and return the most relevant passages "
        "with their source file names. Cite the source in your answer.",
        object_schema({"query": param("string", "Keywords describing the topic")}, ["query"]),
        search,
    )
