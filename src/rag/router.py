"""Route a question to SQL (aggregate) or to document retrieval (semantic).

Aggregate questions need math or sorting across bills: highest, total, average,
count, trends over time. Semantic questions need the bills' text: why, what is this
charge, explain. Keyword rules decide when exactly one side matches; when both or
neither match, the model decides.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

import anthropic
from pydantic import BaseModel

from src.extraction.client import ExtractionConfig
from src.rag.llm import ask


class Route(StrEnum):
    AGGREGATE = "aggregate"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    reason: str
    method: str
    """'rules', 'llm', or 'default' (ambiguous and the model was unavailable or unusable)."""
    input_tokens: int = 0
    output_tokens: int = 0


AGGREGATE_RULES: dict[str, re.Pattern[str]] = {
    "superlative": re.compile(
        r"\b(highest|lowest|largest|smallest|biggest|most|least|maximum|minimum|cheapest|priciest|most expensive)\b"
    ),
    # "mean" alone is left out: "what does CR mean" is not a statistic.
    "statistic": re.compile(
        r"\b(average|median|sum|in total|combined|altogether|total of all)\b|\bmean (of|total|amount|usage|value)\b"
    ),
    "count": re.compile(r"\b(how many|count|number of)\b"),
    "ranking": re.compile(r"\b(rank|ranked|sort|sorted|order by|top \d+|bottom \d+)\b"),
    "trend": re.compile(r"\b(trend|over time|month over month|year over year|per month|per year|by month|by year)\b"),
    "across bills": re.compile(r"\b(across|all|every|each) (the )?(bills|accounts|providers|statements)\b"),
}

SEMANTIC_RULES: dict[str, re.Pattern[str]] = {
    "why": re.compile(r"\bwhy\b"),
    "explain": re.compile(r"\b(explain|describe|meaning|mean|means|stand for|stands for)\b"),
    "what is this charge": re.compile(
        r"\bwhat (is|are|was|were) (the|this|that|these)\b.*\b(charge|fee|credit|rider|levy|tax|adjustment|discount|line)\b"
    ),
    "who / where / how to": re.compile(r"\b(who|where|how do i|how should i|what should i)\b"),
    "says / printed": re.compile(r"\b(say|says|said|printed|mention|mentions|note|notes)\b"),
}

ROUTER_SYSTEM_PROMPT = """\
You route questions about a collection of utility bills to one of two answering paths.

- "aggregate": the answer needs math or sorting across bills, computed with SQL over \
structured fields (provider, account number, billing period, due date, usage, line \
items, total, currency): highest, lowest, total, average, count, trends over time.
- "semantic": the answer needs the text of particular bills: why something is \
charged, what a charge or note means, explanations, anything printed on a bill that \
is not one of those structured fields.

Return the route and a one-sentence reason."""


class _RouterReply(BaseModel):
    route: Route
    reason: str


def match_rules(question: str) -> tuple[list[str], list[str]]:
    """Names of the aggregate rules and semantic rules the question matches."""
    text = question.casefold()
    aggregate = [name for name, pattern in AGGREGATE_RULES.items() if pattern.search(text)]
    semantic = [name for name, pattern in SEMANTIC_RULES.items() if pattern.search(text)]
    return aggregate, semantic


def route_question(
    question: str, client: anthropic.Anthropic | None = None, config: ExtractionConfig | None = None
) -> RouteDecision:
    """Classify the question, by rules when they are unambiguous, else by the model."""
    aggregate, semantic = match_rules(question)
    if aggregate and not semantic:
        return RouteDecision(Route.AGGREGATE, f"matched aggregate rules: {', '.join(aggregate)}", "rules")
    if semantic and not aggregate:
        return RouteDecision(Route.SEMANTIC, f"matched semantic rules: {', '.join(semantic)}", "rules")

    ambiguity = (
        f"matched both aggregate ({', '.join(aggregate)}) and semantic ({', '.join(semantic)}) rules"
        if aggregate
        else "matched no rule"
    )
    if client is None or config is None:
        return RouteDecision(Route.SEMANTIC, f"{ambiguity}; no model available, defaulted to semantic", "default")
    reply = ask(client, config, ROUTER_SYSTEM_PROMPT, f"<question>\n{question}\n</question>", _RouterReply)
    usage = {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens}
    if reply.parsed is None:
        return RouteDecision(
            Route.SEMANTIC, f"{ambiguity}; model reply unusable, defaulted to semantic", "default", **usage
        )
    return RouteDecision(reply.parsed.route, f"{ambiguity}; model: {reply.parsed.reason}", "llm", **usage)
