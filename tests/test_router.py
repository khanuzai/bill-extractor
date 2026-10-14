"""Question routing: keyword rules first, the model (mocked) for ambiguous questions."""

import pytest
from fakes import message, mock_client

from src.extraction.client import ExtractionConfig
from src.rag.router import Route, match_rules, route_question

CONFIG = ExtractionConfig(model="claude-haiku-4-5", effort=None)


@pytest.mark.parametrize(
    "question",
    [
        "Which bill has the highest total amount due?",
        "What is the average total across all CAD bills?",
        "How many bills have no due date?",
        "Show the top 3 bills sorted by usage",
        "How has my electricity usage changed over time?",
        "What is the sum of the totals for all accounts?",
    ],
)
def test_aggregate_questions_route_by_rules(question: str) -> None:
    decision = route_question(question)
    assert decision.route is Route.AGGREGATE
    assert decision.method == "rules"
    assert decision.reason.startswith("matched aggregate rules")


@pytest.mark.parametrize(
    "question",
    [
        "Why does the Acme Power bill include a 'Balance forward' line?",
        "Explain the Carbon levy charge on account 123.",
        "What does 'CR' mean on the Lakeside bill?",
        "What is the 'Delivery charge' line on the Acme bill?",
        "Who should I contact about account AC-1?",
        "What does the Acme bill say about late payments?",
    ],
)
def test_semantic_questions_route_by_rules(question: str) -> None:
    decision = route_question(question)
    assert decision.route is Route.SEMANTIC
    assert decision.method == "rules"


def test_rules_report_which_patterns_matched() -> None:
    assert match_rules("Why is the highest bill so high?") == (["superlative"], ["why"])


def test_ambiguous_question_goes_to_the_model_with_its_reason() -> None:
    client = mock_client(message({"route": "aggregate", "reason": "asks for the largest value across bills"}))

    decision = route_question("What is the largest single credit on any bill?", client, CONFIG)

    assert decision.route is Route.AGGREGATE
    assert decision.method == "llm"
    assert "matched both" in decision.reason and "largest value" in decision.reason
    assert (decision.input_tokens, decision.output_tokens) == (100, 20)
    sent = client.messages.create.call_args.kwargs
    assert "<question>" in sent["messages"][0]["content"]


def test_question_matching_no_rule_goes_to_the_model() -> None:
    client = mock_client(message({"route": "semantic", "reason": "asks about one bill's text"}))
    decision = route_question("Tell me about the Acme bill.", client, CONFIG)
    assert decision.route is Route.SEMANTIC
    assert decision.method == "llm"
    assert decision.reason.startswith("matched no rule")


def test_unambiguous_question_never_calls_the_model() -> None:
    client = mock_client()
    route_question("How many bills are in CAD?", client, CONFIG)
    client.messages.create.assert_not_called()


def test_without_a_model_ambiguous_questions_default_to_semantic() -> None:
    decision = route_question("Tell me about the Acme bill.")
    assert (decision.route, decision.method) == (Route.SEMANTIC, "default")


def test_unusable_model_reply_defaults_to_semantic() -> None:
    client = mock_client(message({"route": "both", "reason": "?"}))
    decision = route_question("Tell me about the Acme bill.", client, CONFIG)
    assert (decision.route, decision.method) == (Route.SEMANTIC, "default")
