"""Shared pytest fixtures.

The bulk of this module is a *deterministic* mock LLM that drives the
full LangGraph end-to-end without making any network calls.

Design: we tag every system prompt with ``[AGENT:<name>]``. The mock
inspects that tag plus a few prompt keywords to dispatch to a small Python
function that returns canned JSON.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from article_learning.ingest.parser import PaperLoader
from article_learning.llm.base import ChatMessage
from article_learning.llm.mock import DeterministicMockLLM
from article_learning.models.paper import Paper

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_markdown() -> str:
    return (FIXTURES / "sample_paper.md").read_text(encoding="utf-8")


@pytest.fixture
def sample_paper(sample_markdown: str) -> Paper:
    return PaperLoader().from_markdown(sample_markdown, title="Tiny Paper")


# ---------- mock LLM ----------


def _last_user(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role.value == "user":
            return msg.content
    return ""


def _system(messages: list[ChatMessage]) -> str:
    for msg in messages:
        if msg.role.value == "system":
            return msg.content
    return ""


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _is_derivation(messages: list[ChatMessage]) -> bool:
    """Sub-agent dispatch: derivation vs response."""
    return "careful derivation" in _normalise(_system(messages))


def _is_judgment(messages: list[ChatMessage]) -> bool:
    """Challenger dispatch: initial challenge vs judgment of an answer."""
    return "evaluating whether the sub-agent's answer" in _normalise(_system(messages))


def _proposition_id_from_user(text: str) -> str:
    """Best-effort extract of "P<n>" from a user prompt for stable dispatch."""
    import re

    match = re.search(r"\bP\d+\b", text)
    return match.group(0) if match else "?"


@pytest.fixture
def deterministic_mock() -> DeterministicMockLLM:
    """A mock that lets the orchestrator finish every proposition with mixed verdicts.

    Behaviour summary (per proposition):
      * main agent: returns 3 propositions (P1 assumption, P2 lemma depending
        on P1, P3 theorem depending on P2) plus a couple of symbols.
      * sub agent: produces a short derivation, and when challenged answers
        with ``answered``.
      * challengers: cycle through ``no_issue`` and ``question`` so each
        proposition reaches CONFIRMED before max_rounds.
    """

    # Counters keyed by tag so we can vary verdicts across rounds.
    counters: dict[str, int] = defaultdict(int)

    def main_handler(_messages: list[ChatMessage]) -> str:
        plan = {
            "propositions": [
                {
                    "proposition_id": "P1",
                    "type": "assumption",
                    "statement": "f is continuous on [0,1]",
                    "formal_statement": "f \\in C([0,1])",
                    "block_id": "block-2",
                    "citation_quote": "we assume that $f$ is continuous on $[0, 1]$",
                    "depends_on": [],
                },
                {
                    "proposition_id": "P2",
                    "type": "lemma",
                    "statement": "If f is continuous on [0,1] then f is bounded.",
                    "formal_statement": None,
                    "block_id": "block-3",
                    "citation_quote": (
                        "If $f$ is continuous on $[0, 1]$, then $f$ is bounded on $[0, 1]$."
                    ),
                    "depends_on": ["P1"],
                },
                {
                    "proposition_id": "P3",
                    "type": "theorem",
                    "statement": "If f is continuous on [0,1] then the integral of f exists.",
                    "formal_statement": None,
                    "block_id": "block-4",
                    "citation_quote": (
                        "If $f$ is continuous on $[0, 1]$, then $\\int_0^1 f(x)\\,dx$ exists."
                    ),
                    "depends_on": ["P1", "P2"],
                },
            ],
            "symbols": [
                {
                    "name": "f",
                    "description": "Real-valued continuous function on [0,1]",
                    "introduced_in_block": "block-2",
                    "scope_blocks": [],
                },
                {
                    "name": "[0,1]",
                    "description": "Closed unit interval",
                    "introduced_in_block": "block-1",
                    "scope_blocks": [],
                },
            ],
        }
        return json.dumps(plan)

    def sub_handler(messages: list[ChatMessage]) -> str:
        user = _last_user(messages)
        prop_id = _proposition_id_from_user(user)
        if _is_derivation(messages):
            return json.dumps(
                {
                    "derivation": (
                        f"Derivation for {prop_id}: by the source block "
                        "this follows from prior dependencies via standard real-analysis tools."
                    ),
                    "extra_citations": [],
                    "notes": None,
                }
            )
        # Response to challenge.
        return json.dumps(
            {
                "answer": (
                    f"For {prop_id}, the question is addressed by the cited block "
                    "and the dependency chain; details are in the source quote."
                ),
                "additional_citations": [],
                "verdict": "answered",
            }
        )

    def make_challenger_handler(kind: str):
        def handler(messages: list[ChatMessage]) -> str:
            key = f"{kind}:{_proposition_id_from_user(_last_user(messages))}"
            counters[key] += 1
            n = counters[key]
            if _is_judgment(messages):
                # Always rule "resolved" so the proposition moves toward CONFIRMED.
                return json.dumps({"verdict": "resolved", "rationale": "answer covers it"})
            # Alternate question / no_issue so we exercise both branches.
            if n % 2 == 1:
                return json.dumps(
                    {
                        "verdict": "question",
                        "question": f"[{kind}] Round {n}: justify the key step for this claim.",
                        "rationale": f"Synthetic {kind} probe number {n}.",
                    }
                )
            return json.dumps(
                {
                    "verdict": "no_issue",
                    "question": "",
                    "rationale": f"{kind} pass on round {n}.",
                }
            )

        return handler

    mock = DeterministicMockLLM()
    mock.register("main", main_handler)
    mock.register("sub", sub_handler)
    mock.register("logic", make_challenger_handler("logic"))
    mock.register("assumption", make_challenger_handler("assumption"))
    mock.register("counterexample", make_challenger_handler("counterexample"))
    mock.register("citation", make_challenger_handler("citation"))
    return mock
