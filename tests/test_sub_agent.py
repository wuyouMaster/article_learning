"""Tests for the Group A sub-agent (per-block prover)."""

from __future__ import annotations

from article_learning.agents.base import AgentContext
from article_learning.agents.group_a.main_agent import MainAgent
from article_learning.agents.group_a.sub_agent import SubAgent
from article_learning.llm.mock import DeterministicMockLLM
from article_learning.models.paper import Paper


def test_derive_writes_a_string_to_blackboard_via_caller(
    deterministic_mock: DeterministicMockLLM, sample_paper: Paper
):
    ctx = AgentContext(paper_title="Tiny Paper")
    bb = MainAgent(deterministic_mock).plan(sample_paper, ctx)
    sub = SubAgent(deterministic_mock)
    result = sub.derive(bb.get("P1"), sample_paper, bb, ctx)
    assert result.derivation
    assert "P1" in result.derivation


def test_respond_returns_answered_when_mock_does(
    deterministic_mock: DeterministicMockLLM, sample_paper: Paper
):
    ctx = AgentContext(paper_title="Tiny Paper")
    bb = MainAgent(deterministic_mock).plan(sample_paper, ctx)
    sub = SubAgent(deterministic_mock)
    response = sub.respond(
        bb.get("P1"),
        sample_paper,
        bb,
        ctx,
        question="why does this hold?",
        history=[],
    )
    assert response.verdict == "answered"
    assert response.answer
