"""Tests for B-group challengers (logic / assumption / counterexample / citation)."""

from __future__ import annotations

import json

from article_learning.agents.base import AgentContext
from article_learning.agents.group_a.main_agent import MainAgent
from article_learning.agents.group_b.assumption_challenger import AssumptionChallenger
from article_learning.agents.group_b.base_challenger import ChallengeVerdict
from article_learning.agents.group_b.citation_checker import CitationChecker
from article_learning.agents.group_b.counterexample_constructor import CounterexampleConstructor
from article_learning.agents.group_b.logic_challenger import LogicChallenger
from article_learning.llm.mock import DeterministicMockLLM, ScriptedMockLLM
from article_learning.models.paper import Paper


def test_logic_challenger_emits_question_and_judges_resolved(
    deterministic_mock: DeterministicMockLLM, sample_paper: Paper
):
    ctx = AgentContext(paper_title="Tiny Paper")
    bb = MainAgent(deterministic_mock).plan(sample_paper, ctx)
    challenger = LogicChallenger(deterministic_mock)
    outcome = challenger.challenge(bb.get("P1"), sample_paper, bb, ctx, history=[])
    assert outcome.verdict == ChallengeVerdict.QUESTION
    assert outcome.question

    judgment = challenger.judge(bb.get("P1"), outcome, "the answer is by EVT", history=[])
    assert judgment.verdict == "resolved"


def test_each_challenger_kind_is_distinct():
    # Distinct prompts -> distinct system messages -> distinct mock calls.
    classes = [LogicChallenger, AssumptionChallenger, CounterexampleConstructor, CitationChecker]
    kinds = {cls.challenger_kind for cls in classes}
    assert kinds == {"logic", "assumption", "counterexample", "citation"}


def test_no_issue_short_circuits_judging(sample_paper: Paper):
    """Verdict no_issue should NOT call the LLM a second time for judging."""
    script = [
        json.dumps({"verdict": "no_issue", "question": "", "rationale": "looks fine"}),
    ]
    mock = ScriptedMockLLM(script=script)

    # Construct a tiny blackboard with one proposition we can challenge.
    from article_learning.models.blackboard import Blackboard
    from article_learning.models.proposition import (
        Proposition,
        PropositionType,
        SourceCitation,
    )

    bb = Blackboard()
    bb.add_proposition(
        Proposition(
            proposition_id="P1",
            type=PropositionType.LEMMA,
            statement="x",
            block_id=sample_paper.blocks[0].block_id,
            citations=[SourceCitation(block_id=sample_paper.blocks[0].block_id, quote="x")],
        )
    )
    challenger = LogicChallenger(mock)
    outcome = challenger.challenge(
        bb.get("P1"), sample_paper, bb, AgentContext(paper_title="t"), history=[]
    )
    assert outcome.verdict == ChallengeVerdict.NO_ISSUE
    judgment = challenger.judge(bb.get("P1"), outcome, "", history=[])
    assert judgment.verdict == "no_issue"
    # Only one LLM call - no judgment round.
    assert len(mock.calls) == 1
