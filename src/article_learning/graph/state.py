"""Shared state object passed between LangGraph nodes."""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from article_learning.agents.base import AgentContext
from article_learning.agents.group_b.base_challenger import ChallengeOutcome
from article_learning.config import get_settings
from article_learning.models.annotation import Annotation
from article_learning.models.blackboard import Blackboard
from article_learning.models.paper import Paper

CHALLENGER_CYCLE = ["logic", "assumption", "counterexample", "citation"]


class GraphState(TypedDict, total=False):
    """LangGraph carries this dict across nodes.

    The ``annotations`` field uses ``operator.add`` so it accumulates as
    propositions stream out.

    Everything else is replace-on-write because there's a single canonical
    blackboard / current proposition at any time.
    """

    paper: Paper
    blackboard: Blackboard
    context: AgentContext
    current_prop_id: str | None
    current_round: int
    pending_outcome: ChallengeOutcome | None
    pending_answer: str | None
    challenger_index: int
    annotations: Annotated[list[Annotation], add]
    max_rounds: int
    soft_pass_streak: int
    doubt_streak: int
    finished: bool


def build_initial_state(
    paper: Paper,
    *,
    blackboard: Blackboard | None = None,
    context: AgentContext | None = None,
) -> GraphState:
    settings = get_settings()
    return GraphState(
        paper=paper,
        blackboard=blackboard or Blackboard(),
        context=context or AgentContext(paper_title=paper.title),
        current_prop_id=None,
        current_round=0,
        pending_outcome=None,
        pending_answer=None,
        challenger_index=0,
        annotations=[],
        max_rounds=settings.max_rounds_per_proposition,
        soft_pass_streak=settings.soft_pass_streak,
        doubt_streak=settings.doubt_streak,
        finished=False,
    )
