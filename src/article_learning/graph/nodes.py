"""LangGraph node implementations.

The graph layout (see ``workflow.py`` for the wiring) is:

    plan -> schedule -> derive -> challenge -> respond -> judge -> annotate -> schedule
                |                                  |                  ^
                v                                  v                  |
              END (no work)                      annotate <-(no_issue or refuted)

``judge`` decides whether the proposition is terminal (annotate) or needs
another round (back to ``challenge``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from article_learning.agents.base import AgentContext
from article_learning.agents.group_a.main_agent import MainAgent
from article_learning.agents.group_a.sub_agent import SubAgent
from article_learning.agents.group_b.assumption_challenger import AssumptionChallenger
from article_learning.agents.group_b.base_challenger import (
    BaseChallenger,
    ChallengeOutcome,
    ChallengeVerdict,
)
from article_learning.agents.group_b.citation_checker import CitationChecker
from article_learning.agents.group_b.counterexample_constructor import CounterexampleConstructor
from article_learning.agents.group_b.logic_challenger import LogicChallenger
from article_learning.graph.state import CHALLENGER_CYCLE, GraphState
from article_learning.llm.base import LLMClient
from article_learning.models.annotation import (
    Annotation,
    ChallengeSummary,
    ConfidenceLevel,
)
from article_learning.models.blackboard import Blackboard, ChallengeRecord
from article_learning.models.proposition import Proposition, PropositionStatus

logger = logging.getLogger(__name__)


@dataclass
class AgentBundle:
    """Prebuilt agents shared by every node."""

    main: MainAgent
    sub: SubAgent
    challengers: dict[str, BaseChallenger]

    @classmethod
    def build(cls, llm: LLMClient) -> AgentBundle:
        return cls(
            main=MainAgent(llm),
            sub=SubAgent(llm),
            challengers={
                "logic": LogicChallenger(llm),
                "assumption": AssumptionChallenger(llm),
                "counterexample": CounterexampleConstructor(llm),
                "citation": CitationChecker(llm),
            },
        )


# ---------- nodes ----------


def make_plan_node(agents: AgentBundle):
    def plan_node(state: GraphState) -> GraphState:
        paper = state["paper"]
        ctx = state.get("context") or AgentContext(paper_title=paper.title)
        bb = agents.main.plan(paper, ctx)
        logger.info("Plan produced %d propositions, %d edges, cycles=%s",
                    len(bb.propositions), len(bb.dag_edges), bb.cycles())
        return {"blackboard": bb, "context": ctx}
    return plan_node


def make_schedule_node():
    def schedule_node(state: GraphState) -> GraphState:
        bb: Blackboard = state["blackboard"]
        next_id = MainAgent.select_next(bb)
        if next_id is None:
            logger.info("Scheduler: no more work, finishing")
            return {"finished": True, "current_prop_id": None}
        bb.set_status(next_id, PropositionStatus.IN_PROGRESS)
        bb.assign_owner(next_id, "sub-default")
        logger.info("Scheduler: picked %s", next_id)
        return {
            "current_prop_id": next_id,
            "current_round": 0,
            "challenger_index": 0,
            "pending_outcome": None,
            "pending_answer": None,
            "blackboard": bb,
            "finished": False,
        }
    return schedule_node


def make_derive_node(agents: AgentBundle):
    def derive_node(state: GraphState) -> GraphState:
        bb: Blackboard = state["blackboard"]
        prop_id = state["current_prop_id"]
        if prop_id is None:
            return {}
        prop = bb.get(prop_id)
        ctx: AgentContext = state["context"]
        result = agents.sub.derive(prop, state["paper"], bb, ctx)
        bb.record_derivation(prop_id, result.derivation)
        for citation in result.extra_citations:
            prop.citations.append(citation)
        return {"blackboard": bb}
    return derive_node


def make_challenge_node(agents: AgentBundle):
    def challenge_node(state: GraphState) -> GraphState:
        bb: Blackboard = state["blackboard"]
        prop_id = state["current_prop_id"]
        if prop_id is None:
            return {}
        prop = bb.get(prop_id)

        cycle_idx = state.get("challenger_index", 0)
        challenger_kind = CHALLENGER_CYCLE[cycle_idx % len(CHALLENGER_CYCLE)]
        challenger = agents.challengers[challenger_kind]
        history = bb.proposition_history(prop_id)
        outcome = challenger.challenge(prop, state["paper"], bb, state["context"], history)
        logger.info(
            "Round %d challenger=%s verdict=%s",
            state.get("current_round", 0), challenger_kind, outcome.verdict,
        )

        round_idx = state.get("current_round", 0) + 1
        verdict_str = (
            "pending"
            if outcome.verdict == ChallengeVerdict.QUESTION
            else outcome.verdict.value
        )
        record = ChallengeRecord(
            proposition_id=prop_id,
            round_index=round_idx,
            challenger=challenger_kind,
            question=outcome.question or "(no question)",
            verdict=verdict_str,
        )
        bb.log_challenge(record)
        if outcome.verdict == ChallengeVerdict.QUESTION:
            bb.add_open_question(prop_id, outcome.question)

        return {
            "blackboard": bb,
            "pending_outcome": outcome,
            "pending_answer": None,
            "current_round": round_idx,
            "challenger_index": cycle_idx + 1,
        }
    return challenge_node


def make_respond_node(agents: AgentBundle):
    def respond_node(state: GraphState) -> GraphState:
        outcome: ChallengeOutcome | None = state.get("pending_outcome")
        if outcome is None or outcome.verdict != ChallengeVerdict.QUESTION:
            return {}
        bb: Blackboard = state["blackboard"]
        prop_id = state["current_prop_id"]
        prop = bb.get(prop_id)
        history = bb.proposition_history(prop_id)
        result = agents.sub.respond(
            prop,
            state["paper"],
            bb,
            state["context"],
            outcome.question,
            history[:-1],  # exclude the just-logged record
        )
        for citation in result.additional_citations:
            prop.citations.append(citation)
        bb.resolve_question(prop_id, outcome.question)
        return {"blackboard": bb, "pending_answer": result.answer}
    return respond_node


def make_judge_node(agents: AgentBundle):
    def judge_node(state: GraphState) -> GraphState:
        bb: Blackboard = state["blackboard"]
        prop_id = state["current_prop_id"]
        if prop_id is None:
            return {}
        prop = bb.get(prop_id)
        outcome: ChallengeOutcome | None = state.get("pending_outcome")
        answer = state.get("pending_answer")

        if outcome is None:
            return {}

        history = bb.proposition_history(prop_id)
        latest = history[-1] if history else None
        challenger = agents.challengers[outcome.challenger]

        # Map the verdict.
        if outcome.verdict == ChallengeVerdict.NO_ISSUE:
            judged = "resolved"
            response_text = ""
            prop.consecutive_unbroken_challenges += 1
            prop.consecutive_unanswered = 0
        elif outcome.verdict == ChallengeVerdict.REFUTED:
            judged = "refuted"
            response_text = ""
            prop.consecutive_unbroken_challenges = 0
            prop.consecutive_unanswered += 1
        else:
            assert outcome.verdict == ChallengeVerdict.QUESTION
            response_text = answer or ""
            judgment = challenger.judge(prop, outcome, response_text, history[:-1])
            judged = judgment.verdict
            if judged == "resolved":
                prop.consecutive_unbroken_challenges += 1
                prop.consecutive_unanswered = 0
            elif judged == "refuted":
                prop.consecutive_unbroken_challenges = 0
                prop.consecutive_unanswered += 1
            else:  # unresolved
                prop.consecutive_unbroken_challenges = 0
                prop.consecutive_unanswered += 1

        if latest is not None:
            latest.verdict = judged
            latest.response = response_text or None

        prop.rounds_completed = state.get("current_round", prop.rounds_completed)

        # Decide proposition fate.
        new_status: PropositionStatus | None = None
        if judged == "refuted":
            new_status = PropositionStatus.REFUTED
        elif prop.consecutive_unanswered >= state["doubt_streak"]:
            new_status = PropositionStatus.DOUBTFUL
        elif prop.consecutive_unbroken_challenges >= state["soft_pass_streak"]:
            new_status = PropositionStatus.CONFIRMED
        elif prop.rounds_completed >= state["max_rounds"]:
            # Escalation: pick best-guess outcome.
            new_status = (
                PropositionStatus.CONFIRMED
                if prop.consecutive_unbroken_challenges > prop.consecutive_unanswered
                else PropositionStatus.DOUBTFUL
            )

        if new_status is not None:
            bb.set_status(prop_id, new_status)
        prop.touch()

        return {"blackboard": bb}
    return judge_node


def make_annotate_node():
    def annotate_node(state: GraphState) -> GraphState:
        bb: Blackboard = state["blackboard"]
        prop_id = state["current_prop_id"]
        if prop_id is None:
            return {}
        prop = bb.get(prop_id)
        if not prop.status.is_terminal:
            return {}
        annotation = _annotate(prop, bb, state)
        logger.info(
            "Annotation: %s confidence=%s rounds=%d",
            prop.proposition_id, annotation.confidence, annotation.rounds,
        )
        return {"annotations": [annotation]}
    return annotate_node


# ---------- helpers ----------


def _annotate(prop: Proposition, bb: Blackboard, state: GraphState) -> Annotation:
    confidence = _confidence_for(prop, state)
    history = bb.proposition_history(prop.proposition_id)
    summaries = [
        ChallengeSummary(
            challenger=r.challenger,
            question=r.question,
            response=r.response or "",
            verdict=r.verdict or "pending",
        )
        for r in history
    ]
    return Annotation(
        proposition_id=prop.proposition_id,
        block_id=prop.block_id,
        statement=prop.statement,
        confidence=confidence,
        derivation=prop.derivation,
        citations=list(prop.citations),
        challenge_history=summaries,
        rounds=prop.rounds_completed,
        notes=_notes_for(prop, state),
    )


def _confidence_for(prop: Proposition, state: GraphState) -> ConfidenceLevel:
    if prop.status == PropositionStatus.REFUTED:
        return ConfidenceLevel.REFUTED
    if prop.status == PropositionStatus.DOUBTFUL:
        return ConfidenceLevel.DOUBTFUL
    if prop.status == PropositionStatus.CONFIRMED:
        strong = (
            prop.consecutive_unbroken_challenges >= state["soft_pass_streak"]
            and prop.rounds_completed >= 3
        )
        return ConfidenceLevel.STRONG if strong else ConfidenceLevel.WEAK
    # ESCALATED falls through here; treat like doubtful by default.
    return ConfidenceLevel.DOUBTFUL


def _notes_for(prop: Proposition, state: GraphState) -> str | None:
    if prop.rounds_completed >= state["max_rounds"]:
        return "max-rounds reached; verdict by streak comparison"
    return None


# ---------- conditional edges ----------


def schedule_router(state: GraphState) -> str:
    return "end" if state.get("finished") else "derive"


def challenge_router(state: GraphState) -> str:
    """After a challenger fires, decide whether sub-agent needs to respond."""
    outcome: ChallengeOutcome | None = state.get("pending_outcome")
    if outcome is None:
        return "judge"
    if outcome.verdict == ChallengeVerdict.QUESTION:
        return "respond"
    return "judge"


def judge_router(state: GraphState) -> str:
    """After judging, either annotate (terminal) or run another challenge round."""
    bb: Blackboard = state["blackboard"]
    prop_id = state.get("current_prop_id")
    if prop_id is None:
        return "annotate"
    prop = bb.get(prop_id)
    if prop.status.is_terminal:
        return "annotate"
    if prop.rounds_completed >= state["max_rounds"]:
        return "annotate"
    return "challenge"
