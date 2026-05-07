"""Shared scaffolding for B-group challengers.

Every challenger emits a structured "outcome" capturing the question it
asks; the orchestrator then routes that question to the responsible
sub-agent and feeds the answer back for evaluation.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from article_learning.agents.base import AgentContext, BaseAgent
from article_learning.llm.base import parse_json_response
from article_learning.models.blackboard import Blackboard, ChallengeRecord
from article_learning.models.paper import Paper
from article_learning.models.proposition import Proposition


class ChallengeVerdict(str, Enum):
    NO_ISSUE = "no_issue"
    """The challenger could not find anything wrong this round (a 'pass')."""

    QUESTION = "question"
    """The challenger raises a question that A-group must answer."""

    REFUTED = "refuted"
    """The challenger constructed a clear counterexample / fatal hole."""


class ChallengeOutcome(BaseModel):
    challenger: str
    verdict: ChallengeVerdict
    question: str = ""
    rationale: str = Field(default="", description="Why this question/refutation matters.")


class _RawChallenge(BaseModel):
    verdict: ChallengeVerdict
    question: str = ""
    rationale: str = ""


class _RawJudgment(BaseModel):
    verdict: str = Field(description="resolved | unresolved | refuted")
    rationale: str = ""


_JUDGMENT_PROMPT = """You are evaluating whether the sub-agent's answer adequately
resolves your earlier challenge.

Return JSON:
{
  "verdict": "resolved" | "unresolved" | "refuted",
  "rationale": "brief explanation"
}

Rules:
  - "resolved" means the answer addresses your question with grounding in the paper.
  - "unresolved" means the answer dodges, hand-waves, or contradicts itself.
  - "refuted" means the answer or its derivation contains a definitive flaw
    you can now point to.
"""


class BaseChallenger(BaseAgent):
    """Common challenger logic: ask a question, then judge a response."""

    challenger_kind: str = "base"

    def system_prompt(self) -> str:  # pragma: no cover - subclasses override
        raise NotImplementedError

    def challenge(
        self,
        prop: Proposition,
        paper: Paper,
        bb: Blackboard,
        context: AgentContext,
        history: list[ChallengeRecord],
    ) -> ChallengeOutcome:
        user = self._render_challenge_user(prop, paper, bb, context, history)
        messages = self._wrap(self.system_prompt(), user)
        response = self.llm.complete(messages, temperature=0.2, json_mode=True)
        raw = parse_json_response(response, _RawChallenge)
        return ChallengeOutcome(
            challenger=self.challenger_kind,
            verdict=raw.verdict,
            question=raw.question,
            rationale=raw.rationale,
        )

    def judge(
        self,
        prop: Proposition,
        outcome: ChallengeOutcome,
        answer: str,
        history: list[ChallengeRecord],
    ) -> _RawJudgment:
        if outcome.verdict != ChallengeVerdict.QUESTION:
            return _RawJudgment(verdict=outcome.verdict.value, rationale=outcome.rationale)
        user = (
            f"Proposition {prop.proposition_id}: {prop.statement}\n\n"
            f"Your earlier challenge:\n  {outcome.question}\n"
            f"  rationale: {outcome.rationale}\n\n"
            f"Sub-agent's answer:\n  {answer}\n\n"
            f"History so far:\n"
            + (
                "\n".join(
                    f"  R{r.round_index} [{r.challenger}] {r.verdict}: {r.question}"
                    for r in history
                )
                or "  (none)"
            )
        )
        messages = self._wrap(_JUDGMENT_PROMPT, user)
        response = self.llm.complete(messages, temperature=0.0, json_mode=True)
        return parse_json_response(response, _RawJudgment)

    # ------------------------------------------------------------------

    def _render_challenge_user(
        self,
        prop: Proposition,
        paper: Paper,
        bb: Blackboard,
        context: AgentContext,
        history: list[ChallengeRecord],
    ) -> str:
        block = paper.block_by_id(prop.block_id)
        prior = (
            "\n".join(
                f"  R{r.round_index} [{r.challenger}] Q: {r.question}\n"
                f"        A: {r.response or '(no answer)'}\n"
                f"        verdict={r.verdict}"
                for r in history
            )
            or "  (none yet)"
        )
        deps = (
            "\n".join(
                f"  - {d} [{bb.propositions[d].status.value}]: {bb.propositions[d].statement}"
                for d in prop.depends_on
                if d in bb.propositions
            )
            or "  (none)"
        )
        return (
            f"Paper: {context.paper_title}\n\n"
            f"Target proposition {prop.proposition_id}: {prop.statement}\n"
            f"Sub-agent's derivation:\n{prop.derivation or '(none)'}\n\n"
            f"Source block ({block.block_id}):\n"
            f"\"\"\"\n{block.text}\n\"\"\"\n\n"
            f"Dependencies:\n{deps}\n\n"
            f"Prior challenge rounds:\n{prior}\n"
        )
