"""Sub-agents in Group A. One sub-agent owns one proposition at a time.

A sub-agent does two things:
    * `derive(...)`: produce an initial proof / derivation grounded in the
      block's text and the current symbol table.
    * `respond(question, history)`: answer a B-group challenge.

Both calls return JSON so the orchestrator can carry the structured output
back onto the blackboard.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from article_learning.agents.base import AgentContext, BaseAgent
from article_learning.llm.base import LLMClient, parse_json_response
from article_learning.models.blackboard import Blackboard, ChallengeRecord
from article_learning.models.paper import Paper
from article_learning.models.proposition import Proposition, SourceCitation

_DERIVATION_PROMPT = """You are a *sub-agent* in Group A responsible for a single
proposition. Read the proposition statement, its source block, the propositions
it depends on (already verified), and the global symbol table. Then produce a
careful derivation.

Rules:
  - Quote from the block when stating premises; do not invent assumptions.
  - Reference dependent propositions by their id (e.g. "By P2, ...").
  - If the paper already gives a proof, follow it; do not invent a new one.
  - If you cannot complete the derivation, say so explicitly in `notes`.

Return JSON:
{
  "derivation": "step-by-step argument as a single string",
  "extra_citations": [
    {"block_id": "block-N", "quote": "exact substring"}
  ],
  "notes": "optional caveat or null"
}
"""


_RESPONSE_PROMPT = """You are a *sub-agent* in Group A defending a proposition
against an adversarial challenger from Group B.

Inputs you'll see:
  * The proposition statement and your prior derivation.
  * The challenger's question (from one of: logic, assumption, counterexample, citation).
  * Prior challenge/response rounds for context.

Rules:
  - Address the challenge directly. Do not change the subject.
  - Cite block text or dependent propositions when justifying.
  - If you cannot answer, return `verdict="cannot_answer"` and explain why.
  - If the challenge reveals a real flaw, concede honestly with `verdict="concede"`.

Return JSON:
{
  "answer": "your reply, addressing the challenger's point",
  "additional_citations": [
    {"block_id": "block-N", "quote": "exact substring"}
  ],
  "verdict": "answered | cannot_answer | concede"
}
"""


class _DerivationResponse(BaseModel):
    derivation: str
    extra_citations: list[SourceCitation] = Field(default_factory=list)
    notes: str | None = None


class _ChallengeResponse(BaseModel):
    answer: str
    additional_citations: list[SourceCitation] = Field(default_factory=list)
    verdict: str = Field(default="answered")


class SubAgent(BaseAgent):
    """Per-block prover. The same instance can handle several propositions
    sequentially (the blackboard tracks ownership)."""

    name = "sub"

    def __init__(self, llm: LLMClient, *, agent_id: str = "sub-default") -> None:
        super().__init__(llm)
        self.agent_id = agent_id

    def system_prompt(self) -> str:
        return _DERIVATION_PROMPT

    def derive(
        self,
        prop: Proposition,
        paper: Paper,
        bb: Blackboard,
        context: AgentContext,
    ) -> _DerivationResponse:
        user = _render_derivation_user(prop, paper, bb, context)
        messages = self._wrap(_DERIVATION_PROMPT, user)
        response = self.llm.complete(messages, temperature=0.0, json_mode=True)
        return parse_json_response(response, _DerivationResponse)

    def respond(
        self,
        prop: Proposition,
        paper: Paper,
        bb: Blackboard,
        context: AgentContext,
        question: str,
        history: list[ChallengeRecord],
    ) -> _ChallengeResponse:
        user = _render_response_user(prop, paper, bb, context, question, history)
        messages = self._wrap(_RESPONSE_PROMPT, user)
        response = self.llm.complete(messages, temperature=0.0, json_mode=True)
        return parse_json_response(response, _ChallengeResponse)


# ---------- prompt rendering helpers ----------


def _render_derivation_user(
    prop: Proposition,
    paper: Paper,
    bb: Blackboard,
    context: AgentContext,
) -> str:
    block = paper.block_by_id(prop.block_id)
    deps = "\n".join(_render_dep(bb, dep) for dep in prop.depends_on) or "  (none)"
    syms = _render_symbol_table(bb)
    return (
        f"Paper: {context.paper_title}\n\n"
        f"Proposition {prop.proposition_id} ({prop.type.value}):\n"
        f"  {prop.statement}\n"
        f"Formal: {prop.formal_statement or '(n/a)'}\n\n"
        f"Source block ({block.block_id}, {block.block_type.value}):\n"
        f"\"\"\"\n{block.text}\n\"\"\"\n\n"
        f"Dependencies (already verified):\n{deps}\n\n"
        f"Symbol table in scope:\n{syms}\n"
    )


def _render_response_user(
    prop: Proposition,
    paper: Paper,
    bb: Blackboard,
    context: AgentContext,
    question: str,
    history: list[ChallengeRecord],
) -> str:
    block = paper.block_by_id(prop.block_id)
    history_str = (
        "\n".join(
            f"  R{r.round_index} [{r.challenger}] Q: {r.question}\n"
            f"        A: {r.response or '(no answer)'}\n"
            f"        verdict={r.verdict}"
            for r in history
        )
        or "  (this is the first round)"
    )
    return (
        f"Paper: {context.paper_title}\n\n"
        f"Proposition {prop.proposition_id}: {prop.statement}\n"
        f"Your prior derivation:\n{prop.derivation or '(none yet)'}\n\n"
        f"Source block ({block.block_id}):\n"
        f"\"\"\"\n{block.text}\n\"\"\"\n\n"
        f"Prior rounds:\n{history_str}\n\n"
        f"Current challenger question:\n{question}\n"
    )


def _render_dep(bb: Blackboard, dep_id: str) -> str:
    prop = bb.propositions.get(dep_id)
    if prop is None:
        return f"  - {dep_id}: (unknown)"
    return f"  - {dep_id} [{prop.status.value}]: {prop.statement}"


def _render_symbol_table(bb: Blackboard) -> str:
    syms = bb.symbol_table.all_symbols()
    if not syms:
        return "  (empty)"
    return "\n".join(f"  - {s.name}: {s.description}" for s in syms)
