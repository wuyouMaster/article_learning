"""Main agent for Group A.

Responsibilities (per the spec):
    1. Read every semantic block, extract candidate propositions, and build
       a dependency DAG (parent -> child means parent must be verified first).
    2. Maintain the global symbol table.
    3. Schedule the next proposition for sub-agents to work on, accounting
       for cycles (mutually-referential lemmas need joint verification).
    4. Adjudicate when a proposition is escalated.

Design note: this agent never *writes the proof*; it plans. SubAgents do
the actual derivation. Splitting these roles avoids the classic failure
mode where one giant prompt tries to do scheduling AND proving at once.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from article_learning.agents.base import AgentContext, BaseAgent
from article_learning.llm.base import LLMClient, parse_json_response
from article_learning.models.blackboard import Blackboard
from article_learning.models.paper import Paper
from article_learning.models.proposition import (
    Proposition,
    PropositionStatus,
    PropositionType,
    SourceCitation,
)
from article_learning.models.symbols import Symbol

logger = logging.getLogger(__name__)


class _PropDraft(BaseModel):
    proposition_id: str
    type: PropositionType
    statement: str
    formal_statement: str | None = None
    block_id: str
    citation_quote: str
    depends_on: list[str] = Field(default_factory=list)


class _SymbolDraft(BaseModel):
    name: str
    description: str
    introduced_in_block: str
    scope_blocks: list[str] = Field(default_factory=list)


class _PlanResponse(BaseModel):
    propositions: list[_PropDraft]
    symbols: list[_SymbolDraft] = Field(default_factory=list)


_SYSTEM_PROMPT = """You are the *Main Agent* coordinating Group A in an adversarial
paper-verification framework.

Your job is to read the semantic blocks of a paper and produce a plan:
  1. Extract every formally-stated *proposition* (definition / assumption /
     lemma / theorem / corollary / experimental claim / general claim).
  2. For each proposition give a stable id like "P1", "P2", ...
  3. Identify dependencies: proposition X "depends_on" Y when verifying X
     requires Y to already be established.
  4. Anchor each proposition to its source via a verbatim quote from the
     block (used later by the citation challenger). The quote MUST be a
     contiguous substring of the block text.
  5. List mathematical/typographic symbols that get introduced and whether
     they are global or scoped to specific block ids.

Return a single JSON object that matches this schema exactly:

{
  "propositions": [
    {
      "proposition_id": "P1",
      "type": "theorem | lemma | definition | assumption | corollary | claim | experimental_result",
      "statement": "natural-language statement",
      "formal_statement": "optional symbolic restatement, or null",
      "block_id": "block-N",
      "citation_quote": "exact substring of the block",
      "depends_on": ["P0", ...]
    }
  ],
  "symbols": [
    {
      "name": "X",
      "description": "what it stands for",
      "introduced_in_block": "block-N",
      "scope_blocks": ["block-N", "block-M"]
    }
  ]
}

Do NOT invent dependencies that the paper doesn't state. When unsure,
omit the edge — false dependencies poison downstream scheduling.
"""


class MainAgent(BaseAgent):
    name = "main"

    def __init__(self, llm: LLMClient) -> None:
        super().__init__(llm)

    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def plan(self, paper: Paper, context: AgentContext) -> Blackboard:
        """Build the initial blackboard: propositions + DAG + symbols."""
        user = self._render_paper(paper, context)
        messages = self._wrap(self.system_prompt(), user)
        response = self.llm.complete(messages, temperature=0.0, json_mode=True)
        plan = parse_json_response(response, _PlanResponse)
        return self._materialize(plan, paper)

    # ------------------------------------------------------------------

    def _render_paper(self, paper: Paper, context: AgentContext) -> str:
        lines: list[str] = [
            f"Paper title: {context.paper_title}",
            "Semantic blocks:",
        ]
        for block in paper.blocks:
            lines.append(
                f"\n--- {block.block_id} | {block.block_type.value}"
                f"{' | ' + block.title if block.title else ''} ---"
            )
            lines.append(block.text)
        return "\n".join(lines)

    def _materialize(self, plan: _PlanResponse, paper: Paper) -> Blackboard:
        bb = Blackboard()
        block_ids = {b.block_id for b in paper.blocks}
        block_text = {b.block_id: b.text for b in paper.blocks}

        # First pass: add propositions (without depends_on yet, so ordering
        # of the LLM's output doesn't break us).
        prop_ids: list[str] = []
        for draft in plan.propositions:
            if draft.block_id not in block_ids:
                logger.warning(
                    "Skipping proposition %s with unknown block %s",
                    draft.proposition_id,
                    draft.block_id,
                )
                continue
            citation = SourceCitation(
                block_id=draft.block_id,
                quote=draft.citation_quote.strip(),
            )
            prop = Proposition(
                proposition_id=draft.proposition_id,
                type=draft.type,
                statement=draft.statement,
                formal_statement=draft.formal_statement,
                block_id=draft.block_id,
                citations=[citation],
            )
            bb.add_proposition(prop)
            prop_ids.append(draft.proposition_id)

        # Second pass: dependencies (only those pointing to known props).
        for draft in plan.propositions:
            if draft.proposition_id not in bb.propositions:
                continue
            for parent in draft.depends_on:
                if parent in bb.propositions and parent != draft.proposition_id:
                    bb.add_dependency(parent, draft.proposition_id)

        # Symbols.
        for sym in plan.symbols:
            bb.symbol_table.add(
                Symbol(
                    name=sym.name,
                    description=sym.description,
                    introduced_in_block=sym.introduced_in_block,
                    scope_blocks=sym.scope_blocks,
                )
            )

        # Sanity-check citations against block text; warn on mismatches.
        for prop in bb.all_propositions():
            for cit in prop.citations:
                src = block_text.get(cit.block_id, "")
                if cit.quote and cit.quote not in src:
                    logger.warning(
                        "Citation quote not found verbatim in %s for %s; "
                        "B-group citation challenger may flag this.",
                        cit.block_id,
                        prop.proposition_id,
                    )
        return bb

    # ------------------------------------------------------------------

    @staticmethod
    def select_next(bb: Blackboard) -> str | None:
        """Pick the next proposition to work on.

        Strategy:
            1. Prefer propositions whose dependencies are all CONFIRMED.
            2. Among those, follow topological order.
            3. Cycle members are returned only after every acyclic
               proposition is terminal (joint verification phase).
        """
        order = bb.topological_order()
        cyclic = {n for cyc in bb.cycles() for n in cyc}

        for prop_id in order:
            prop = bb.get(prop_id)
            if prop.status.is_terminal:
                continue
            if prop.status == PropositionStatus.IN_PROGRESS:
                # Already being worked on, skip.
                continue
            if prop_id in cyclic:
                # Defer cycles until non-cyclic part is done.
                continue
            if not _deps_confirmed(bb, prop):
                continue
            return prop_id

        # Fall back to cycle members if nothing acyclic remains.
        for prop_id in order:
            prop = bb.get(prop_id)
            if prop.status.is_terminal:
                continue
            return prop_id
        return None


def _deps_confirmed(bb: Blackboard, prop: Proposition) -> bool:
    for dep_id in prop.depends_on:
        dep = bb.propositions.get(dep_id)
        if dep is None:
            continue
        if dep.status != PropositionStatus.CONFIRMED:
            return False
    return True
