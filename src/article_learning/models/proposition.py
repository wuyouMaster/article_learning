"""Propositions: the atomic units the adversarial loop validates."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class PropositionType(str, Enum):
    DEFINITION = "definition"
    ASSUMPTION = "assumption"
    LEMMA = "lemma"
    THEOREM = "theorem"
    COROLLARY = "corollary"
    CLAIM = "claim"
    EXPERIMENTAL_RESULT = "experimental_result"


class PropositionStatus(str, Enum):
    """State machine values held on the blackboard.

    Transitions:
        PENDING -> IN_PROGRESS              (sub-agent picked it up)
        IN_PROGRESS -> UNDER_CHALLENGE      (B group is questioning)
        UNDER_CHALLENGE -> CONFIRMED        (challenges resolved, strong/soft pass)
        UNDER_CHALLENGE -> REFUTED          (B group constructed a counterexample)
        UNDER_CHALLENGE -> DOUBTFUL         (A group failed to respond on K rounds)
        any -> ESCALATED                    (max rounds hit, main agent must decide)
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    UNDER_CHALLENGE = "under_challenge"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    DOUBTFUL = "doubtful"
    ESCALATED = "escalated"

    @property
    def is_terminal(self) -> bool:
        return self in (
            PropositionStatus.CONFIRMED,
            PropositionStatus.REFUTED,
            PropositionStatus.DOUBTFUL,
        )


class SourceCitation(BaseModel):
    """Pointer back into the original paper. Required on every proposition.

    Forces sub-agents to ground claims in the source so B-group's citation
    checker can verify against raw text (mitigates hallucination propagation).
    """

    block_id: str
    quote: str = Field(description="Verbatim snippet from the block backing this claim.")
    char_offset: tuple[int, int] | None = None

    def __str__(self) -> str:
        return f"[{self.block_id}] {self.quote[:80]}..."


class Proposition(BaseModel):
    """A single claim/lemma/theorem/etc. tracked through the adversarial loop."""

    proposition_id: str
    type: PropositionType
    statement: str = Field(description="Natural-language statement of the claim.")
    formal_statement: str | None = Field(
        default=None,
        description="Optional formal/symbolic restatement (LaTeX-ish).",
    )
    block_id: str = Field(description="Semantic block this proposition was extracted from.")

    citations: list[SourceCitation] = Field(default_factory=list)
    depends_on: list[str] = Field(
        default_factory=list,
        description="Other proposition_ids this claim depends on (DAG edge).",
    )

    derivation: str | None = Field(
        default=None,
        description="The sub-agent's derivation/proof sketch for this proposition.",
    )

    status: PropositionStatus = PropositionStatus.PENDING
    owner_agent: str | None = Field(
        default=None,
        description="Identifier of the sub-agent currently responsible for this proposition.",
    )

    rounds_completed: int = 0
    consecutive_unbroken_challenges: int = Field(
        default=0,
        description="Number of consecutive B-group rounds that failed to break this proposition.",
    )
    consecutive_unanswered: int = Field(
        default=0,
        description="Number of consecutive rounds A-group failed to respond to a challenge.",
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def touch(self) -> None:
        self.updated_at = datetime.utcnow()
