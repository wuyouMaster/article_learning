"""Output annotation produced once a proposition exits the adversarial loop."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from article_learning.models.proposition import SourceCitation


class ConfidenceLevel(str, Enum):
    """User-facing confidence grade for the annotation.

    STRONG  - multiple challenge rounds resolved successfully (>= soft_pass_streak).
    WEAK    - some challenges resolved but not all challenger types ran clean.
    DOUBTFUL- A group failed to respond on K consecutive rounds.
    REFUTED - B group constructed a counterexample / found a logical hole.
    """

    STRONG = "strong"
    WEAK = "weak"
    DOUBTFUL = "doubtful"
    REFUTED = "refuted"

    @property
    def emoji(self) -> str:
        return {
            ConfidenceLevel.STRONG: "[OK]",
            ConfidenceLevel.WEAK: "[WARN]",
            ConfidenceLevel.DOUBTFUL: "[?]",
            ConfidenceLevel.REFUTED: "[X]",
        }[self]


class ChallengeSummary(BaseModel):
    """One row in the adversarial trail accompanying an annotation."""

    challenger: str
    question: str
    response: str
    verdict: str = Field(description="resolved | unresolved | refuted")


class Annotation(BaseModel):
    """The finished verdict on a proposition; what gets written into the PDF."""

    proposition_id: str
    block_id: str
    statement: str
    confidence: ConfidenceLevel
    derivation: str | None = None
    citations: list[SourceCitation] = Field(default_factory=list)
    challenge_history: list[ChallengeSummary] = Field(default_factory=list)
    rounds: int = 0
    notes: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
