"""Parsed paper representation: semantic blocks rather than physical paragraphs."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SemanticBlockType(str, Enum):
    """Semantic role of a block. Physical paragraphs are merged/split into these."""

    BACKGROUND = "background"
    ASSUMPTION = "assumption"
    DEFINITION = "definition"
    LEMMA = "lemma"
    THEOREM = "theorem"
    PROOF = "proof"
    EXPERIMENT = "experiment"
    DISCUSSION = "discussion"
    OTHER = "other"


class SemanticBlock(BaseModel):
    """A semantically coherent chunk of a paper.

    Physical paragraph boundaries are intentionally NOT used because proofs
    routinely span multiple paragraphs; cutting them mid-derivation breaks
    context for the sub-agent.
    """

    block_id: str
    block_type: SemanticBlockType
    title: str | None = None
    text: str
    page_range: tuple[int, int] | None = None
    char_offset: tuple[int, int] | None = Field(
        default=None,
        description="(start, end) char offsets in the canonical markdown for citation grounding.",
    )
    references: list[str] = Field(
        default_factory=list,
        description="Bibliography keys cited in this block.",
    )

    model_config = {"frozen": False}


class Paper(BaseModel):
    """A paper after parsing & semantic segmentation."""

    title: str
    raw_markdown: str = ""
    blocks: list[SemanticBlock]
    bibliography: dict[str, str] = Field(
        default_factory=dict,
        description="Map from cite-key to formatted reference text.",
    )

    def block_by_id(self, block_id: str) -> SemanticBlock:
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        raise KeyError(f"Unknown block: {block_id}")
