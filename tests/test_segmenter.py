"""Segmenter unit tests."""

from __future__ import annotations

from article_learning.ingest.segmenter import SemanticSegmenter
from article_learning.models.paper import SemanticBlockType


def test_segments_theorem_lemma_proof_definition(sample_markdown: str):
    blocks = SemanticSegmenter().segment(sample_markdown)
    types = {b.block_type for b in blocks}
    assert SemanticBlockType.LEMMA in types or SemanticBlockType.THEOREM in types
    assert SemanticBlockType.ASSUMPTION in types


def test_proof_block_swallows_until_qed(sample_markdown: str):
    blocks = SemanticSegmenter().segment(sample_markdown)
    proof_blocks = [b for b in blocks if b.block_type == SemanticBlockType.PROOF]
    # Each proof in the fixture ends with \blacksquare so should self-terminate.
    for b in proof_blocks:
        assert "blacksquare" in b.text or "QED" in b.text.upper()


def test_returns_single_block_when_no_markers():
    blocks = SemanticSegmenter().segment("Just a single paragraph of text without structure.")
    assert len(blocks) == 1
    assert blocks[0].block_type == SemanticBlockType.OTHER


def test_block_ids_are_sequential(sample_markdown: str):
    blocks = SemanticSegmenter().segment(sample_markdown)
    for idx, block in enumerate(blocks):
        assert block.block_id == f"block-{idx}"


def test_char_offsets_make_sense(sample_markdown: str):
    blocks = SemanticSegmenter().segment(sample_markdown)
    for block in blocks:
        assert block.char_offset is not None
        start, end = block.char_offset
        assert 0 <= start < end <= len(sample_markdown)
