"""Semantic segmentation: markdown -> SemanticBlock[].

Rule-based first pass that respects mathematical paper conventions:

  - Recognises "Theorem N", "Lemma N", "Proposition N", "Definition N",
    "Assumption N", "Proof.", "Experiment", "Corollary N" markers.
  - A `Proof.` block keeps going until the next theorem-like marker or a
    `\\qed`/`Q.E.D.` token; this prevents the proof being chopped.
  - Markdown headings (`#`, `##`, ...) are used as fall-back boundaries.

This is intentionally lightweight; an LLM-based refiner can replace it later
without changing the downstream contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from article_learning.models.paper import SemanticBlock, SemanticBlockType

_THEOREM_RE = re.compile(
    r"^\s*(?:\*\*|__)?\s*"
    r"(?P<kind>Theorem|Lemma|Proposition|Corollary|Definition|Assumption|Experiment)"
    r"\s*(?P<num>\d+(?:\.\d+)*)?\.?\s*(?:\*\*|__)?\s*(?:[:\.]\s*(?P<title>.*))?$",
    re.IGNORECASE | re.MULTILINE,
)
_PROOF_START_RE = re.compile(r"^\s*(?:\*\*|__)?\s*Proof\s*\.?\s*(?:\*\*|__)?", re.IGNORECASE)
_PROOF_END_RE = re.compile(r"(\\qed|Q\.E\.D\.|\u220e|\u25a0)", re.IGNORECASE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class _Marker:
    line_start: int  # inclusive
    char_start: int
    kind: SemanticBlockType
    title: str | None
    raw_label: str


_KIND_MAP: dict[str, SemanticBlockType] = {
    "theorem": SemanticBlockType.THEOREM,
    "lemma": SemanticBlockType.LEMMA,
    "proposition": SemanticBlockType.THEOREM,
    "corollary": SemanticBlockType.THEOREM,
    "definition": SemanticBlockType.DEFINITION,
    "assumption": SemanticBlockType.ASSUMPTION,
    "experiment": SemanticBlockType.EXPERIMENT,
}

_HEADING_KIND_HINTS: list[tuple[re.Pattern[str], SemanticBlockType]] = [
    (
        re.compile(r"\b(introduction|background|related work|preliminari)", re.I),
        SemanticBlockType.BACKGROUND,
    ),
    (re.compile(r"\b(assumption)", re.I), SemanticBlockType.ASSUMPTION),
    (re.compile(r"\b(experiment|evaluation|result)", re.I), SemanticBlockType.EXPERIMENT),
    (re.compile(r"\b(discussion|conclusion|limitation)", re.I), SemanticBlockType.DISCUSSION),
    (re.compile(r"\b(definition)", re.I), SemanticBlockType.DEFINITION),
    (re.compile(r"\b(theorem|lemma|proof)", re.I), SemanticBlockType.THEOREM),
]


class SemanticSegmenter:
    """Markdown -> list[SemanticBlock] with proof boundaries preserved."""

    def segment(self, markdown: str) -> list[SemanticBlock]:
        markers = self._collect_markers(markdown)
        if not markers:
            return [
                SemanticBlock(
                    block_id="block-0",
                    block_type=SemanticBlockType.OTHER,
                    title=None,
                    text=markdown.strip(),
                    char_offset=(0, len(markdown)),
                )
            ]

        # Sort by char position (markers are emitted in scan order already).
        markers.sort(key=lambda m: m.char_start)

        blocks: list[SemanticBlock] = []
        for idx, marker in enumerate(markers):
            end = markers[idx + 1].char_start if idx + 1 < len(markers) else len(markdown)
            text = markdown[marker.char_start : end].strip()
            if not text:
                continue
            block_type = self._coerce_proof(marker, text)
            blocks.append(
                SemanticBlock(
                    block_id=f"block-{idx}",
                    block_type=block_type,
                    title=marker.title or marker.raw_label,
                    text=text,
                    char_offset=(marker.char_start, end),
                )
            )
        return self._merge_proofs(blocks)

    # ------------------------------------------------------------------

    def _collect_markers(self, markdown: str) -> list[_Marker]:
        markers: list[_Marker] = []

        for match in _THEOREM_RE.finditer(markdown):
            kind_word = match.group("kind").lower()
            block_type = _KIND_MAP.get(kind_word, SemanticBlockType.OTHER)
            title = match.group("title")
            label = f"{match.group('kind')} {match.group('num') or ''}".strip()
            markers.append(
                _Marker(
                    line_start=markdown[: match.start()].count("\n"),
                    char_start=match.start(),
                    kind=block_type,
                    title=title.strip() if title else None,
                    raw_label=label,
                )
            )

        for match in _PROOF_START_RE.finditer(markdown):
            markers.append(
                _Marker(
                    line_start=markdown[: match.start()].count("\n"),
                    char_start=match.start(),
                    kind=SemanticBlockType.PROOF,
                    title=None,
                    raw_label="Proof",
                )
            )

        # Headings as soft separators (only if not already covered by a marker).
        for match in _HEADING_RE.finditer(markdown):
            heading = match.group(2).strip()
            kind = SemanticBlockType.OTHER
            for pattern, candidate in _HEADING_KIND_HINTS:
                if pattern.search(heading):
                    kind = candidate
                    break
            markers.append(
                _Marker(
                    line_start=markdown[: match.start()].count("\n"),
                    char_start=match.start(),
                    kind=kind,
                    title=heading,
                    raw_label=heading,
                )
            )

        # Dedupe: if two markers fall on the exact same position, keep the
        # most-specific one (theorem > heading).
        markers.sort(key=lambda m: (m.char_start, 0 if m.kind != SemanticBlockType.OTHER else 1))
        deduped: list[_Marker] = []
        last_pos: int | None = None
        for m in markers:
            if last_pos is not None and m.char_start == last_pos:
                continue
            deduped.append(m)
            last_pos = m.char_start
        return deduped

    @staticmethod
    def _coerce_proof(marker: _Marker, text: str) -> SemanticBlockType:
        if marker.kind == SemanticBlockType.PROOF:
            return SemanticBlockType.PROOF
        return marker.kind

    @staticmethod
    def _merge_proofs(blocks: list[SemanticBlock]) -> list[SemanticBlock]:
        """If a proof block was cut prematurely (no QED), absorb the next sibling."""
        merged: list[SemanticBlock] = []
        i = 0
        while i < len(blocks):
            block = blocks[i]
            if block.block_type == SemanticBlockType.PROOF and not _PROOF_END_RE.search(block.text):
                # Walk forward until QED or another theorem-like block starts.
                j = i + 1
                while j < len(blocks):
                    nxt = blocks[j]
                    if nxt.block_type in (
                        SemanticBlockType.THEOREM,
                        SemanticBlockType.LEMMA,
                        SemanticBlockType.DEFINITION,
                        SemanticBlockType.ASSUMPTION,
                    ):
                        break
                    block = block.model_copy(
                        update={
                            "text": block.text + "\n\n" + nxt.text,
                            "char_offset": (
                                block.char_offset[0] if block.char_offset else None,
                                nxt.char_offset[1] if nxt.char_offset else None,
                            )
                            if block.char_offset and nxt.char_offset
                            else block.char_offset,
                        }
                    )
                    j += 1
                    if _PROOF_END_RE.search(nxt.text):
                        break
                merged.append(block)
                i = j
                continue
            merged.append(block)
            i += 1
        # Re-id sequentially after merging.
        return [b.model_copy(update={"block_id": f"block-{idx}"}) for idx, b in enumerate(merged)]
