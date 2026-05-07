"""Paper loaders.

`marker-pdf` is treated as a soft dependency: the framework runs perfectly
on plain markdown/text input (the common case for tests). Only PDF input
requires the `pdf` extra.
"""

from __future__ import annotations

import logging
from pathlib import Path

from article_learning.ingest.segmenter import SemanticSegmenter
from article_learning.models.paper import Paper

logger = logging.getLogger(__name__)


class PaperLoader:
    """Front-door for getting a `Paper` regardless of source format."""

    def __init__(self, segmenter: SemanticSegmenter | None = None) -> None:
        self.segmenter = segmenter or SemanticSegmenter()

    def from_markdown(self, markdown: str, *, title: str | None = None) -> Paper:
        blocks = self.segmenter.segment(markdown)
        return Paper(
            title=title or self._extract_title(markdown),
            raw_markdown=markdown,
            blocks=blocks,
        )

    def from_text_file(self, path: str | Path) -> Paper:
        text = Path(path).read_text(encoding="utf-8")
        return self.from_markdown(text, title=Path(path).stem)

    def from_pdf(self, path: str | Path) -> Paper:
        markdown = self._pdf_to_markdown(Path(path))
        return self.from_markdown(markdown, title=Path(path).stem)

    @staticmethod
    def _extract_title(markdown: str) -> str:
        for line in markdown.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
            if stripped:
                return stripped[:120]
        return "Untitled"

    @staticmethod
    def _pdf_to_markdown(path: Path) -> str:
        try:
            from marker.convert import convert_single_pdf
            from marker.models import load_all_models
        except ImportError as exc:  # pragma: no cover - exercised only with PDF input
            raise ImportError(
                "PDF input requires the 'pdf' extra. Install with:\n"
                "    pip install 'article-learning[pdf]'"
            ) from exc

        logger.info("Loading marker models (first call may take a while)")
        models = load_all_models()
        full_text, _images, _meta = convert_single_pdf(str(path), models)
        return full_text


def load_paper(source: str | Path, *, title: str | None = None) -> Paper:
    """Convenience helper. Detects PDF by extension."""
    path = Path(source)
    loader = PaperLoader()
    if path.suffix.lower() == ".pdf":
        return loader.from_pdf(path)
    if path.exists():
        return loader.from_text_file(path)
    return loader.from_markdown(str(source), title=title)
