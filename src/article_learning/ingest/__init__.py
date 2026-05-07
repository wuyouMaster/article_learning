"""Paper ingestion: PDF -> markdown -> semantic blocks."""

from article_learning.ingest.parser import PaperLoader, load_paper
from article_learning.ingest.segmenter import SemanticSegmenter

__all__ = ["PaperLoader", "SemanticSegmenter", "load_paper"]
