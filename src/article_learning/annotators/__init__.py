"""Output annotators. Currently JSON-only; MCP/PDF annotator is the planned extension."""

from article_learning.annotators.base import Annotator
from article_learning.annotators.json_annotator import JSONFileAnnotator, JSONLAnnotator

__all__ = ["Annotator", "JSONFileAnnotator", "JSONLAnnotator"]
