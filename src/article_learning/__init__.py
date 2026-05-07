"""Adversarial multi-agent framework for paper derivation and annotation."""

from article_learning.models.annotation import Annotation, ConfidenceLevel
from article_learning.models.blackboard import Blackboard
from article_learning.models.proposition import Proposition, PropositionStatus, PropositionType
from article_learning.orchestrator import Orchestrator, run_pipeline

__all__ = [
    "Annotation",
    "Blackboard",
    "ConfidenceLevel",
    "Orchestrator",
    "Proposition",
    "PropositionStatus",
    "PropositionType",
    "run_pipeline",
]

__version__ = "0.1.0"
