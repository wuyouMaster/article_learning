"""Group B: structured adversarial challengers (not random)."""

from article_learning.agents.group_b.assumption_challenger import AssumptionChallenger
from article_learning.agents.group_b.base_challenger import BaseChallenger, ChallengeOutcome
from article_learning.agents.group_b.citation_checker import CitationChecker
from article_learning.agents.group_b.counterexample_constructor import CounterexampleConstructor
from article_learning.agents.group_b.logic_challenger import LogicChallenger

__all__ = [
    "AssumptionChallenger",
    "BaseChallenger",
    "ChallengeOutcome",
    "CitationChecker",
    "CounterexampleConstructor",
    "LogicChallenger",
]
