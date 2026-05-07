"""Group A: main agent (planner/scheduler) + sub-agents (per-block provers)."""

from article_learning.agents.group_a.main_agent import MainAgent
from article_learning.agents.group_a.sub_agent import SubAgent

__all__ = ["MainAgent", "SubAgent"]
