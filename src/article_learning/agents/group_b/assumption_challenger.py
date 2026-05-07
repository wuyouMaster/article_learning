"""Assumption challenger: questions whether stated premises actually hold."""

from __future__ import annotations

from article_learning.agents.group_b.base_challenger import BaseChallenger

_PROMPT = """You are the *Assumption Challenger* in Group B. Find a premise that
the proposition or its derivation depends on, and question whether that premise
is actually established (not just asserted).

Examples of assumptions to attack:
  * "We assume f is convex." - is convexity actually verified for the f used?
  * Implicit independence assumptions in probability arguments.
  * "By the central limit theorem" - do the conditions actually hold here?
  * Type/dimensional compatibility ("X * Y" - shapes match?).

If every assumption is either explicitly granted by the paper or already in
the dependency graph as a verified proposition, set verdict=no_issue.

Return JSON:
{
  "verdict": "question | no_issue | refuted",
  "question": "the question (empty if no_issue)",
  "rationale": "which assumption and why it's suspicious"
}
"""


class AssumptionChallenger(BaseChallenger):
    name = "assumption"
    challenger_kind = "assumption"

    def system_prompt(self) -> str:
        return _PROMPT
