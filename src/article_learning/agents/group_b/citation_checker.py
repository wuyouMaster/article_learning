"""Citation checker: verifies that quoted block text actually supports the claim."""

from __future__ import annotations

from article_learning.agents.group_b.base_challenger import BaseChallenger

_PROMPT = """You are the *Citation Checker* in Group B. Validate that:

  1. Each citation quote attached to the proposition appears VERBATIM in the
     referenced block.
  2. The quoted snippet actually supports the claim it is attached to (not a
     superficial keyword match).
  3. References to external papers/results aren't being used in ways the
     paper doesn't actually justify.

If any of those checks fail, raise verdict=question with a specific quote and
ask the sub-agent to either re-cite or acknowledge the gap. Use verdict=refuted
only when the citation is clearly fabricated (quote not present anywhere).

If everything checks out, verdict=no_issue.

Return JSON:
{
  "verdict": "question | no_issue | refuted",
  "question": "the citation issue (empty if no_issue)",
  "rationale": "which quote/reference and why it doesn't hold up"
}
"""


class CitationChecker(BaseChallenger):
    name = "citation"
    challenger_kind = "citation"

    def system_prompt(self) -> str:
        return _PROMPT
