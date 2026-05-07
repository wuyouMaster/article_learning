"""Run the full pipeline against the sample paper using the mock LLM.

This is the same code path the test suite exercises, exposed as a script
so you can eyeball the streamed annotations without setting up an API key.

Usage:
    python examples/run_with_mock.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from article_learning.annotators.json_annotator import StreamAnnotator
from article_learning.ingest.parser import PaperLoader
from article_learning.orchestrator import Orchestrator

# Reuse the deterministic mock from the test fixture so the example is
# self-contained.
sys.path.insert(0, str(ROOT / "tests"))
from conftest import _is_derivation, _is_judgment, _last_user, _proposition_id_from_user  # noqa: E402
from collections import defaultdict  # noqa: E402

from article_learning.llm.mock import DeterministicMockLLM  # noqa: E402


def build_mock() -> DeterministicMockLLM:
    counters: dict[str, int] = defaultdict(int)

    def main_handler(_messages):
        return json.dumps(
            {
                "propositions": [
                    {
                        "proposition_id": "P1",
                        "type": "assumption",
                        "statement": "f is continuous on [0,1]",
                        "formal_statement": "f \\in C([0,1])",
                        "block_id": "block-2",
                        "citation_quote": "we assume that $f$ is continuous on $[0, 1]$",
                        "depends_on": [],
                    },
                    {
                        "proposition_id": "P2",
                        "type": "lemma",
                        "statement": "If f is continuous on [0,1] then f is bounded.",
                        "formal_statement": None,
                        "block_id": "block-3",
                        "citation_quote": (
                            "If $f$ is continuous on $[0, 1]$, "
                            "then $f$ is bounded on $[0, 1]$."
                        ),
                        "depends_on": ["P1"],
                    },
                    {
                        "proposition_id": "P3",
                        "type": "theorem",
                        "statement": (
                            "If f is continuous on [0,1] then the integral of f exists."
                        ),
                        "formal_statement": None,
                        "block_id": "block-4",
                        "citation_quote": (
                            "If $f$ is continuous on $[0, 1]$, "
                            "then $\\int_0^1 f(x)\\,dx$ exists."
                        ),
                        "depends_on": ["P1", "P2"],
                    },
                ],
                "symbols": [
                    {
                        "name": "f",
                        "description": "Real-valued continuous function on [0,1]",
                        "introduced_in_block": "block-2",
                        "scope_blocks": [],
                    },
                ],
            }
        )

    def sub_handler(messages):
        prop_id = _proposition_id_from_user(_last_user(messages))
        if _is_derivation(messages):
            return json.dumps(
                {
                    "derivation": (
                        f"Derivation for {prop_id}: by the source block this follows "
                        "from prior dependencies via standard real-analysis tools."
                    ),
                    "extra_citations": [],
                    "notes": None,
                }
            )
        return json.dumps(
            {
                "answer": (
                    f"For {prop_id}, the question is addressed by the cited block "
                    "and the dependency chain."
                ),
                "additional_citations": [],
                "verdict": "answered",
            }
        )

    def make_challenger_handler(kind: str):
        def handler(messages):
            key = f"{kind}:{_proposition_id_from_user(_last_user(messages))}"
            counters[key] += 1
            n = counters[key]
            if _is_judgment(messages):
                return json.dumps({"verdict": "resolved", "rationale": "covered"})
            if n % 2 == 1:
                return json.dumps(
                    {
                        "verdict": "question",
                        "question": f"[{kind}] R{n}: justify the key step",
                        "rationale": f"synthetic {kind} probe {n}",
                    }
                )
            return json.dumps(
                {"verdict": "no_issue", "question": "", "rationale": f"{kind} pass {n}"}
            )

        return handler

    mock = DeterministicMockLLM()
    mock.register("main", main_handler)
    mock.register("sub", sub_handler)
    mock.register("logic", make_challenger_handler("logic"))
    mock.register("assumption", make_challenger_handler("assumption"))
    mock.register("counterexample", make_challenger_handler("counterexample"))
    mock.register("citation", make_challenger_handler("citation"))
    return mock


def main() -> None:
    sample = ROOT / "tests" / "fixtures" / "sample_paper.md"
    paper = PaperLoader().from_text_file(sample)
    print(f"Parsed {len(paper.blocks)} semantic blocks from '{paper.title}'\n")

    final = Orchestrator(build_mock()).run(
        paper, annotators=[StreamAnnotator(sys.stdout)]
    )

    print("\n--- summary ---")
    for ann in final["annotations"]:
        print(
            f"  {ann.proposition_id} [{ann.confidence.value}] "
            f"rounds={ann.rounds} block={ann.block_id}"
        )


if __name__ == "__main__":
    main()
