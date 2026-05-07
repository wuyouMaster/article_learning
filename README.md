# article-learning

Adversarial multi-agent framework for **automatic paper derivation and
annotation**. Two agent groups argue over every claim; whatever survives
becomes a structured annotation.

[简体中文 README](README.zh-CN.md)

## Architecture

```
                         +-----------------------+
                         |     Blackboard        |  <- single source of truth
                         |  (state machine, DAG, |
                         |   symbol table, log)  |
                         +-----------------------+
                                  ^   ^
                                  |   |
        +-------------------------+   +---------------------------+
        |                                                         |
+-------------------+                                  +----------------------+
|     Group A       |                                  |       Group B        |
| MainAgent (DAG)   |                                  | LogicChallenger      |
| SubAgent  (block) |                                  | AssumptionChallenger |
+-------------------+                                  | CounterexampleConst. |
                                                       | CitationChecker      |
                                                       +----------------------+
                                                                  |
                                                                  v
                                                          streaming Annotator
                                                          (JSON now / MCP later)
```

### Group A

* **`MainAgent`** reads every semantic block, extracts propositions, builds
  a dependency DAG, maintains the global symbol table, and decides which
  proposition is next via topological order. Cycles (mutually-referential
  lemmas) are flagged for joint verification.
* **`SubAgent`** owns one proposition at a time. It produces a derivation
  grounded in the source block and answers Group B's questions.

### Group B (structured, not random)

| Challenger              | Mission                                                |
|-------------------------|--------------------------------------------------------|
| `LogicChallenger`       | Hunt for unjustified leaps in the derivation           |
| `AssumptionChallenger`  | Question whether the stated premises actually hold     |
| `CounterexampleConstr.` | Try to construct a concrete counterexample             |
| `CitationChecker`       | Verify quoted block text really supports the claim     |

The orchestrator rotates through these every round, so pressure is
diversified.

### State machine

```
PENDING -> IN_PROGRESS -> UNDER_CHALLENGE -+-> CONFIRMED
                                           +-> REFUTED
                                           +-> DOUBTFUL
                                           +-> ESCALATED
```

* `consecutive_unbroken_challenges >= soft_pass_streak` -> CONFIRMED
* `consecutive_unanswered >= doubt_streak` -> DOUBTFUL
* `rounds_completed >= max_rounds` without a streak -> ESCALATED

### Confidence grades

| Level    | Meaning                                                       |
|----------|---------------------------------------------------------------|
| STRONG   | Multiple challenger types passed cleanly                      |
| WEAK     | Confirmed but with a short streak / few challenger types      |
| DOUBTFUL | A group failed to respond, or escalation could not decide     |
| REFUTED  | A counterexample / fatal hole was found                       |

## Streaming annotation

`Orchestrator.run(...)` accepts any number of `Annotator` sinks. Each
proposition that exits the adversarial loop is written **immediately** -
you can `tail -f` the JSONL file while the workflow is still running.

A future MCP/PDF annotator will plug into the same protocol; nothing in
the core needs to change.

## Mitigations against the spec's risks

| Risk                          | Mitigation                                                                |
|-------------------------------|---------------------------------------------------------------------------|
| Hallucination propagation     | Every proposition carries a verbatim `SourceCitation`; CitationChecker validates |
| Cross-section symbol clashes  | Global `SymbolTable` with per-block scope; sub-agent re-renders on switch |
| Lemma circular dependencies   | `Blackboard.cycles()` detects them; topological order defers them        |
| Runaway adversarial loops     | `max_rounds_per_proposition`, `soft_pass_streak`, `doubt_streak` limits  |

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest                      # full test suite, mock LLM end-to-end
```

To use a real OpenAI model:

```bash
cp .env.example .env
# fill in OPENAI_API_KEY, optionally OPENAI_MODEL / OPENAI_BASE_URL
python -m article_learning.cli path/to/paper.md  # see CLI section
```

## Programmatic example

```python
from article_learning import Orchestrator
from article_learning.annotators import JSONLAnnotator
from article_learning.ingest import PaperLoader
from article_learning.llm import OpenAIClient

paper = PaperLoader().from_text_file("paper.md")
sinks = [JSONLAnnotator("annotations.jsonl")]
final = Orchestrator(OpenAIClient()).run(paper, annotators=sinks)
print(f"Produced {len(final['annotations'])} annotations")
```

## PDF input

Install the optional `pdf` extra:

```bash
pip install 'article-learning[pdf]'
```

Then `PaperLoader().from_pdf("paper.pdf")` will route through `marker-pdf`.

## Roadmap

* MCP-backed annotator that writes directly into the source PDF.
* LLM-driven semantic segmenter to replace the rule-based first pass.
* Joint verification mode for cycle-of-lemma cases.
* Human-in-the-loop checkpoint when a proposition becomes DOUBTFUL.
