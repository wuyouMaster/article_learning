# article-learning

Adversarial multi-agent framework for **automatic paper derivation and
annotation**. Two agent groups argue over every claim; whatever survives
becomes a structured annotation.

[![CI](https://github.com/wuyouMaster/article_learning/actions/workflows/ci.yml/badge.svg)](https://github.com/wuyouMaster/article_learning/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/article-learning.svg)](https://pypi.org/project/article-learning/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

> Feed a paper (Markdown or PDF), get back structured, confidence-graded
> annotations — each one stress-tested by adversarial agents before it ships.

[简体中文 README](README.zh-CN.md)

## Features

- **Adversarial verification** — four challenger types (logic, assumption,
  counterexample, citation) stress-test every proposition before it's accepted
- **Streaming annotations** — results are written as they're produced; no need
  to wait for the full run to finish
- **Structured output** — every annotation is a Pydantic model with confidence
  level, derivation, citations, and full challenge history
- **PDF & Markdown input** — feed a `.pdf` (via `marker-pdf`) or `.md` file
- **Pluggable LLM backend** — works with any OpenAI-compatible API (OpenAI,
  DeepSeek, local models via vLLM/Ollama)
- **Dependency DAG** — propositions are topologically sorted; circular
  dependencies are detected and handled via joint verification
- **Symbol table** — tracks notation across sections so the same glyph isn't
  silently overloaded
- **Fully testable** — `DeterministicMockLLM` and `ScriptedMockLLM` let you
  run the entire pipeline without API keys

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

## Configuration

All settings are loaded from environment variables (or a `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | API key for the LLM provider |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name |
| `OPENAI_BASE_URL` | — | Override for non-OpenAI providers (e.g. DeepSeek) |
| `MAX_ROUNDS_PER_PROPOSITION` | `4` | Max adversarial rounds per proposition |
| `SOFT_PASS_STREAK` | `2` | Consecutive clean rounds to mark CONFIRMED |
| `DOUBT_STREAK` | `2` | Consecutive unanswered rounds to mark DOUBTFUL |
| `ARTICLE_LEARNING_LOG_LEVEL` | `INFO` | Logging verbosity |

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

## Import examples

### One-shot pipeline helper

`run_pipeline` is a convenience function that handles parsing and execution
in a single call:

```python
from article_learning import run_pipeline
from article_learning.annotators import JSONLAnnotator
from article_learning.llm import OpenAIClient

llm = OpenAIClient(model="gpt-4o")
annotations = run_pipeline(llm, "paper.md", annotators=[JSONLAnnotator("out.jsonl")])

for ann in annotations:
    print(f"{ann.proposition_id}: {ann.confidence.value} — {ann.statement[:80]}")
```

### Inspecting the Blackboard

After a run, the `GraphState` exposes a fully-populated `Blackboard` with
every proposition, its status, the challenge log, and the dependency DAG:

```python
from article_learning import Orchestrator, Blackboard, PropositionStatus
from article_learning.ingest import load_paper
from article_learning.llm import OpenAIClient

paper = load_paper("paper.md")
state = Orchestrator(OpenAIClient()).run(paper)
bb: Blackboard = state["blackboard"]

# All confirmed propositions
confirmed = bb.by_status(PropositionStatus.CONFIRMED)
print(f"{len(confirmed)} propositions confirmed")

# Walk the dependency DAG
import networkx as nx
graph: nx.DiGraph = bb.build_graph()
for node in nx.topological_sort(graph):
    prop = bb.get(node)
    print(f"  {node} ({prop.type.value}): {prop.statement[:60]}")

# Inspect adversarial history for a specific proposition
for record in bb.proposition_history("P1"):
    print(f"  round {record.round_index}: [{record.challenger}] {record.verdict}")
```

### Working with individual models

Every model is a Pydantic `BaseModel` — you can construct, serialize, and
validate them independently:

```python
from article_learning.models import (
    Annotation,
    ConfidenceLevel,
    Proposition,
    PropositionType,
    PropositionStatus,
    SourceCitation,
    Symbol,
    SymbolTable,
)

# Create a proposition manually
prop = Proposition(
    proposition_id="P1",
    type=PropositionType.THEOREM,
    statement="If f is continuous on [0,1] then f is bounded.",
    block_id="block-3",
    citations=[SourceCitation(block_id="block-3", quote="f is continuous on [0,1]")],
    depends_on=["P0"],
)

# Symbol table: track notation across sections
st = SymbolTable()
st.add(Symbol(
    name="f",
    description="Real-valued continuous function on [0,1]",
    introduced_in_block="block-1",
    scope_blocks=[],
))
resolved = st.lookup("f", "block-3")
print(resolved.description if resolved else "unknown symbol")

# Serialize an annotation to JSON
ann = Annotation(
    proposition_id="P1",
    block_id="block-3",
    statement=prop.statement,
    confidence=ConfidenceLevel.STRONG,
    rounds=3,
)
print(ann.model_dump_json(indent=2))
```

### Custom annotator

Implement the `Annotator` protocol to write annotations to any destination
(database, stdout, websocket, etc.):

```python
from article_learning.annotators import Annotator
from article_learning.models import Annotation


class PrintAnnotator:
    """Minimal custom annotator that prints to stdout."""

    def write(self, annotation: Annotation) -> None:
        icon = annotation.confidence.emoji
        print(f"{icon} {annotation.proposition_id}: {annotation.statement[:80]}")

    def close(self) -> None:
        pass


# Use it
from article_learning import Orchestrator
from article_learning.ingest import load_paper
from article_learning.llm import OpenAIClient

paper = load_paper("paper.md")
Orchestrator(OpenAIClient()).run(paper, annotators=[PrintAnnotator()])
```

### Writing to both JSONL and a final JSON file

Combine multiple annotators to get streaming output *and* a single-file
summary:

```python
from article_learning import Orchestrator
from article_learning.annotators import JSONFileAnnotator, JSONLAnnotator
from article_learning.ingest import load_paper
from article_learning.llm import OpenAIClient

paper = load_paper("paper.md")
annotators = [
    JSONLAnnotator("stream.jsonl"),      # tail -f this while running
    JSONFileAnnotator("annotations.json"), # single JSON array on close
]
Orchestrator(OpenAIClient()).run(paper, annotators=annotators)
```

### Using a mock LLM for testing / development

`DeterministicMockLLM` dispatches on agent tags so you can exercise the
full pipeline without API keys:

```python
import json
from article_learning import Orchestrator
from article_learning.ingest import PaperLoader
from article_learning.llm.mock import DeterministicMockLLM

mock = DeterministicMockLLM()

# Register handlers by agent tag
mock.register("main", lambda msgs: json.dumps({
    "propositions": [
        {
            "proposition_id": "P1",
            "type": "theorem",
            "statement": "Every bounded sequence has a convergent subsequence.",
            "formal_statement": None,
            "block_id": "block-0",
            "citation_quote": "bounded sequence ... convergent subsequence",
            "depends_on": [],
        }
    ],
    "symbols": [],
}))

mock.register("sub", lambda msgs: json.dumps({
    "derivation": "By the Bolzano-Weierstrass theorem.",
    "extra_citations": [],
    "notes": None,
}))

# Challengers: return a question on odd calls, pass on even
for tag in ("logic", "assumption", "counterexample", "citation"):
    mock.register(tag, lambda msgs, t=tag: json.dumps({
        "verdict": "no_issue", "question": "", "rationale": f"{t} pass"
    }))

paper = PaperLoader().from_markdown("# Test\nSome math here.")
state = Orchestrator(mock).run(paper)
print(f"Annotations: {len(state['annotations'])}")
```

### Streaming to a Rich console

`StreamAnnotator` writes JSON lines to any text stream — pair it with
`rich.console.Console` for pretty live output:

```python
import sys
from article_learning.annotators import StreamAnnotator
from article_learning import Orchestrator
from article_learning.ingest import load_paper
from article_learning.llm import OpenAIClient

paper = load_paper("paper.md")
stream_annotator = StreamAnnotator(sys.stdout)
Orchestrator(OpenAIClient()).run(paper, annotators=[stream_annotator])
```

### Accessing the LangGraph workflow directly

For full control over the graph (custom breakpoints, partial execution,
streaming individual nodes), use `build_workflow`:

```python
from article_learning.graph import build_workflow, build_initial_state
from article_learning.ingest import load_paper
from article_learning.llm import OpenAIClient

llm = OpenAIClient()
paper = load_paper("paper.md")
workflow = build_workflow(llm, recursion_limit=300)
initial = build_initial_state(paper)

# Stream node-by-node
for event in workflow.stream(initial, stream_mode="values"):
    annotations = event.get("annotations", [])
    if annotations:
        print(f"Got {len(annotations)} annotation(s) this step")
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
