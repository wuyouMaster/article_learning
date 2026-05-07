"""High-level entrypoint: paper in, structured annotations out.

The orchestrator wraps LangGraph's ``stream`` API so we can hand each
finished annotation to an attached :class:`Annotator` immediately rather
than waiting for the full graph to finish - that's the "streaming
annotation" requirement from the spec.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from article_learning.agents.base import AgentContext
from article_learning.annotators.base import Annotator
from article_learning.graph.state import GraphState, build_initial_state
from article_learning.graph.workflow import build_workflow
from article_learning.ingest.parser import PaperLoader
from article_learning.llm.base import LLMClient
from article_learning.models.annotation import Annotation
from article_learning.models.paper import Paper

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.workflow = build_workflow(llm)

    def run(
        self,
        paper: Paper,
        *,
        annotators: Iterable[Annotator] = (),
        context: AgentContext | None = None,
    ) -> GraphState:
        annotators = list(annotators)
        initial = build_initial_state(paper, context=context)
        emitted: set[str] = set()
        last_state: GraphState | None = None
        try:
            for chunk in self.workflow.stream(initial, stream_mode="values"):
                last_state = chunk  # type: ignore[assignment]
                for ann in chunk.get("annotations", []) or []:
                    if ann.proposition_id in emitted:
                        continue
                    emitted.add(ann.proposition_id)
                    for sink in annotators:
                        sink.write(ann)
        finally:
            for sink in annotators:
                try:
                    sink.close()
                except Exception:  # pragma: no cover - defensive
                    logger.exception("Annotator close failed")
        if last_state is None:
            raise RuntimeError("Workflow produced no state - check inputs")
        return last_state


def run_pipeline(
    llm: LLMClient,
    source: str | Path,
    *,
    annotators: Iterable[Annotator] = (),
) -> list[Annotation]:
    """One-shot helper: parse + run + return final annotation list."""
    loader = PaperLoader()
    path = Path(source)
    if path.suffix.lower() == ".pdf":
        paper = loader.from_pdf(path)
    elif path.exists():
        paper = loader.from_text_file(path)
    else:
        paper = loader.from_markdown(str(source))
    final = Orchestrator(llm).run(paper, annotators=annotators)
    return list(final.get("annotations", []) or [])
