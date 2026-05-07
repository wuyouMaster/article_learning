"""End-to-end LangGraph workflow tests with a mock LLM."""

from __future__ import annotations

from io import StringIO

from article_learning.annotators.json_annotator import JSONLAnnotator, StreamAnnotator
from article_learning.graph.workflow import run_workflow
from article_learning.llm.mock import DeterministicMockLLM
from article_learning.models.annotation import ConfidenceLevel
from article_learning.models.paper import Paper
from article_learning.orchestrator import Orchestrator


def test_workflow_runs_to_completion_for_all_propositions(
    deterministic_mock: DeterministicMockLLM, sample_paper: Paper
):
    final = run_workflow(deterministic_mock, sample_paper)
    annotations = final.get("annotations", [])
    assert len(annotations) == 3
    ids = {a.proposition_id for a in annotations}
    assert ids == {"P1", "P2", "P3"}
    for ann in annotations:
        assert ann.derivation, "every confirmed proposition should ship a derivation"
        assert ann.citations, "annotations must carry source citations"


def test_confidence_levels_are_assigned(
    deterministic_mock: DeterministicMockLLM, sample_paper: Paper
):
    final = run_workflow(deterministic_mock, sample_paper)
    levels = {a.confidence for a in final["annotations"]}
    assert levels.issubset(set(ConfidenceLevel))


def test_orchestrator_streams_to_annotators(
    deterministic_mock: DeterministicMockLLM, sample_paper: Paper, tmp_path
):
    jsonl_path = tmp_path / "out.jsonl"
    sink_jsonl = JSONLAnnotator(jsonl_path)
    buf = StringIO()
    sink_stream = StreamAnnotator(buf)

    Orchestrator(deterministic_mock).run(sample_paper, annotators=[sink_jsonl, sink_stream])

    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 3
    assert "proposition_id" in lines[0]
    streamed = buf.getvalue().strip().splitlines()
    assert len(streamed) == 3


def test_dependency_order_is_respected_in_emission(
    deterministic_mock: DeterministicMockLLM, sample_paper: Paper
):
    final = run_workflow(deterministic_mock, sample_paper)
    order = [a.proposition_id for a in final["annotations"]]
    assert order.index("P1") < order.index("P2") < order.index("P3")


def test_challenge_history_attached_to_annotation(
    deterministic_mock: DeterministicMockLLM, sample_paper: Paper
):
    final = run_workflow(deterministic_mock, sample_paper)
    for ann in final["annotations"]:
        assert ann.challenge_history, "expected non-empty adversarial trail"
        kinds = {row.challenger for row in ann.challenge_history}
        # At least 2 different challenger kinds should fire across rounds.
        assert len(kinds) >= 1
