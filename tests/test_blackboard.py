"""Blackboard / DAG / state-machine unit tests."""

from __future__ import annotations

import pytest

from article_learning.models.blackboard import Blackboard
from article_learning.models.proposition import (
    Proposition,
    PropositionStatus,
    PropositionType,
    SourceCitation,
)


def _prop(pid: str, *, deps: list[str] | None = None) -> Proposition:
    return Proposition(
        proposition_id=pid,
        type=PropositionType.LEMMA,
        statement=f"Statement of {pid}",
        block_id="block-0",
        citations=[SourceCitation(block_id="block-0", quote="x")],
        depends_on=deps or [],
    )


def test_add_and_get_proposition():
    bb = Blackboard()
    bb.add_proposition(_prop("P1"))
    assert bb.get("P1").status == PropositionStatus.PENDING


def test_status_transitions_touch_updated_at():
    bb = Blackboard()
    bb.add_proposition(_prop("P1"))
    before = bb.get("P1").updated_at
    bb.set_status("P1", PropositionStatus.IN_PROGRESS)
    assert bb.get("P1").updated_at >= before
    assert bb.get("P1").status == PropositionStatus.IN_PROGRESS


def test_terminal_states():
    for status in (
        PropositionStatus.CONFIRMED,
        PropositionStatus.REFUTED,
        PropositionStatus.DOUBTFUL,
    ):
        assert status.is_terminal
    for status in (
        PropositionStatus.PENDING,
        PropositionStatus.IN_PROGRESS,
        PropositionStatus.UNDER_CHALLENGE,
        PropositionStatus.ESCALATED,
    ):
        assert not status.is_terminal


def test_dag_dependency_and_topological_order():
    bb = Blackboard()
    for pid in ("P1", "P2", "P3"):
        bb.add_proposition(_prop(pid))
    bb.add_dependency("P1", "P2")
    bb.add_dependency("P2", "P3")
    order = bb.topological_order()
    assert order.index("P1") < order.index("P2") < order.index("P3")


def test_add_self_dependency_rejected():
    bb = Blackboard()
    bb.add_proposition(_prop("P1"))
    with pytest.raises(ValueError):
        bb.add_dependency("P1", "P1")


def test_add_unknown_parent_rejected():
    bb = Blackboard()
    bb.add_proposition(_prop("P1"))
    with pytest.raises(KeyError):
        bb.add_dependency("ghost", "P1")


def test_cycle_detection_and_special_ordering():
    bb = Blackboard()
    for pid in ("A", "B", "C"):
        bb.add_proposition(_prop(pid))
    # A <-> B mutual dependency, C standalone.
    bb.add_dependency("A", "B")
    bb.add_dependency("B", "A")
    cycles = bb.cycles()
    assert any(set(c) == {"A", "B"} for c in cycles)
    order = bb.topological_order()
    # Cycle members should come after acyclic ones.
    assert order.index("C") < order.index("A")
    assert order.index("C") < order.index("B")


def test_open_questions_and_resolution_flips_status():
    bb = Blackboard()
    bb.add_proposition(_prop("P1"))
    bb.add_open_question("P1", "why?")
    assert bb.get("P1").status == PropositionStatus.UNDER_CHALLENGE
    assert bb.open_questions["P1"] == ["why?"]
    bb.resolve_question("P1", "why?")
    assert bb.open_questions["P1"] == []


def test_duplicate_proposition_rejected():
    bb = Blackboard()
    bb.add_proposition(_prop("P1"))
    with pytest.raises(ValueError):
        bb.add_proposition(_prop("P1"))
