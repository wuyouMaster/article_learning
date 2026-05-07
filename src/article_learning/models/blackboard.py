"""Blackboard: shared state store for the A/B agent groups.

Implements the central store described in the spec:
    - Proposition status state-machine
    - Owner of each proposition
    - Open B-group questions per proposition
    - Symbol table updates
    - Cross-proposition DAG (built by the main agent)
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from article_learning.models.proposition import Proposition, PropositionStatus
from article_learning.models.symbols import SymbolTable


class ChallengeRecord(BaseModel):
    """Audit trail entry for one challenge/response exchange."""

    proposition_id: str
    round_index: int
    challenger: str = Field(description="logic | assumption | counterexample | citation")
    question: str
    response: str | None = None
    verdict: str | None = Field(
        default=None,
        description="resolved | unresolved | refuted | pending",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Blackboard(BaseModel):
    """Single source of truth shared across the LangGraph state.

    All mutations go through methods on this class so the state-machine
    transitions are auditable and testable in isolation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    propositions: dict[str, Proposition] = Field(default_factory=dict)
    open_questions: dict[str, list[str]] = Field(
        default_factory=dict,
        description="proposition_id -> list of unresolved B-group questions.",
    )
    challenge_log: list[ChallengeRecord] = Field(default_factory=list)
    symbol_table: SymbolTable = Field(default_factory=SymbolTable)

    # The DAG is stored as adjacency (parent -> children) for serialization.
    # We rebuild a networkx graph on demand via `build_graph`.
    dag_edges: list[tuple[str, str]] = Field(
        default_factory=list,
        description="(parent_id, child_id) - child depends on parent.",
    )

    # ---------- proposition lifecycle ----------

    def add_proposition(self, prop: Proposition) -> None:
        if prop.proposition_id in self.propositions:
            raise ValueError(f"Duplicate proposition: {prop.proposition_id}")
        self.propositions[prop.proposition_id] = prop

    def get(self, prop_id: str) -> Proposition:
        return self.propositions[prop_id]

    def set_status(self, prop_id: str, status: PropositionStatus) -> None:
        prop = self.propositions[prop_id]
        prop.status = status
        prop.touch()

    def assign_owner(self, prop_id: str, owner: str) -> None:
        prop = self.propositions[prop_id]
        prop.owner_agent = owner
        prop.touch()

    def record_derivation(self, prop_id: str, derivation: str) -> None:
        prop = self.propositions[prop_id]
        prop.derivation = derivation
        prop.touch()

    # ---------- DAG ----------

    def add_dependency(self, parent_id: str, child_id: str) -> None:
        if parent_id == child_id:
            raise ValueError("Self-loop dependency is invalid")
        if parent_id not in self.propositions:
            raise KeyError(f"Unknown parent proposition {parent_id}")
        if child_id not in self.propositions:
            raise KeyError(f"Unknown child proposition {child_id}")
        edge = (parent_id, child_id)
        if edge not in self.dag_edges:
            self.dag_edges.append(edge)
            self.propositions[child_id].depends_on.append(parent_id)

    def build_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.propositions.keys())
        graph.add_edges_from(self.dag_edges)
        return graph

    def cycles(self) -> list[list[str]]:
        """Returns lemma/theorem cycles for special handling.

        Mutually-referential lemmas need joint verification rather than
        pure topological order; the orchestrator inspects this on every
        scheduling pass.
        """
        graph = self.build_graph()
        try:
            return [list(c) for c in nx.simple_cycles(graph)]
        except nx.NetworkXNoCycle:
            return []

    def topological_order(self) -> list[str]:
        """Topological order of propositions, ignoring cycle members.

        Cycle members are returned grouped at the end (joint-verification).
        """
        graph = self.build_graph()
        cycles = self.cycles()
        cyclic_nodes: set[str] = {n for cyc in cycles for n in cyc}
        acyclic = graph.subgraph([n for n in graph.nodes if n not in cyclic_nodes]).copy()
        order = list(nx.topological_sort(acyclic))
        order.extend(sorted(cyclic_nodes))
        return order

    # ---------- adversarial loop ----------

    def add_open_question(self, prop_id: str, question: str) -> None:
        self.open_questions.setdefault(prop_id, []).append(question)
        self.set_status(prop_id, PropositionStatus.UNDER_CHALLENGE)

    def resolve_question(self, prop_id: str, question: str) -> None:
        if prop_id in self.open_questions and question in self.open_questions[prop_id]:
            self.open_questions[prop_id].remove(question)

    def log_challenge(self, record: ChallengeRecord) -> None:
        self.challenge_log.append(record)

    def proposition_history(self, prop_id: str) -> list[ChallengeRecord]:
        return [r for r in self.challenge_log if r.proposition_id == prop_id]

    # ---------- queries ----------

    def by_status(self, status: PropositionStatus) -> list[Proposition]:
        return [p for p in self.propositions.values() if p.status == status]

    def all_propositions(self) -> Iterable[Proposition]:
        return self.propositions.values()

    def all_terminal(self) -> bool:
        return all(p.status.is_terminal for p in self.propositions.values())
