"""Wire the LangGraph nodes into a runnable workflow."""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from article_learning.graph.nodes import (
    AgentBundle,
    challenge_router,
    judge_router,
    make_annotate_node,
    make_challenge_node,
    make_derive_node,
    make_judge_node,
    make_plan_node,
    make_respond_node,
    make_schedule_node,
    schedule_router,
)
from article_learning.graph.state import GraphState, build_initial_state
from article_learning.llm.base import LLMClient
from article_learning.models.paper import Paper

logger = logging.getLogger(__name__)


def build_workflow(llm: LLMClient, *, recursion_limit: int = 200):
    """Compile the LangGraph workflow.

    ``recursion_limit`` is set generously: the loop revisits the schedule
    node once per proposition-round pair, which can be many per paper.
    """
    agents = AgentBundle.build(llm)
    graph: StateGraph = StateGraph(GraphState)

    graph.add_node("plan", make_plan_node(agents))
    graph.add_node("schedule", make_schedule_node())
    graph.add_node("derive", make_derive_node(agents))
    graph.add_node("challenge", make_challenge_node(agents))
    graph.add_node("respond", make_respond_node(agents))
    graph.add_node("judge", make_judge_node(agents))
    graph.add_node("annotate", make_annotate_node())

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "schedule")
    graph.add_conditional_edges(
        "schedule",
        schedule_router,
        {"derive": "derive", "end": END},
    )
    graph.add_edge("derive", "challenge")
    graph.add_conditional_edges(
        "challenge",
        challenge_router,
        {"respond": "respond", "judge": "judge"},
    )
    graph.add_edge("respond", "judge")
    graph.add_conditional_edges(
        "judge",
        judge_router,
        {"challenge": "challenge", "annotate": "annotate"},
    )
    graph.add_edge("annotate", "schedule")

    return graph.compile().with_config({"recursion_limit": recursion_limit})


def run_workflow(llm: LLMClient, paper: Paper) -> GraphState:
    workflow = build_workflow(llm)
    initial = build_initial_state(paper)
    final = workflow.invoke(initial)
    return final  # type: ignore[return-value]
