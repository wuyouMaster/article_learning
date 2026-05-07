"""Common scaffolding for every agent in groups A and B."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from article_learning.llm.base import ChatMessage, LLMClient, MessageRole


@dataclass
class AgentContext:
    """Light context object passed to every agent invocation.

    Kept tiny on purpose: heavy state lives on the Blackboard which is
    threaded through the LangGraph state, not the agents themselves.
    """

    paper_title: str


class BaseAgent(ABC):
    """All agents have a stable ``name`` (used as the routing tag for mocks)."""

    name: str = "base"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    @abstractmethod
    def system_prompt(self) -> str: ...

    def _wrap(self, system: str, user: str) -> list[ChatMessage]:
        tag = f"[AGENT:{self.name}]"
        return [
            ChatMessage(role=MessageRole.SYSTEM, content=f"{tag}\n{system}"),
            ChatMessage(role=MessageRole.USER, content=user),
        ]
