"""LLM client abstraction layer (real + mock)."""

from article_learning.llm.base import ChatMessage, LLMClient, LLMResponse, MessageRole
from article_learning.llm.mock import DeterministicMockLLM, ScriptedMockLLM
from article_learning.llm.openai_client import OpenAIClient

__all__ = [
    "ChatMessage",
    "DeterministicMockLLM",
    "LLMClient",
    "LLMResponse",
    "MessageRole",
    "OpenAIClient",
    "ScriptedMockLLM",
]
