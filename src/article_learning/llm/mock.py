"""Mock LLMs used by the test suite.

Two flavours:
    * ``ScriptedMockLLM`` - returns a pre-canned sequence of responses.
    * ``DeterministicMockLLM`` - dispatches on a *tag* embedded in the system
      prompt so each agent gets a sensible reply without running real LLMs.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from article_learning.llm.base import ChatMessage, LLMResponse


@dataclass
class ScriptedMockLLM:
    """Returns the next entry from a list each time ``complete`` is called.

    Cycles when the script is exhausted (avoids surprise ``IndexError`` when
    a test composes a longer adversarial chain than expected).
    """

    script: list[str] = field(default_factory=list)
    cursor: int = 0
    calls: list[list[ChatMessage]] = field(default_factory=list)

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        if not self.script:
            return LLMResponse(text="{}")
        text = self.script[self.cursor % len(self.script)]
        self.cursor += 1
        return LLMResponse(text=text, metadata={"mock": True})


@dataclass
class DeterministicMockLLM:
    """Routes calls to handlers keyed on an ``[AGENT:<name>]`` tag.

    Every prompt the framework issues includes a tag like ``[AGENT:main]`` in
    the system message; the mock dispatches on that tag so we can exercise
    the full LangGraph in tests without hitting OpenAI.

    A default handler is invoked when no tag matches; tests fail loudly when
    that happens (instead of silently returning ``{}`` and letting JSON
    parsing fail many lines later).
    """

    handlers: dict[str, Callable[[list[ChatMessage]], str]] = field(default_factory=dict)
    default: Callable[[list[ChatMessage]], str] | None = None
    calls: list[tuple[str, list[ChatMessage]]] = field(default_factory=list)

    _tag_re = re.compile(r"\[AGENT:(?P<name>[a-zA-Z0-9_\-]+)\]")

    def register(self, tag: str, handler: Callable[[list[ChatMessage]], str]) -> None:
        self.handlers[tag] = handler

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        tag = self._extract_tag(messages)
        self.calls.append((tag, messages))
        handler = self.handlers.get(tag)
        if handler is None:
            if self.default is None:
                raise AssertionError(
                    f"No mock handler registered for tag {tag!r}. "
                    f"Registered: {sorted(self.handlers)}"
                )
            handler = self.default
        text = handler(messages)
        return LLMResponse(text=text, metadata={"mock": True, "tag": tag})

    def _extract_tag(self, messages: list[ChatMessage]) -> str:
        for msg in messages:
            match = self._tag_re.search(msg.content)
            if match:
                return match.group("name")
        return "<untagged>"
