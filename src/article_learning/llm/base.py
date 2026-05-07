"""LLM client protocol and shared types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    role: MessageRole
    content: str


@dataclass
class LLMResponse:
    text: str
    raw: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    """Minimal interface every backend (real or mock) must satisfy."""

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...


def parse_json_response(response: LLMResponse, schema: type[T]) -> T:
    """Best-effort JSON extraction from an LLM response.

    Tolerates fenced ``` blocks and stray prose around the JSON object.
    """
    text = response.text.strip()
    if text.startswith("```"):
        # Strip code fences. Handles ```json ... ``` and ``` ... ```.
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    # Find the outermost JSON object/array.
    start_obj = text.find("{")
    start_arr = text.find("[")
    candidates = [c for c in (start_obj, start_arr) if c != -1]
    if not candidates:
        raise ValueError(f"No JSON object found in response:\n{response.text}")
    start = min(candidates)
    end = max(text.rfind("}"), text.rfind("]"))
    if end < start:
        raise ValueError(f"Malformed JSON in response:\n{response.text}")
    payload = text[start : end + 1]

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to decode JSON: {exc}\nPayload was:\n{payload}") from exc

    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"JSON did not match {schema.__name__}: {exc}") from exc
