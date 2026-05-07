"""Thin OpenAI wrapper that conforms to the ``LLMClient`` protocol."""

from __future__ import annotations

import logging
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from article_learning.config import get_settings
from article_learning.llm.base import ChatMessage, LLMResponse, MessageRole

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Real OpenAI-backed client.

    Imports the SDK lazily so the rest of the framework (and the test suite)
    doesn't require an OPENAI_API_KEY to be set.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.openai_model
        self._api_key = api_key or settings.openai_api_key
        self._base_url = base_url or settings.openai_base_url
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only with real LLM
            raise ImportError(
                "Real OpenAI client requested but the `openai` package is missing. "
                "Install with: pip install openai"
            ) from exc

        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured; set it in the environment or .env file."
            )

        kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)
        return self._client

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        retry=retry_if_exception_type(Exception),
    )
    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        client = self._ensure_client()
        payload = {
            "model": self.model,
            "messages": [{"role": _role(m.role), "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        completion = client.chat.completions.create(**payload)
        choice = completion.choices[0]
        text = choice.message.content or ""
        return LLMResponse(
            text=text,
            raw=completion,
            metadata={"model": self.model, "finish_reason": choice.finish_reason},
        )


def _role(role: MessageRole) -> str:
    return role.value
