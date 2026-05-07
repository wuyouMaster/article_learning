"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    log_level: str = os.getenv("ARTICLE_LEARNING_LOG_LEVEL", "INFO")

    # Adversarial loop limits.
    max_rounds_per_proposition: int = int(os.getenv("MAX_ROUNDS_PER_PROPOSITION", "4"))
    soft_pass_streak: int = int(os.getenv("SOFT_PASS_STREAK", "2"))
    doubt_streak: int = int(os.getenv("DOUBT_STREAK", "2"))


def get_settings() -> Settings:
    return Settings()


def configure_logging(level: str | None = None) -> None:
    settings = get_settings()
    logging.basicConfig(
        level=(level or settings.log_level).upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
