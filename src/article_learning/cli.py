"""Minimal CLI: ``python -m article_learning <paper.md>``.

Streams JSONL annotations to stdout (and optionally a file) using the
real OpenAI client. Picks input format by extension: ``.pdf`` -> marker,
otherwise plain text/markdown.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from article_learning.annotators.json_annotator import JSONLAnnotator, StreamAnnotator
from article_learning.config import configure_logging
from article_learning.ingest.parser import PaperLoader
from article_learning.llm.openai_client import OpenAIClient
from article_learning.orchestrator import Orchestrator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="article-learning")
    parser.add_argument("source", help="Path to paper (.md / .txt / .pdf)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSONL output path. Stdout always streams.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override OPENAI_MODEL (default from .env).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="logging level (default INFO)",
    )
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    logging.getLogger().setLevel(args.log_level.upper())

    loader = PaperLoader()
    src = Path(args.source)
    if src.suffix.lower() == ".pdf":
        paper = loader.from_pdf(src)
    else:
        paper = loader.from_text_file(src)

    sinks = [StreamAnnotator(sys.stdout)]
    if args.out is not None:
        sinks.append(JSONLAnnotator(args.out))

    llm = OpenAIClient(model=args.model)
    Orchestrator(llm).run(paper, annotators=sinks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
