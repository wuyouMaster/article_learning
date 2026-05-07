"""JSON annotators (streaming JSONL + final dump)."""

from __future__ import annotations

import json
from io import TextIOBase
from pathlib import Path
from typing import IO

from article_learning.models.annotation import Annotation


class JSONLAnnotator:
    """Append-only JSONL writer; one annotation per line.

    The file is left open across writes so partial output is observable
    while the workflow runs.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp: IO[str] | None = self.path.open("w", encoding="utf-8")

    def write(self, annotation: Annotation) -> None:
        if self._fp is None:
            raise RuntimeError("Annotator already closed")
        self._fp.write(annotation.model_dump_json() + "\n")
        self._fp.flush()

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None


class JSONFileAnnotator:
    """Buffers annotations and writes a single JSON array on close.

    Useful when the consumer expects a regular JSON file rather than JSONL.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buf: list[Annotation] = []

    def write(self, annotation: Annotation) -> None:
        self._buf.append(annotation)

    def close(self) -> None:
        payload = [a.model_dump(mode="json") for a in self._buf]
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


class StreamAnnotator:
    """Writes each annotation as JSON to an arbitrary text stream.

    Handy for piping to stdout / a Rich console / a websocket.
    """

    def __init__(self, stream: TextIOBase) -> None:
        self.stream = stream

    def write(self, annotation: Annotation) -> None:
        self.stream.write(annotation.model_dump_json() + "\n")
        self.stream.flush()

    def close(self) -> None:
        # The stream is owned by the caller.
        pass
