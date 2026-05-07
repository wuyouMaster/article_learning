"""Annotator protocol.

The orchestrator calls ``write`` once per proposition that exits the
adversarial loop, enabling true *streaming* annotation - the spec calls
this out so a human can intervene mid-run.

A future MCP/PDF annotator will satisfy the same interface; nothing in
the core framework needs to change to swap implementations.
"""

from __future__ import annotations

from typing import Protocol

from article_learning.models.annotation import Annotation


class Annotator(Protocol):
    def write(self, annotation: Annotation) -> None:
        ...

    def close(self) -> None:
        ...
