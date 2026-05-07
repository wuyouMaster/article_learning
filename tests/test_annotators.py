"""JSON annotator unit tests."""

from __future__ import annotations

import json
from io import StringIO

from article_learning.annotators.json_annotator import (
    JSONFileAnnotator,
    JSONLAnnotator,
    StreamAnnotator,
)
from article_learning.models.annotation import Annotation, ConfidenceLevel


def _ann(pid: str = "P1") -> Annotation:
    return Annotation(
        proposition_id=pid,
        block_id="block-0",
        statement="hello",
        confidence=ConfidenceLevel.STRONG,
        derivation="proof here",
    )


def test_jsonl_annotator_writes_one_line_per_record(tmp_path):
    path = tmp_path / "anns.jsonl"
    sink = JSONLAnnotator(path)
    sink.write(_ann("P1"))
    sink.write(_ann("P2"))
    sink.close()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["proposition_id"] == "P1"
    assert parsed[1]["proposition_id"] == "P2"


def test_json_file_annotator_writes_array_on_close(tmp_path):
    path = tmp_path / "anns.json"
    sink = JSONFileAnnotator(path)
    sink.write(_ann("P1"))
    sink.close()
    payload = json.loads(path.read_text())
    assert isinstance(payload, list)
    assert payload[0]["proposition_id"] == "P1"


def test_stream_annotator_writes_to_buffer():
    buf = StringIO()
    sink = StreamAnnotator(buf)
    sink.write(_ann("P1"))
    sink.write(_ann("P2"))
    sink.close()
    assert buf.getvalue().count("\n") == 2


def test_confidence_emoji_rendering():
    for level in ConfidenceLevel:
        assert level.emoji
