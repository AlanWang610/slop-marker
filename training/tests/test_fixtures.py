"""Assert the Python side matches fixtures/. The TypeScript suite reads the same files.

Neither side generates them at test time -- fixtures are committed artifacts produced by
training/scripts/make_fixtures.py. If a change here makes these fail, that is the point:
regenerate deliberately and review the diff, because it is also an extension change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slopmarker.data.chunking import chunk_block
from slopmarker.data.normalize import (
    collapse_whitespace,
    content_hash,
    count_words,
    normalize_for_hash,
)
from slopmarker.data.sentences import sentences

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def cases(name: str, key: str) -> list:
    data = load(name)
    return [pytest.param(c, id=c["name"]) for c in data[key]]


@pytest.mark.parametrize("case", cases("normalize.json", "cases"))
def test_normalize(case: dict) -> None:
    assert collapse_whitespace(case["input"]) == case["collapsed"]
    assert normalize_for_hash(case["input"]) == case["normalized"]
    assert content_hash(case["input"]) == case["sha256"]
    assert count_words(case["input"]) == case["words"]


@pytest.mark.parametrize("case", cases("windows.json", "sentence_cases"))
def test_sentences(case: dict) -> None:
    assert sentences(case["input"]) == case["sentences"]


@pytest.mark.parametrize("case", cases("windows.json", "chunk_cases"))
def test_chunk_block(case: dict) -> None:
    got = chunk_block(case["input"], case["min_words"], case["max_words"])
    assert got == [c["text"] for c in case["chunks"]]
    assert [count_words(c) for c in got] == [c["words"] for c in case["chunks"]]


def test_sentences_cover_input_exactly() -> None:
    """Spans must partition the input: no dropped or duplicated characters."""
    for case in load("windows.json")["sentence_cases"]:
        text = case["input"]
        assert " ".join(sentences(text)) == text.strip()


def test_chunks_respect_max_words() -> None:
    """The only chunk allowed to exceed max_words is a lone oversized sentence."""
    for case in load("windows.json")["chunk_cases"]:
        chunks = case["chunks"]
        if len(chunks) <= 1:
            continue
        assert all(c["words"] <= case["max_words"] for c in chunks), case["name"]


def test_hash_ignores_typography_but_model_input_does_not() -> None:
    """The coupling that matters: the cache key folds typography, the model input keeps it."""
    curly = "The report — released Tuesday — said “yes”."
    plain = 'The report - released Tuesday - said "yes".'
    assert content_hash(curly) == content_hash(plain)
    assert collapse_whitespace(curly) != collapse_whitespace(plain)
