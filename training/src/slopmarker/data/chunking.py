"""Deterministic chunking of a paragraph block into scoreable units (scope.md 7.3).

This is what the extension does at inference time, so it is a coupling point pinned by
fixtures/windows.json. Keep it separate in your head from the *training* window sampler
(data/windows.py), which is stochastic and never ported.
"""

from __future__ import annotations

from .normalize import collapse_whitespace, count_words
from .sentences import split_sentences

MIN_WORDS = 40
MAX_WORDS = 400


def chunk_block(text: str, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> list[str]:
    """Split one block into <= max_words chunks at sentence boundaries.

    Blocks under `min_words` are skipped entirely (return []). A single sentence longer
    than `max_words` is emitted alone rather than cut mid-sentence -- that is the only
    way a chunk may exceed the cap.

    A trailing chunk may come out under `min_words`. That is deliberate: merging it back
    into the previous chunk would breach `max_words`, and scope.md 8 already shrinks
    short chunks toward 0.5 by `words / 80`, so a short tail can only extend a run, never
    start one.
    """
    text = collapse_whitespace(text)
    if count_words(text) < min_words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for start, end in split_sentences(text):
        sentence = text[start:end]
        words = count_words(sentence)
        if current and current_words + words > max_words:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += words
    if current:
        chunks.append(" ".join(current))
    return chunks
