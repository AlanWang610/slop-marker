"""Training window sampler (scope.md 4.3).

This is the *stochastic* windower and it is training-only. Do not confuse it with
chunking.chunk_block, which is deterministic, is what the extension does at inference
time, and is the thing fixtures pin. Both cut at boundaries from the same sentence
splitter, which is what keeps the training distribution aligned with what gets scored.

Two properties matter more than the sampling detail:

Per-document seeding. The RNG is seeded from (corpus_seed, doc_id), so windowing is
independent of shard count and of the order documents are processed. Adding documents
never changes any existing document's windows, which makes the corpus append-only and a
partial rerun safe.

Span-aware labels. A window inherits its ai_fraction from the text it actually covers,
not from its document. A window cut from the human half of a half-AI document is 100%
human; giving it the document's 0.5 would be label noise proportional to how localized
the AI text is.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .normalize import collapse_whitespace, count_words
from .sentences import split_sentences

# (min_words, max_words, weight). Explicit and inspectable rather than a parametric
# distribution: scope.md 4.3 asks for 40-400 weighted toward 80-250, and this says
# exactly that. Short windows are kept deliberately so the model learns low confidence
# on them rather than confident noise.
LengthBuckets = tuple[tuple[int, int, float], ...]
DEFAULT_BUCKETS: LengthBuckets = (
    (40, 80, 0.10),
    (80, 130, 0.20),
    (130, 170, 0.30),
    (170, 250, 0.20),
    (250, 400, 0.20),
)

MAX_PLACEMENT_TRIES = 20
MAX_OVERLAP = 0.5  # reject a window overlapping an accepted one by more than this


@dataclass(frozen=True)
class Window:
    start: int  # character offsets into the collapsed document text
    end: int
    text: str
    n_words: int
    ai_fraction: float


def seed_for(corpus_seed: int, doc_id: str) -> int:
    digest = hashlib.blake2b(f"{corpus_seed}:{doc_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def ai_fraction_in(start: int, end: int, ai_spans: list[tuple[int, int]] | None) -> float:
    """Fraction of characters in [start, end) that fall inside an AI-authored span.

    Characters rather than tokens: it needs no tokenizer, is exact, and is computed the
    same way at document and window level so the two can never disagree.
    """
    if end <= start:
        return 0.0
    if not ai_spans:
        return 0.0
    covered = sum(
        max(0, min(end, span_end) - max(start, span_start)) for span_start, span_end in ai_spans
    )
    return covered / (end - start)


def _draw_target_length(rng: np.random.Generator, buckets: LengthBuckets) -> int:
    """Pick a bucket by weight, then a length uniformly inside it.

    The cumulative sum is walked by hand rather than using Generator.choice(p=...),
    whose algorithm has changed between NumPy versions and would silently change the
    corpus.
    """
    total = sum(w for _, _, w in buckets)
    draw = rng.random() * total
    running = 0.0
    for low, high, weight in buckets:
        running += weight
        if draw < running:
            return int(rng.integers(low, high + 1))
    low, high, _ = buckets[-1]
    return int(rng.integers(low, high + 1))


def sample_windows(
    text: str,
    doc_id: str,
    *,
    corpus_seed: int,
    ai_spans: list[tuple[int, int]] | None = None,
    buckets: LengthBuckets = DEFAULT_BUCKETS,
    max_windows: int = 10,
) -> list[Window]:
    """Cut sentence-aligned windows from one document. Deterministic given doc_id."""
    text = collapse_whitespace(text)
    sentence_spans = split_sentences(text)
    if not sentence_spans:
        return []

    # Per-sentence word counts, computed once. The placement loop below runs up to
    # MAX_PLACEMENT_TRIES times per window and walks sentences each time, so
    # recomputing these inside it would call count_words -- a character-by-character
    # scan -- thousands of times per document.
    sentence_words = [count_words(text[start:end]) for start, end in sentence_spans]
    starts_at_word: list[int] = []
    running = 0
    for words in sentence_words:
        starts_at_word.append(running)
        running += words
    total_words = running
    if total_words < buckets[0][0]:
        return []

    n_target = min(max(1, -(-total_words // 200)), max_windows)
    rng = np.random.Generator(np.random.PCG64(seed_for(corpus_seed, doc_id)))

    windows: list[Window] = []
    for _ in range(n_target):
        window = _place_one(
            text, sentence_spans, sentence_words, starts_at_word, total_words, rng, buckets, windows
        )
        if window is not None:
            windows.append(window)

    windows.sort(key=lambda w: w.start)
    return [
        Window(
            start=w.start,
            end=w.end,
            text=w.text,
            n_words=w.n_words,
            ai_fraction=ai_fraction_in(w.start, w.end, ai_spans),
        )
        for w in windows
    ]


def _place_one(
    text: str,
    sentence_spans: list[tuple[int, int]],
    sentence_words: list[int],
    starts_at_word: list[int],
    total_words: int,
    rng: np.random.Generator,
    buckets: LengthBuckets,
    accepted: list[Window],
) -> Window | None:
    min_words = buckets[0][0]
    max_words = buckets[-1][1]
    for _ in range(MAX_PLACEMENT_TRIES):
        target = _draw_target_length(rng, buckets)
        # Uniform over word offsets, not sentence indices: sentence-uniform placement
        # would over-sample regions made of short sentences.
        word_offset = int(rng.integers(0, max(1, total_words)))
        first = _sentence_containing_word(starts_at_word, word_offset)

        # `last` only advances over sentences actually included, so `words` always
        # describes exactly the text between `start` and `end`.
        last = -1
        words = 0
        for index in range(first, len(sentence_spans)):
            span_words = sentence_words[index]
            if words and words + span_words > max_words:
                break
            words += span_words
            last = index
            if words >= target:
                break
        if last < first or words < min_words:
            continue

        start = sentence_spans[first][0]
        end = sentence_spans[last][1]
        if _overlap_fraction(start, end, accepted) > MAX_OVERLAP:
            continue
        return Window(start=start, end=end, text=text[start:end], n_words=words, ai_fraction=0.0)
    return None


def _sentence_containing_word(starts_at_word: list[int], word_offset: int) -> int:
    index = int(np.searchsorted(starts_at_word, word_offset, side="right")) - 1
    return max(0, index)


def _overlap_fraction(start: int, end: int, accepted: list[Window]) -> float:
    """Worst overlap with an already-accepted window, relative to the shorter of the two.

    Measuring against the candidate alone would be asymmetric: a short window sitting
    entirely inside a long one is rejected if the long one came first, but accepted if
    it came second.
    """
    if end <= start:
        return 1.0
    worst = 0.0
    for window in accepted:
        shared = max(0, min(end, window.end) - max(start, window.start))
        shorter = min(end - start, window.end - window.start)
        if shorter > 0:
            worst = max(worst, shared / shorter)
    return worst
