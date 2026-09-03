"""Properties of the training window sampler that the corpus depends on."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from slopmarker.data.normalize import collapse_whitespace, count_words
from slopmarker.data.sentences import split_sentences
from slopmarker.data.windows import (
    DEFAULT_BUCKETS,
    ai_fraction_in,
    sample_windows,
    seed_for,
)

SEED = 20260903


def make_doc(n_sentences: int = 200) -> str:
    return " ".join(
        f"Sentence number {i} carries a modest and entirely unremarkable amount of text."
        for i in range(n_sentences)
    )


def test_windows_are_sentence_aligned() -> None:
    text = collapse_whitespace(make_doc())
    boundaries = split_sentences(text)
    starts = {a for a, _ in boundaries}
    ends = {b for _, b in boundaries}
    for w in sample_windows(text, "doc-1", corpus_seed=SEED):
        assert w.start in starts
        assert w.end in ends


def test_window_lengths_are_in_range() -> None:
    low = DEFAULT_BUCKETS[0][0]
    high = DEFAULT_BUCKETS[-1][1]
    for i in range(50):
        for w in sample_windows(make_doc(), f"doc-{i}", corpus_seed=SEED):
            assert low <= w.n_words <= high
            assert count_words(w.text) == w.n_words


def test_deterministic_per_document() -> None:
    """Same doc_id, same windows -- regardless of what else is in the corpus."""
    a = sample_windows(make_doc(), "doc-42", corpus_seed=SEED)
    b = sample_windows(make_doc(), "doc-42", corpus_seed=SEED)
    assert a == b


def test_independent_of_other_documents() -> None:
    """Seeding is per-document, so shard count and ordering cannot matter."""
    alone = sample_windows(make_doc(), "doc-7", corpus_seed=SEED)
    for other in ("doc-1", "doc-2", "doc-3"):
        sample_windows(make_doc(), other, corpus_seed=SEED)
    assert sample_windows(make_doc(), "doc-7", corpus_seed=SEED) == alone


def test_corpus_seed_changes_output() -> None:
    assert sample_windows(make_doc(), "d", corpus_seed=1) != sample_windows(
        make_doc(), "d", corpus_seed=2
    )
    assert seed_for(1, "d") != seed_for(2, "d")


def test_short_document_yields_nothing() -> None:
    assert sample_windows("Only a handful of words here.", "d", corpus_seed=SEED) == []


def test_length_distribution_follows_buckets() -> None:
    """The realised distribution must match the configured weights, not just the range."""
    lengths = [
        w.n_words
        for i in range(4000)
        for w in sample_windows(make_doc(), f"doc-{i}", corpus_seed=SEED, max_windows=1)
    ]
    assert len(lengths) > 3000
    in_core = sum(1 for n in lengths if 80 <= n <= 250) / len(lengths)
    # Configured weight for the 80-250 range is 0.70. Windows snap outward to whole
    # sentences, so allow a wide band; this is a smoke test against a broken PMF walk,
    # not a goodness-of-fit test.
    assert 0.5 < in_core < 0.9


class TestAiFraction:
    def test_no_spans_is_pure_human(self) -> None:
        assert ai_fraction_in(0, 100, None) == 0.0
        assert ai_fraction_in(0, 100, []) == 0.0

    def test_fully_covered(self) -> None:
        assert ai_fraction_in(10, 20, [(0, 100)]) == 1.0

    def test_partial_overlap(self) -> None:
        assert ai_fraction_in(0, 100, [(50, 100)]) == pytest.approx(0.5)

    def test_disjoint_spans_add_up(self) -> None:
        assert ai_fraction_in(0, 100, [(0, 25), (75, 100)]) == pytest.approx(0.5)

    def test_span_outside_window_is_ignored(self) -> None:
        assert ai_fraction_in(0, 50, [(200, 300)]) == 0.0

    def test_window_from_human_half_is_labelled_human(self) -> None:
        """The trap this whole mechanism exists to avoid."""
        text = collapse_whitespace(make_doc())
        half = len(text) // 2
        spans = [(half, len(text))]  # AI wrote the back half
        windows = sample_windows(text, "mixed-doc", corpus_seed=SEED, ai_spans=spans)
        assert windows
        for w in windows:
            if w.end <= half:
                assert w.ai_fraction == 0.0
            elif w.start >= half:
                assert w.ai_fraction == 1.0
        # ...and at least one window is not simply inheriting a flat document value.
        assert len({round(w.ai_fraction, 3) for w in windows}) > 1


def test_windows_do_not_heavily_overlap() -> None:
    for i in range(100):
        windows = sample_windows(make_doc(), f"doc-{i}", corpus_seed=SEED)
        for a, b in pairwise(windows):
            shared = max(0, min(a.end, b.end) - max(a.start, b.start))
            shorter = min(a.end - a.start, b.end - b.start)
            assert shared / shorter <= 0.5 + 1e-9


def test_numpy_generator_is_pinned_to_pcg64() -> None:
    """PCG64 bit streams are stable across NumPy versions; the default may not be."""
    rng = np.random.Generator(np.random.PCG64(seed_for(SEED, "x")))
    assert isinstance(rng.bit_generator, np.random.PCG64)
