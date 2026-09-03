"""The corpus gate, tested against deliberately broken corpora.

Each test builds a corpus with a known defect and asserts the probe notices. If these
pass, a probe failure on the real corpus means something real.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("sklearn")

from slopmarker.eval.shortcut_probe import (
    ProbeResult,
    gate,
    run_probes,
    surface_features,
)

WORDS = [
    "harbour",
    "lantern",
    "gravel",
    "meridian",
    "tundra",
    "copper",
    "salvage",
    "thicket",
    "bureau",
    "kettle",
    "quarry",
    "mantle",
    "drifting",
    "fathom",
    "pigment",
    "trellis",
    "cobalt",
    "marrow",
    "ledger",
    "tempest",
    "furrow",
    "beacon",
]
TOPIC_A = ["railway", "signal", "timetable", "platform", "junction", "siding", "freight"]
TOPIC_B = ["orchid", "pollen", "greenhouse", "compost", "trellis", "grafting", "seedling"]


def doc(rng: random.Random, n: int = 90, vocab: list[str] | None = None) -> str:
    bank = vocab or WORDS
    return " ".join(
        " ".join(rng.choice(bank) for _ in range(11)).capitalize() + "." for _ in range(n // 11)
    )


def corpus(builder, n: int = 220, seed: int = 0):
    """Return (texts, labels) with a balanced split."""
    rng = random.Random(seed)
    texts, labels = [], []
    for i in range(n):
        label = i % 2
        texts.append(builder(rng, label))
        labels.append(label)
    return texts, labels


def split(texts, labels, frac: float = 0.6):
    cut = int(len(texts) * frac)
    return texts[:cut], labels[:cut], texts[cut:], labels[cut:]


class TestSurfaceFeatures:
    def test_shape_is_stable(self) -> None:
        assert len(surface_features("Hello world.")) == 24
        assert len(surface_features("")) == 24

    def test_no_nans_on_degenerate_input(self) -> None:
        for text in ("", " ", "\n\n", "a"):
            assert all(v == v for v in surface_features(text))


class TestDetectsBrokenCorpora:
    def test_catches_markdown_only_separation(self) -> None:
        """The classic failure: AI rows keep their headings, human rows never had any."""

        def builder(rng: random.Random, label: int) -> str:
            body = doc(rng)
            return f"## Overview\n\n- point one\n- point two\n\n{body}" if label else body

        texts, labels = corpus(builder)
        results = run_probes(*split(texts, labels))
        by_name = {r.name: r for r in results}
        assert not by_name["P1 surface features"].passed
        ok, failures = gate(results)
        assert not ok and failures

    def test_catches_topic_separation(self) -> None:
        """If the classes talk about different things, topic alone separates them."""

        def builder(rng: random.Random, label: int) -> str:
            return doc(rng, vocab=TOPIC_B if label else TOPIC_A)

        results = run_probes(*split(*corpus(builder)))
        by_name = {r.name: r for r in results}
        assert not by_name["P5 topic only (content words)"].passed

    def test_catches_length_separation(self) -> None:
        """If AI documents are systematically longer, length alone separates them."""

        def builder(rng: random.Random, label: int) -> str:
            return doc(rng, n=330 if label else 66)

        results = run_probes(*split(*corpus(builder)))
        assert not {r.name: r for r in results}["P6 length only"].passed

    def test_catches_a_trivially_separable_corpus(self) -> None:
        """Disjoint vocabularies: bag-of-words hits the ceiling."""

        def builder(rng: random.Random, label: int) -> str:
            return doc(rng, vocab=TOPIC_A if label else TOPIC_B)

        results = run_probes(*split(*corpus(builder)))
        assert not {r.name: r for r in results}["P3 bag of words"].passed


class TestBandsNotCeilings:
    def test_an_indistinguishable_corpus_also_fails(self) -> None:
        """The half people forget: 0.5 is as wrong as 0.99.

        AI text genuinely carries lexical signatures. A corpus where nothing separates
        the classes means they were preprocessed differently or over-scrubbed, not that
        the corpus is exemplary.
        """

        def builder(rng: random.Random, label: int) -> str:
            return doc(rng)  # label carries no information at all

        results = run_probes(*split(*corpus(builder)))
        assert not {r.name: r for r in results}["P3 bag of words"].passed

    def test_probe_result_verdict(self) -> None:
        assert ProbeResult("x", 0.8, 0.7, 0.9).passed
        assert not ProbeResult("x", 0.95, 0.7, 0.9).passed
        assert not ProbeResult("x", 0.60, 0.7, 0.9).passed


def test_held_out_generator_transfer_is_measured() -> None:
    """P9: a probe that transfers perfectly to unseen generators learned the harness."""
    rng = random.Random(7)
    marker = "HARNESSTOKEN"

    def builder(_rng: random.Random, label: int) -> str:
        body = doc(rng)
        return f"{marker} {body}" if label else body

    texts, labels = corpus(builder)
    tr_x, tr_y, te_x, te_y = split(texts, labels)
    # Held-out rows carry the same harness marker, so a harness-learning probe
    # transfers perfectly and the gap collapses to zero.
    results = run_probes(tr_x, tr_y, te_x, te_y, heldout_texts=te_x, heldout_labels=te_y)
    p9 = {r.name: r for r in results}.get("P9 held-out generator transfer")
    assert p9 is not None
    assert not p9.passed, "a zero transfer gap must fail"
