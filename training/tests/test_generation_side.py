"""Prompt construction, harness hygiene, and honest mixed-authorship fractions."""

from __future__ import annotations

import random
import zlib

import pytest

from slopmarker.corpus.hygiene import clean_generation
from slopmarker.corpus.mixing import draw_target_fraction, splice
from slopmarker.corpus.prompts import (
    GENRE_REGISTER,
    PERSONAS,
    SeedCard,
    build,
    structure_clause,
)
from slopmarker.data.genre import ALL_LABELS

BANK = [
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
]


def para(tag: str, n: int = 40) -> str:
    """Varied text tagged by `tag`, so it is not itself degenerate repetition."""
    # zlib.crc32, not hash(): Python randomizes string hashing per process, which
    # would make this fixture -- and therefore these tests -- non-deterministic.
    rng = random.Random(zlib.crc32(tag.encode()))
    words = [tag] + [rng.choice(BANK) for _ in range(n - 1)]
    return " ".join(words) + "."


HUMAN = "\n\n".join(f"human{i} " + para(f"h{i}") for i in range(5))
AI = "\n\n".join(f"MACHINE{i} " + para(f"a{i}") for i in range(5))


class TestPrompts:
    def test_structure_mirrors_the_seed(self) -> None:
        plain = SeedCard(doc_id="d", genre="news", topic="t", headings=0, lists=0)
        assert "Do not use headings" in structure_clause(plain)
        rich = SeedCard(doc_id="d", genre="news", topic="t", headings=3, lists=5)
        assert "3 section headings" in structure_clause(rich)

    def test_every_genre_has_a_register(self) -> None:
        for genre in ALL_LABELS:
            assert genre in GENRE_REGISTER
            assert PERSONAS.get(genre)

    def test_topic_prompt_never_shows_the_human_text(self) -> None:
        """The one style whose wording is not derived from human wording."""
        card = SeedCard(doc_id="d", genre="news", topic="rugs", text="SECRET HUMAN TEXT")
        prompt = build(card, "topic_prompt")
        assert "SECRET HUMAN TEXT" not in prompt.system + prompt.user

    def test_rewrite_carries_the_seed_text(self) -> None:
        card = SeedCard(doc_id="d", genre="news", topic="rugs", text="THE SEED")
        assert "THE SEED" in build(card, "rewrite").user

    def test_prompt_id_is_stable_and_content_addressed(self) -> None:
        card = SeedCard(doc_id="d", genre="news", topic="rugs")
        a, b = build(card, "topic_prompt"), build(card, "topic_prompt")
        assert a.prompt_id == b.prompt_id
        assert a.prompt_id != build(card, "structured").prompt_id


class TestHygiene:
    def test_strips_preamble(self) -> None:
        text = "Certainly! Here's a blog post about rugs.\n\n" + para("rug", 300)
        result = clean_generation(text, target_words=300)
        assert result.kept
        assert "Certainly" not in result.text
        assert "strip_preamble" in result.ops

    def test_strips_leaked_thinking(self) -> None:
        text = "<thinking>I should write about rugs</thinking>\n" + para("rug", 300)
        result = clean_generation(text, target_words=300)
        assert "<thinking>" not in result.text
        assert "strip_leaked_thinking" in result.ops

    def test_strips_trailing_offer(self) -> None:
        text = para("rug", 300) + "\n\nLet me know if you'd like me to expand on this!"
        result = clean_generation(text, target_words=300)
        assert "Let me know" not in result.text
        assert "strip_trailing_offer" in result.ops

    def test_strips_code_fence(self) -> None:
        result = clean_generation("```markdown\n" + para("rug", 300) + "\n```", target_words=300)
        assert "```" not in result.text
        assert "strip_code_fence" in result.ops

    def test_wrapper_title_kept_when_seed_had_headings(self) -> None:
        text = "# Rugs\n\n" + para("rug", 300)
        assert "strip_wrapper_title" in clean_generation(text, target_words=300).ops
        with_heading = clean_generation(text, target_words=300, seed_had_heading=True)
        assert "strip_wrapper_title" not in with_heading.ops

    def test_rejects_refusal(self) -> None:
        assert clean_generation("I can't help with that.", target_words=300).rejected == "refusal"

    def test_rejects_degenerate_repetition(self) -> None:
        text = "the same eight words repeated over and over " * 60
        assert clean_generation(text, target_words=300).rejected == "degenerate_repetition"

    def test_rejects_too_short(self) -> None:
        assert clean_generation(para("rug", 50), target_words=300).rejected == "too_short"

    def test_truncates_overlong(self) -> None:
        result = clean_generation(para("rug", 2000), target_words=300)
        assert result.kept
        assert "length_conform" in result.ops

    def test_keeps_model_style(self) -> None:
        """Em dashes, headings and 'In conclusion' are AI style, not harness artifacts."""
        text = (
            "## Overview\n\n"
            + para("body-one", 150)
            + "\n\nIn conclusion \u2014 and this matters \u2014 rugs endure. "
            + para("body-two", 150)
        )
        result = clean_generation(text, target_words=300, seed_had_heading=True)
        assert result.kept
        assert "In conclusion" in result.text
        assert "\u2014" in result.text
        assert "##" in result.text


class TestMixing:
    def test_fraction_is_measured_not_assumed(self) -> None:
        doc = splice(HUMAN, AI, 0.4, seed=1)
        assert doc is not None
        covered = sum(e - s for s, e in doc.ai_spans)
        assert doc.ai_fraction == pytest.approx(covered / len(doc.text))

    def test_spans_actually_cover_ai_text(self) -> None:
        """The spans must point at AI content, or every window label is wrong."""
        doc = splice(HUMAN, AI, 0.5, seed=2)
        assert doc is not None
        for start, end in doc.ai_spans:
            block = doc.text[start:end]
            assert "MACHINE" in block
            assert "human" not in block

    def test_fraction_tracks_the_target(self) -> None:
        for target in (0.2, 0.4, 0.6, 0.8):
            doc = splice(HUMAN, AI, target, seed=3)
            assert doc is not None
            assert abs(doc.ai_fraction - target) < 0.3

    def test_zero_target_yields_pure_human(self) -> None:
        doc = splice(HUMAN, AI, 0.0, seed=4)
        assert doc is not None
        assert doc.ai_spans == []
        assert doc.ai_fraction == 0.0

    def test_spans_are_merged_and_ordered(self) -> None:
        doc = splice(HUMAN, AI, 0.9, seed=5)
        assert doc is not None
        for (_, end), (start, _) in zip(doc.ai_spans, doc.ai_spans[1:], strict=False):
            assert end < start

    def test_deterministic(self) -> None:
        assert splice(HUMAN, AI, 0.5, seed=7) == splice(HUMAN, AI, 0.5, seed=7)

    def test_target_distribution_covers_the_low_band(self) -> None:
        """scope.md 1's non-goal is only enforceable if low fractions exist."""
        rng = random.Random(0)
        draws = [draw_target_fraction(rng) for _ in range(4000)]
        assert sum(1 for d in draws if d < 0.15) / len(draws) > 0.10
        assert sum(1 for d in draws if 0.5 <= d <= 0.9) / len(draws) > 0.25
        assert all(0.0 < d < 1.0 for d in draws)
