"""Cleaning, quality filters, and the rule that the human class is never screened
for sounding like AI."""

from __future__ import annotations

import inspect

import pytest

from slopmarker.corpus import clean
from slopmarker.corpus.clean import (
    CleanResult,
    clean_document,
    find_repeated_lines,
    quality_reason,
    slop_lexicon_score,
    strip_boilerplate,
)
from slopmarker.corpus.config import CleaningConfig

CFG = CleaningConfig()


def prose(n_sentences: int = 20) -> str:
    return " ".join(
        f"The committee met on a {'wet' if i % 2 else 'dry'} morning and reviewed "
        f"proposal number {i} without reaching any firm conclusion."
        for i in range(n_sentences)
    )


class TestBoilerplate:
    def test_drops_navigation_and_footer(self) -> None:
        text = "Home\nAbout Us\nReal content lives here.\n© 2019 Acme\nAll rights reserved"
        assert strip_boilerplate(text) == "Real content lives here."

    def test_drops_host_repeated_lines(self) -> None:
        text = "Acme Corp Newsroom\nActual article text.\nFollow Acme on Twitter"
        out = strip_boilerplate(text, frozenset({"Acme Corp Newsroom"}))
        assert "Acme Corp Newsroom" not in out
        assert "Actual article text." in out

    def test_keeps_ordinary_prose(self) -> None:
        text = "The report was released on Tuesday.\nIt ran to sixty pages."
        assert strip_boilerplate(text) == text

    def test_find_repeated_lines(self) -> None:
        docs = [f"Acme Newsroom\nStory {i} body text.\nFollow us" for i in range(60)]
        repeated = find_repeated_lines(docs, min_hosts=50)
        assert "Acme Newsroom" in repeated
        assert "Story 0 body text." not in repeated


class TestQuality:
    def test_ordinary_prose_passes(self) -> None:
        assert quality_reason(prose(), CFG) == ""

    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            ("", "empty"),
            (r"### | ^^ ~~ {} [] \ <> #### || ^^^ ~~~", "few_alpha_words"),
            ("a b c d e f g h i j k l m n o p", "mean_word_length"),
            ("\n".join(["the same line here"] * 10), "duplicate_lines"),
            (
                "\n".join(
                    f"the {w} sentence here continues..."
                    for w in ["first", "second", "third", "fourth", "fifth", "sixth"]
                ),
                "truncated_lines",
            ),
        ],
    )
    def test_rejects_junk(self, text: str, reason: str) -> None:
        assert quality_reason(text, CFG) == reason


class TestCleanDocument:
    def test_keeps_good_document(self) -> None:
        result = clean_document(prose(30), CFG)
        assert result.kept
        assert result.n_words >= CFG.min_words

    def test_rejects_short_document(self) -> None:
        result = clean_document("Three words only.", CFG)
        assert not result.kept
        assert result.reason == "too_short"

    def test_rejects_non_english(self) -> None:
        spanish = " ".join(
            "El comite se reunio por la manana y reviso la propuesta sin llegar "
            "a ninguna conclusion firme sobre el asunto tratado."
            for _ in range(15)
        )
        assert clean_document(spanish, CFG).reason == "not_english"

    def test_truncates_rather_than_drops_long_documents(self) -> None:
        cfg = CleaningConfig(min_words=10, max_words=60)
        result = clean_document(prose(50), cfg)
        assert result.kept
        assert result.n_words <= cfg.max_words

    def test_truncation_lands_on_a_sentence_boundary(self) -> None:
        cfg = CleaningConfig(min_words=10, max_words=60)
        result = clean_document(prose(50), cfg)
        assert result.text.rstrip().endswith(".")

    def test_quality_check_can_be_skipped(self) -> None:
        """FineWeb has already run C4 and Gopher; running them again removes nothing."""
        text = "\n".join(["repeated identical line of prose"] * 10) + "\n" + prose(20)
        assert clean_document(text, CFG, check_quality=True).reason == "duplicate_lines"
        assert clean_document(text, CFG, check_quality=False).kept


class TestSlopLexiconIsDiagnosticOnly:
    def test_score_detects_llm_phrasing(self) -> None:
        slop = "Let us delve into the rich tapestry of this multifaceted landscape. " * 5
        assert slop_lexicon_score(slop) > slop_lexicon_score(prose())

    def test_slop_heavy_human_text_is_still_kept(self) -> None:
        """The hard negatives scope.md 4.2 asks for must survive cleaning."""
        slop = (
            "In today's fast-paced landscape it is important to note that this "
            "serves as a testament to the multifaceted realm of enterprise. "
        ) * 8
        assert clean_document(slop, CFG).kept

    def test_no_filter_reads_the_slop_score(self) -> None:
        """Structural guard, not a behavioural one: the score must stay unread."""
        source = inspect.getsource(clean)
        body = source.split("def slop_lexicon_score", 1)[1].split("def ", 1)[1]
        assert "slop_lexicon_score" not in body
        assert "SLOP_LEXICON" not in body

    def test_clean_result_carries_no_slop_field(self) -> None:
        assert "slop" not in CleanResult.__dataclass_fields__
