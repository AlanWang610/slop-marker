"""Hard-negative routing and structural signals."""

from __future__ import annotations

import pytest

from slopmarker.corpus.heuristics import genre_from_text, signals
from slopmarker.corpus.hosts import classify_url, is_corporate_host, registered_domain


class TestRegisteredDomain:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://www.prnewswire.com/a/b", "prnewswire.com"),
            ("https://news.acme.co.uk/x", "acme.co.uk"),  # multi-part suffix
            ("http://EXAMPLE.COM/", "example.com"),
            ("https://a.b.c.example.org/p", "example.org"),
            ("not a url", None),
        ],
    )
    def test_extracts_etld_plus_one(self, url: str, expected: str | None) -> None:
        assert registered_domain(url) == expected


class TestClassifyUrl:
    @pytest.mark.parametrize(
        ("url", "kind"),
        [
            ("https://www.prnewswire.com/news-releases/x.html", "press_release"),
            ("https://acme.co.uk/press-releases/q3", "press_release"),
            ("https://x.com/investor-relations/2021/a", "press_release"),
            ("https://www.thespruce.com/how-to-1234", "seo_marketing"),
            ("https://shop.example.com/products/widget", "product_template"),
            ("https://forum.example.org/viewtopic.php?t=9", "non_native_forum"),
            ("https://en.wikipedia.org/wiki/Rug", None),
        ],
    )
    def test_routes(self, url: str, kind: str | None) -> None:
        assert classify_url(url)[0] == kind

    def test_corporate_blog_needs_a_corporate_host(self) -> None:
        url = "https://acme.io/blog/why-we-built-this"
        assert classify_url(url, corporate_host=False)[0] is None
        assert classify_url(url, corporate_host=True)[0] == "corporate_blog"

    def test_blogging_platform_is_never_a_corporate_blog(self) -> None:
        """A /blog/ path on blogspot is a person, whatever else the host serves."""
        url = "https://someone.blogspot.com/blog/holiday"
        assert classify_url(url, corporate_host=True)[0] is None

    def test_every_hit_carries_an_audit_rule(self) -> None:
        kind, rule = classify_url("https://www.prnewswire.com/news-releases/x.html")
        assert kind is not None
        assert rule

    def test_query_string_is_searched(self) -> None:
        """FORUM_PATH matches on the query, so the path must include it."""
        assert classify_url("https://e.org/index.php?topic=5")[0] == "non_native_forum"


def test_is_corporate_host() -> None:
    assert is_corporate_host({"/blog/a", "/pricing/"})
    assert not is_corporate_host({"/blog/a", "/about-me"})


class TestSignals:
    def test_listicle(self) -> None:
        s = signals("10 Best Ways to Clean a Rug\n- one\n- two\n- three\n")
        assert s.listicle_title
        assert s.looks_listicle

    def test_marketing_cta(self) -> None:
        s = signals("Sign up today for free!\nBook a demo with our team.\n")
        assert s.cta_count >= 2
        assert s.looks_marketing

    def test_faq_structure(self) -> None:
        s = signals("What is it?\nHow does it work?\nIs it worth it?\n")
        assert s.faq_headings == 3
        assert s.looks_seo_structured

    def test_templated_product(self) -> None:
        s = signals("Material: wool\nColour: blue\nSize: 3x5\nSKU 12345\nFree shipping\n")
        assert s.looks_templated_product

    def test_ordinary_prose_trips_nothing(self) -> None:
        prose = (
            "The rain had been falling since morning and the road out of town was "
            "already soft. She walked anyway, because waiting had stopped feeling "
            "like a choice some hours earlier.\n"
        )
        s = signals(prose)
        assert not s.looks_listicle
        assert not s.looks_marketing
        assert not s.looks_templated_product
        assert not s.looks_seo_structured

    def test_empty_text_is_safe(self) -> None:
        s = signals("")
        assert s.bullet_density == 0.0
        assert not s.looks_listicle

    def test_title_used_when_body_has_no_lines(self) -> None:
        assert signals("", title="7 Tips for Better Sleep").listicle_title


class TestGenreFromText:
    """Text-only genre assignment. The AI side has no URL, so a URL-derived human
    label and a prompt-derived AI label would make the labelling process itself
    carry the class -- which is exactly the leak the auxiliary head would amplify."""

    @pytest.mark.parametrize(
        ("genre", "text"),
        [
            (
                "press_release",
                "ACME CORP, London, March 3, 2021 - Acme today announced that our new "
                "platform helps customers. Our team is excited.\n\nAbout Acme Corporation:\n"
                "We are a leading provider.",
            ),
            (
                "product_marketing",
                "Material: wool\nColour: blue\nSize: 3x5\nSKU 12345\nFree shipping. Add to cart.",
            ),
            (
                "academic_formal",
                "The results (2019) agree with prior work [12]. See Smith et al. (2020). "
                "Further analysis [7] confirms it.",
            ),
            (
                "technical_docs",
                "Run `npm install` first. Then import the module. Use `pip install foo` if needed.",
            ),
            (
                "forum_comment",
                "I tried this myself and my setup broke. I am not sure why. Anyone else seen this?",
            ),
        ],
    )
    def test_assigns_expected_genre(self, genre: str, text: str) -> None:
        assert genre_from_text(text)[0] == genre

    def test_long_first_person_prose_is_a_blog_not_a_forum_post(self) -> None:
        text = (
            " ".join(["I rebuilt my kitchen and my partner thought I was mad."] * 4)
            + " "
            + (" ".join(["The plumbing took three weekends of patient work."] * 20))
        )
        assert genre_from_text(text)[0] == "blog_personal"

    def test_falls_back_rather_than_guessing(self) -> None:
        genre, confidence = genre_from_text("The committee met on Tuesday. The report ran long.")
        assert genre in {"blog_personal", "news"}
        assert confidence <= 0.4, "an unmatched document must report low confidence"

    def test_never_returns_other(self) -> None:
        """Everything gets a genre; 'other' is what this function exists to remove."""
        for text in ("", "x", "The quick brown fox jumps over the lazy dog."):
            assert genre_from_text(text)[0] != "other"
