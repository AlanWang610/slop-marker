"""Structural signals for hard-negative mining.

Host lists (hosts.py) catch the well-known offenders; these catch the shape of the
writing itself, which generalizes to hosts nobody listed. Everything here is a cheap
count over lines, deliberately: it runs over hundreds of millions of documents.

Nothing here is ever used to filter the human class. See `slop_lexicon_score` in
schema.py for why that distinction matters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LISTICLE_TITLE = re.compile(
    r"^\s*(?:the\s+)?(?:top\s+)?\d{1,2}\s+"
    r"(best|top|ways|tips|reasons|things|steps|ideas|examples|tools|"
    r"strategies|mistakes|benefits|trends|hacks|myths)\b",
    re.I,
)
LISTICLE_PHRASE = re.compile(
    r"\b(ultimate guide|complete guide|step[-\s]by[-\s]step|"
    r"everything you need to know|the definitive guide|"
    r"(best|top)\s+\d{1,2}\b)",
    re.I,
)
CTA = re.compile(
    r"\b(sign up (today|now|for free)|get started (today|now|for free)|"
    r"book a (demo|call)|request a (quote|demo)|contact us today|"
    r"shop now|free trial|limited time|no credit card required|"
    r"subscribe (to our|now)|download the (free )?(ebook|guide|whitepaper)|"
    r"talk to (an expert|sales))\b",
    re.I,
)
BULLET_LINE = re.compile(r"^\s*(?:[-*\u2022\u2013\u2014\u25cf\u25aa]|\d{1,2}[.)])\s+")
SPEC_LINE = re.compile(r"^[A-Z][A-Za-z /]{2,25}:\s*\S")
PRODUCT_MARK = re.compile(
    r"\b(SKU|Model (No\.?|Number)|Part (No\.?|Number)|UPC|EAN|MPN|"
    r"Add to (cart|bag)|In stock|Out of stock|Free shipping|"
    r"Item\s?#|Ships? within)\b",
    re.I,
)
FAQ_HEADING = re.compile(r"^\s*\S[^\n?]{0,80}\?\s*$")
FIRST_PERSON_PLURAL = re.compile(r"\b(we|our|us)\b", re.I)


@dataclass(frozen=True)
class Signals:
    bullet_density: float
    heading_density: float
    faq_headings: int
    cta_count: int
    spec_density: float
    product_marks: int
    listicle_title: bool
    listicle_phrases: int
    we_our_rate: float

    @property
    def looks_listicle(self) -> bool:
        return self.listicle_title or self.bullet_density >= 0.25

    @property
    def looks_marketing(self) -> bool:
        return self.cta_count >= 2 or (self.we_our_rate > 0.02 and self.cta_count >= 1)

    @property
    def looks_templated_product(self) -> bool:
        return self.spec_density >= 0.15 or self.product_marks >= 2

    @property
    def looks_seo_structured(self) -> bool:
        return self.faq_headings >= 3 or (self.heading_density >= 0.08 and self.faq_headings >= 1)


def _is_heading(line: str) -> bool:
    words = line.split()
    if not 1 <= len(words) <= 12:
        return False
    if line.rstrip().endswith((".", "!", "?", ",", ";", ":")):
        return False
    return line[:1].isupper()


def signals(text: str, title: str = "") -> Signals:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    n_lines = max(1, len(lines))
    words = text.split()
    n_words = max(1, len(words))
    return Signals(
        bullet_density=sum(bool(BULLET_LINE.match(ln)) for ln in lines) / n_lines,
        heading_density=sum(_is_heading(ln) for ln in lines) / n_lines,
        faq_headings=sum(bool(FAQ_HEADING.match(ln)) for ln in lines),
        cta_count=len(CTA.findall(text)),
        spec_density=sum(bool(SPEC_LINE.match(ln)) for ln in lines) / n_lines,
        product_marks=len(PRODUCT_MARK.findall(text)),
        listicle_title=bool(LISTICLE_TITLE.match(title or (lines[0] if lines else ""))),
        listicle_phrases=len(LISTICLE_PHRASE.findall(text)),
        we_our_rate=len(FIRST_PERSON_PLURAL.findall(text)) / n_words,
    )


# Text-only genre assignment, for documents no URL rule matched.
#
# Text-only is a requirement, not a convenience. The AI side has no URL at all, so if
# human genre came from URLs and AI genre came from prompts, the labelling *process*
# would carry the class and scope.md 4.4's auxiliary head would amplify that leak
# rather than suppress it. The same function runs over both sides.
FIRST_PERSON_SINGULAR = re.compile(r"\b(i|me|my|mine|myself)\b", re.I)
CITATION = re.compile(r"\(\d{4}\)|\[\d{1,3}\]|\bet al\.|\bdoi:", re.I)
CODE_MARK = re.compile(r"`[^`]+`|^\s{4}\S|\b(function|import|def|class|npm|pip|sudo)\b", re.M)
ENCYCLOPEDIC = re.compile(
    r"\b(is a|was a|refers to|is an?\s+\w+\s+(that|which))\b.{0,80}\b(born|located|founded"
    r"|established|genus|species|term|concept)\b",
    re.I,
)
# A newswire dateline: a place, a date, then a dash. The place often carries commas
# ("ACME CORP, London, March 3, 2021 -"), so only the date and dash are pinned.
DATELINE = re.compile(
    r"^[^\n]{0,70}\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+"
    r"\d{1,2},?\s+\d{4}\s*[-\u2013\u2014]",
    re.M,
)
ABOUT_BOILERPLATE = re.compile(r"^\s*About [A-Z][\w &.]{2,40}:?\s*$", re.M)


def genre_from_text(text: str, title: str = "") -> tuple[str, float]:
    """Assign one of the eight genres from the text alone.

    Returns (genre, confidence). Order matters: the most specific signals are tested
    first, and the fallback is the generic web-prose bucket rather than a guess.
    """
    shape = signals(text, title)
    words = text.split()
    n_words = max(1, len(words))

    if (DATELINE.search(text) or ABOUT_BOILERPLATE.search(text)) and shape.we_our_rate > 0.004:
        return "press_release", 0.7
    if shape.looks_templated_product:
        return "product_marketing", 0.7
    if shape.looks_marketing or (shape.looks_listicle and shape.cta_count):
        return "product_marketing", 0.6
    if len(CITATION.findall(text)) >= 3:
        return "academic_formal", 0.7
    if len(CODE_MARK.findall(text)) >= 3:
        return "technical_docs", 0.6
    if ENCYCLOPEDIC.search(text[:400]) and shape.we_our_rate < 0.002:
        return "encyclopedia", 0.6
    first_person = len(FIRST_PERSON_SINGULAR.findall(text)) / n_words
    # A forum comment is short *and* conversational; a short first-person passage that
    # addresses nobody is far more likely to be a blog excerpt.
    conversational = "?" in text or re.search(r"(you|your|anyone|thanks|edit:)", text, re.I)
    if first_person > 0.015 and n_words < 250 and conversational:
        return "forum_comment", 0.6
    if first_person > 0.008:
        return "blog_personal", 0.6
    if shape.looks_seo_structured or shape.looks_listicle:
        return "product_marketing", 0.5
    return "blog_personal", 0.3
