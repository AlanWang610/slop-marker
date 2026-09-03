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
