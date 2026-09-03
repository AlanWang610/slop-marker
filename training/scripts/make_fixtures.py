"""Generate the cross-language parity fixtures in fixtures/.

Run locally, no torch needed:  uv run python training/scripts/make_fixtures.py

Fixtures are generated once and committed. Both training/tests/ and extension/tests/
read them; neither generates them at test time. Cases are curated by hand here rather
than sampled, so every one is hand-checkable and pins a specific known hazard. Hazard
characters are written as \\u escapes so the source stays readable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "src"))

from slopmarker.data.chunking import chunk_block  # noqa: E402
from slopmarker.data.normalize import (  # noqa: E402
    collapse_whitespace,
    content_hash,
    count_words,
    normalize_for_hash,
)
from slopmarker.data.sentences import sentences  # noqa: E402

# (name, input). Each case pins one hazard.
NORMALIZE_CASES: list[tuple[str, str]] = [
    ("plain", "The quick brown fox."),
    ("leading-trailing-space", "   padded text   "),
    ("tab-and-newline", "line one\tand\nline two"),
    ("collapse-runs", "many     spaces    here"),
    ("crlf", "line one\r\nline two"),
    ("nbsp", "Al dente tips"),
    ("zero-width-space", "hid​den"),
    ("soft-hyphen", "encyc­lopedia"),
    ("bom-midstring", "before﻿after"),
    ("word-joiner", "a⁠b"),
    ("en-quad-em-quad", "a b c"),
    ("thin-and-hair-space", "a b c"),
    ("figure-and-punctuation-space", "1 2 3"),
    ("narrow-nbsp", "12 000"),
    ("medium-math-space", "a b"),
    ("ideographic-space", "a　b"),
    ("line-and-paragraph-separator", "a b c"),
    ("ogham-space", "a b"),
    ("vertical-tab-formfeed", "abc"),
    ("curly-single-quotes", "‘quoted’ and it’s"),
    ("curly-double-quotes", "“quoted” and „low‟"),
    ("prime-marks", "5′ and 6″"),
    ("all-dashes", "a‐b‑c‒d–e—f―g−h"),
    ("ellipsis", "wait… what"),
    ("nfc-decomposed", "café résumé"),
    ("nfc-already-composed", "café résumé"),
    ("emoji-zwj-survives", "family \U0001f468‍\U0001f469‍\U0001f467 here"),
    ("zwnj-survives", "क्‍ष"),
    ("cjk-untouched", "中文测试"),
    ("empty", ""),
    ("whitespace-only", " \t\n "),
    (
        "scraped-paragraph",
        "  The report — released Tuesday — said​ costs rose 3.5%.  ",
    ),
]

SENTENCE_CASES: list[tuple[str, str]] = [
    ("simple-pair", "First sentence here. Second sentence here."),
    ("abbreviation-titles", "Dr. Smith visited St. Louis in Jan. 2021. He liked it."),
    ("initials", "J. R. R. Tolkien wrote books. They were long."),
    ("quote-ends-sentence", 'He said "Hello." Then he left.'),
    ("url-not-a-boundary", "Visit example.com. Next sentence here."),
    ("decimal-not-a-boundary", "Pi is 3.14 exactly. Or close."),
    ("multi-terminator", "What?! Really. Yes... I think so."),
    ("no-terminator", "One sentence only"),
    ("etc-abbreviation", "We sell apples, pears, etc. They are fresh."),
    ("eg-abbreviation", "Use a fruit, e.g. an apple. Then eat it."),
    ("lowercase-after-period-not-boundary", "See Fig. 3 for details. Then continue."),
    ("digit-starts-sentence", "Costs rose. 2021 was worse."),
    ("open-paren-starts-sentence", "That is done. (Mostly, anyway.)"),
    ("trailing-space", "Done. "),
    ("company-suffix", "He joined Acme Inc. She did not."),
]

# (name, input, min_words, max_words)
CHUNK_CASES: list[tuple[str, str, int, int]] = [
    ("under-min-words-skipped", "Too short to score at all.", 40, 400),
    ("exactly-at-min-words", " ".join(f"word{i}" for i in range(40)) + ".", 40, 400),
    (
        "splits-at-sentence-boundary",
        " ".join(
            f"This is sentence number {i} and it carries a handful of extra words."
            for i in range(20)
        ),
        40,
        120,
    ),
    (
        "single-sentence-over-max-emitted-alone",
        " ".join(f"word{i}" for i in range(500)),
        40,
        400,
    ),
    (
        "short-trailing-chunk-kept",
        " ".join(f"Sentence {i} has several words in it right here." for i in range(14)),
        40,
        100,
    ),
    (
        "whitespace-normalized-before-chunking",
        "  Lots   of\tirregular whitespace " + " ".join(f"filler{i}" for i in range(50)) + ".",
        40,
        400,
    ),
]


def build_normalize() -> dict[str, object]:
    return {
        "version": 1,
        "note": "collapsed = model input; normalized = hash input; sha256 = sha256(normalized)",
        "cases": [
            {
                "name": name,
                "input": text,
                "collapsed": collapse_whitespace(text),
                "normalized": normalize_for_hash(text),
                "sha256": content_hash(text),
                "words": count_words(text),
            }
            for name, text in NORMALIZE_CASES
        ],
    }


def build_windows() -> dict[str, object]:
    return {
        "version": 1,
        "note": "sentence_cases pin split_sentences; chunk_cases pin chunk_block",
        "sentence_cases": [
            {"name": name, "input": text, "sentences": sentences(text)}
            for name, text in SENTENCE_CASES
        ],
        "chunk_cases": [
            {
                "name": name,
                "input": text,
                "min_words": lo,
                "max_words": hi,
                "chunks": [{"text": c, "words": count_words(c)} for c in chunk_block(text, lo, hi)],
            }
            for name, text, lo, hi in CHUNK_CASES
        ],
    }


def write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    out = ROOT / "fixtures"
    write(out / "normalize.json", build_normalize())
    write(out / "windows.json", build_windows())


if __name__ == "__main__":
    main()
