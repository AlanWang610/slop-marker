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
from slopmarker.eval.aggregate import Chunk, aggregate  # noqa: E402
from slopmarker.eval.calibration import AggregateParams, Calibration  # noqa: E402

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


# Deliberately synthetic, NOT the shipped calibration. If these were the real constants,
# every recalibration would invalidate the extension's test suite for no reason.
SYNTHETIC = Calibration(
    version="fixture-only",
    temperature=1.25,
    t_on=0.90,
    t_off=0.80,
    min_words=40,
    aggregate=AggregateParams(
        length_penalty_words=80,
        run_min_words=150,
        doc_prior_min_fraction=0.20,
        doc_prior_bump_logodds=0.25,
    ),
)
# t_on close to 1.0, to pin the case where scope.md 8's probability-space bump would
# have produced an unreachable threshold above 1.0.
HIGH_T_ON = Calibration(
    version="fixture-only-high",
    temperature=1.0,
    t_on=0.97,
    t_off=0.87,
    min_words=40,
)

# (name, calibration, chunks as (logit, words))
AGGREGATE_CASES: list[tuple[str, Calibration, list[tuple[float, int]]]] = [
    ("empty", SYNTHETIC, []),
    ("all-below-t-off", SYNTHETIC, [(-3.0, 200), (-2.5, 200), (-4.0, 200)]),
    ("single-short-chunk-above-t-on-not-flagged", SYNTHETIC, [(4.0, 60)]),
    ("single-long-chunk-under-run-minimum", SYNTHETIC, [(4.0, 140)]),
    ("single-long-chunk-at-run-minimum", SYNTHETIC, [(4.0, 150)]),
    ("two-chunks-reach-run-minimum", SYNTHETIC, [(4.0, 90), (3.5, 90)]),
    (
        "mid-band-chunk-extends-run",
        SYNTHETIC,
        # logit 2.2 at T=1.25 -> p=0.853, between t_off (0.80) and t_on (0.90):
        # too low to open a run, high enough to keep one going.
        [(4.0, 100), (2.2, 100), (4.0, 100), (-3.0, 100)],
    ),
    (
        "below-t-off-chunk-breaks-run",
        SYNTHETIC,
        [(4.0, 100), (-3.0, 100), (4.0, 100)],
    ),
    # 1 of 10 chunks above t_off -> 1 < 2.0 -> prior fires.
    ("doc-prior-fires", SYNTHETIC, [(4.0, 200)] + [(-4.0, 200)] * 9),
    # 2 of 10 above t_off -> 2 < 2.0 is false -> prior does not fire.
    ("doc-prior-not-firing-at-exactly-20-percent", SYNTHETIC, [(4.0, 200)] * 2 + [(-4.0, 200)] * 8),
    # The regression case: t_on 0.97 + prior bump must stay inside (0, 1).
    ("doc-prior-bump-stays-below-one", HIGH_T_ON, [(6.0, 200)] + [(-6.0, 200)] * 9),
    ("length-penalty-cannot-flag-short-chunk-alone", SYNTHETIC, [(8.0, 40)]),
]


def build_aggregate() -> dict[str, object]:
    cases = []
    for name, cal, raw in AGGREGATE_CASES:
        chunks = [Chunk(logit=lg, words=w) for lg, w in raw]
        result = aggregate(chunks, cal)
        cases.append(
            {
                "name": name,
                "calibration": json.loads(cal.to_json()),
                "chunks": [{"logit": c.logit, "words": c.words} for c in chunks],
                "expected": {
                    "t_on_effective": result.t_on_effective,
                    "chunk_p": result.chunk_p,
                    "chunk_p_penalized": result.chunk_p_penalized,
                    "runs": [
                        {
                            "start": r.start,
                            "end": r.end,
                            "words": r.words,
                            "score": r.score,
                            "flagged": r.flagged,
                        }
                        for r in result.runs
                    ],
                },
            }
        )
    return {
        "version": 1,
        "note": "calibration constants here are synthetic, never the shipped ones",
        "cases": cases,
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
    write(out / "aggregate.json", build_aggregate())


if __name__ == "__main__":
    main()
