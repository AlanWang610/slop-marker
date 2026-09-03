"""English detection for the harvest.

Two different questions, easily conflated.

The *gate* asks whether a document is English at all. py3langid answers it: pure
Python, no model download, deterministic across platforms. Note that its module-level
`classify` returns a raw log-probability, not a confidence -- the identifier has to be
built with `norm_probs=True` to get a number in [0, 1].

The *non-native band* is a different question and py3langid cannot answer it. FineWeb
ships a fastText English confidence per row, and disfluent second-language English
scores measurably lower than native prose. So the band is read from FineWeb's own score
where it exists and is simply unavailable elsewhere, which is more honest than
inventing a proxy for a proxy.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _identifier():  # type: ignore[no-untyped-def]
    from py3langid.langid import MODEL_FILE, LanguageIdentifier

    return LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)


def detect(text: str) -> tuple[str, float]:
    """Return (language code, normalized confidence in [0, 1])."""
    if not text.strip():
        return "und", 0.0
    lang, score = _identifier().classify(text)
    return str(lang), float(score)


def is_english(text: str, min_score: float = 0.85) -> bool:
    lang, score = detect(text)
    return lang == "en" and score >= min_score


def in_l2_band(language_score: float | None, band: tuple[float, float]) -> bool:
    """Is a source-provided English confidence inside the non-native band?

    Only meaningful for a score from the source's own detector (FineWeb's fastText
    confidence). Returns False when the source carries no such score.
    """
    if language_score is None:
        return False
    low, high = band
    return low <= language_score < high
