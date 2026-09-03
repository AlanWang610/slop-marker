"""Strip generation-harness artifacts from AI output.

There are two categories here and conflating them ruins the corpus.

*Harness artifacts* are text the model produced about the task rather than as the
document: code fences, leaked thinking tags, "Certainly! Here's...", trailing offers to
help, meta-commentary about word counts. Humans do not write these. They are properties
of our pipeline, and a classifier that learns them has learned nothing transferable.
Remove them.

*Model style* is em-dash density, "In conclusion", markdown headings, hedging,
tricolons. These are properties of AI writing and they appear on the real web. Removing
them would make the corpus easier than reality -- a good validation number and a bad
production false-positive rate. Keep them. They are decorrelated from the label instead,
by mirroring each seed's structure (prompts.py) rather than by deletion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..data.normalize import count_words
from ..data.sentences import split_sentences

CODE_FENCE = re.compile(r"\A\s*```[a-zA-Z]*\n(.*?)\n?```\s*\Z", re.S)
THINKING = re.compile(r"<(thinking|think|reasoning|scratchpad|answer)>.*?</\1>", re.S | re.I)
CHANNEL_MARKER = re.compile(r"^\s*(analysis|final|commentary)\s*channel\s*:.*$", re.I | re.M)
PREAMBLE = re.compile(
    r"\A\s*(Sure|Certainly|Of course|Absolutely|Here(?:'s| is)|I'd be happy to|"
    r"Below is|Great question)\b[^\n]*\n+",
    re.I,
)
TITLE_LINE = re.compile(r"\A\s*(?:\*\*)?(?:Title|Headline)\s*:?\s*(?:\*\*)?[^\n]*\n+", re.I)
MARKDOWN_TITLE = re.compile(r"\A\s*#\s+[^\n]*\n+")
TRAILING_OFFER = re.compile(
    r"\n+\s*(Let me know|Feel free to|I hope this helps|Would you like me to|"
    r"If you'd like|Hope (?:this|that) helps)\b.*\Z",
    re.S | re.I,
)
META_SENTENCE = re.compile(
    r"\b(as an AI|I cannot|I'm unable to|word count|approximately \d+ words|"
    r"as requested|per your request|in this (?:article|piece), I will)\b",
    re.I,
)
REFUSAL = re.compile(r"\A\s*(I can't|I cannot|I'm sorry|I am sorry|I won't|I'm not able)\b", re.I)


@dataclass
class HygieneResult:
    text: str
    ops: list[str] = field(default_factory=list)
    rejected: str = ""

    @property
    def kept(self) -> bool:
        return not self.rejected


def _repetition_ratio(text: str, n: int = 8) -> float:
    """Share of n-grams that are repeats. Catches degenerate high-temperature output."""
    words = text.split()
    if len(words) < n * 2:
        return 0.0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def _truncate_to_words(text: str, max_words: int) -> str:
    kept: list[str] = []
    total = 0
    for start, end in split_sentences(text):
        sentence = text[start:end]
        words = count_words(sentence)
        if total and total + words > max_words:
            break
        kept.append(sentence)
        total += words
    return " ".join(kept) if kept else text


def clean_generation(
    text: str,
    *,
    target_words: int,
    seed_had_heading: bool = False,
    min_ratio: float = 0.6,
    max_ratio: float = 1.8,
) -> HygieneResult:
    """Remove harness artifacts and decide whether to keep the generation."""
    ops: list[str] = []

    if REFUSAL.match(text):
        return HygieneResult(text, ops, "refusal")

    fenced = CODE_FENCE.match(text)
    if fenced:
        text = fenced.group(1)
        ops.append("strip_code_fence")

    # Non-optional: models with thinking disabled still leak tags occasionally, and
    # open-weight reasoning models leak constantly. A leaked <thinking> block would be
    # the single strongest artifact in the corpus.
    if THINKING.search(text):
        text = THINKING.sub("", text)
        ops.append("strip_leaked_thinking")
    if CHANNEL_MARKER.search(text):
        text = CHANNEL_MARKER.sub("", text)
        ops.append("strip_channel_marker")

    for pattern, op in ((PREAMBLE, "strip_preamble"), (TITLE_LINE, "strip_title_line")):
        if pattern.match(text):
            text = pattern.sub("", text, count=1)
            ops.append(op)

    # A bare "# Title" is only an artifact if the seed had no heading of its own.
    if not seed_had_heading and MARKDOWN_TITLE.match(text):
        text = MARKDOWN_TITLE.sub("", text, count=1)
        ops.append("strip_wrapper_title")

    if TRAILING_OFFER.search(text):
        text = TRAILING_OFFER.sub("", text)
        ops.append("strip_trailing_offer")

    kept_sentences = []
    dropped_meta = False
    for start, end in split_sentences(text):
        sentence = text[start:end]
        if META_SENTENCE.search(sentence):
            dropped_meta = True
            continue
        kept_sentences.append(sentence)
    if dropped_meta:
        text = " ".join(kept_sentences)
        ops.append("strip_meta_sentences")

    text = text.strip()
    words = count_words(text)

    if words < target_words * min_ratio:
        return HygieneResult(text, ops, "too_short")
    if words > target_words * max_ratio:
        text = _truncate_to_words(text, int(target_words * max_ratio))
        ops.append("length_conform")
        words = count_words(text)
    if _repetition_ratio(text) > 0.3:
        return HygieneResult(text, ops, "degenerate_repetition")

    return HygieneResult(text, ops)
