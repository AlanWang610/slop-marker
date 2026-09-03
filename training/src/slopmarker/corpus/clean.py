"""Document cleaning and quality filtering.

Ordering matters: strip boilerplate first, then judge length and shape on what is left,
or navigation chrome inflates every count.

One rule governs this whole module and it is worth stating plainly. Nothing here may
reject a human document for *reading like AI*. That filter is the most tempting one
available and the most damaging: it would strip exactly the hard negatives scope.md 4.2
asks for, deflate the measured false-positive rate, and ship an optimistic threshold
into a product whose entire premise is that false positives are the failure to
minimize. `slop_lexicon_score` is computed for reporting only, and a test asserts no
filter reads it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..data.normalize import collapse_whitespace, count_words
from ..data.sentences import split_sentences
from .config import CleaningConfig

BOILERPLATE = re.compile(
    r"^\s*("
    r"home|about( us)?|contact( us)?|privacy policy|terms of (use|service)|"
    r"cookie policy|all rights reserved|copyright\s+(19|20)\d{2}|"
    r"share (this )?on \w+|\d+ comments?|read more|related (posts|articles)|"
    r"tags?:|posted (on|by)|filed under|next post|previous post|"
    r"skip to (main )?content|subscribe to our newsletter|follow us on"
    r")\b",
    re.I,
)
COPYRIGHT_MARK = re.compile(r"^\s*©")

# Phrases that correlate with LLM prose. Reported, never filtered on. See the module
# docstring for why that distinction is the important part.
SLOP_LEXICON = re.compile(
    r"\b(delve|tapestry|it'?s important to note|in today'?s fast[- ]paced|"
    r"navigate the complexities|a testament to|in conclusion|furthermore|"
    r"moreover|multifaceted|underscores the|plays a (crucial|vital|pivotal) role|"
    r"landscape of|realm of|embark on)\b",
    re.I,
)


@dataclass(frozen=True)
class CleanResult:
    text: str
    n_words: int
    n_sentences: int
    kept: bool
    reason: str = ""


def strip_boilerplate(text: str, repeated_lines: frozenset[str] = frozenset()) -> str:
    """Drop navigation and footer lines.

    `repeated_lines` comes from a per-host pass over line frequencies: any line that
    appears near the top or bottom of many documents from the same host is chrome. That
    catches per-host furniture no static pattern will.
    """
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in repeated_lines:
            continue
        if BOILERPLATE.match(stripped) or COPYRIGHT_MARK.match(stripped):
            continue
        kept.append(line)
    return "\n".join(kept)


def slop_lexicon_score(text: str) -> float:
    """Diagnostic only. Rate of LLM-associated phrases per 1000 words."""
    words = max(1, count_words(text))
    return 1000.0 * len(SLOP_LEXICON.findall(text)) / words


def quality_reason(text: str, cfg: CleaningConfig) -> str:
    """Return the reason a document fails the quality bar, or '' if it passes.

    These are shape checks -- symbol ratios, duplicate lines, truncated lines -- of the
    kind that catch scraped junk. They say nothing about who wrote the text.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "empty"

    words = text.split()
    if not words:
        return "empty"

    alpha_bearing = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_bearing / len(words) < 0.8:
        return "few_alpha_words"

    mean_word_length = sum(len(w) for w in words) / len(words)
    if not 3.0 <= mean_word_length <= 10.0:
        return "mean_word_length"

    symbols = sum(text.count(c) for c in "#<>{}[]|\\^~")
    if symbols / len(words) > 0.1:
        return "symbol_ratio"

    unique_lines = len(set(lines))
    if 1 - unique_lines / len(lines) > 0.3:
        return "duplicate_lines"

    ellipsis_terminated = sum(1 for ln in lines if ln.rstrip().endswith(("...", "…")))
    if ellipsis_terminated / len(lines) > 0.3:
        return "truncated_lines"

    return ""


def clean_document(
    text: str,
    cfg: CleaningConfig,
    *,
    repeated_lines: frozenset[str] = frozenset(),
    check_quality: bool = True,
    check_language: bool = True,
) -> CleanResult:
    """Clean one document and decide whether to keep it.

    `check_quality` is off for FineWeb, which has already run C4 and Gopher filters;
    running them twice costs time and removes nothing.
    """
    text = strip_boilerplate(text, repeated_lines)
    collapsed = collapse_whitespace(text)
    n_words = count_words(collapsed)

    if n_words < cfg.min_words:
        return CleanResult(text, n_words, 0, False, "too_short")
    if n_words > cfg.max_words:
        # Long documents are truncated, not dropped: the head of a long article is
        # perfectly good training material.
        text = _truncate_to_words(collapsed, cfg.max_words)
        collapsed = text
        n_words = count_words(collapsed)

    n_sentences = len(split_sentences(collapsed))
    if n_sentences < cfg.min_sentences:
        return CleanResult(text, n_words, n_sentences, False, "too_few_sentences")

    if check_quality:
        reason = quality_reason(text, cfg)
        if reason:
            return CleanResult(text, n_words, n_sentences, False, reason)

    if check_language:
        from .lang import is_english

        if not is_english(collapsed, cfg.min_language_score):
            return CleanResult(text, n_words, n_sentences, False, "not_english")

    return CleanResult(text, n_words, n_sentences, True)


def _truncate_to_words(text: str, max_words: int) -> str:
    """Cut at the last sentence boundary that stays under the word cap."""
    kept: list[str] = []
    total = 0
    for start, end in split_sentences(text):
        sentence = text[start:end]
        words = count_words(sentence)
        if total + words > max_words:
            break
        kept.append(sentence)
        total += words
    return " ".join(kept) if kept else " ".join(text.split(" ")[:max_words])


def find_repeated_lines(
    documents: list[str], *, edge_lines: int = 5, min_hosts: int = 50
) -> frozenset[str]:
    """Lines appearing at the head or tail of many documents from one host.

    Run per host over its documents. This is what removes per-site chrome that no
    global regex anticipates.
    """
    counts: dict[str, int] = {}
    for doc in documents:
        lines = [ln.strip() for ln in doc.splitlines() if ln.strip()]
        edges = lines[:edge_lines] + lines[-edge_lines:]
        for line in set(edges):
            counts[line] = counts.get(line, 0) + 1
    return frozenset(line for line, n in counts.items() if n >= min_hosts)
