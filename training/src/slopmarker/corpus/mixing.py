"""Mixed-authorship documents with an honest AI fraction (scope.md 4.2).

The rule: never ask a model to edit a fraction. A model told to "rewrite 30% of this"
will not comply, and the resulting label would be a wish rather than a measurement.

Instead a human document and its AI rewrite -- same facts, same order of ideas, which is
why the rewrite prompt style exists -- are aligned into blocks, a subset of blocks is
taken from the AI side to approach a target fraction, and then the fraction that
actually resulted is *measured* from the emitted text. The target is recorded as
provenance; the label is the measurement.

Granularity is paragraph-first because whole-paragraph swaps have no seams inside them.
Scattered sentence swaps leave detectable joins -- dangling pronouns, entities
reintroduced by full name mid-document -- and a classifier can learn "seam implies
mixed", which is not a property of AI text at all.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from ..data.normalize import collapse_whitespace
from ..data.sentences import split_sentences

Granularity = Literal["paragraph", "prefix_suffix", "sentence"]

# Anaphora and discourse connectives. Cutting immediately before one of these leaves a
# sentence whose referent has just been replaced.
_UNSAFE_OPENERS = frozenset(
    """
    it its it's they them their theirs this that these those he she his her him hers
    however therefore thus moreover furthermore additionally consequently instead
    meanwhile nonetheless nevertheless also then so besides
    """.split()  # noqa: SIM905 - a word list reads better than a literal
)


@dataclass(frozen=True)
class MixedDocument:
    text: str
    ai_spans: list[tuple[int, int]]
    ai_fraction: float  # measured, not requested
    ai_fraction_target: float
    granularity: Granularity
    blocks_from_ai: int
    blocks_total: int


def _paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return parts if len(parts) > 1 else _sentence_blocks(text)


def _sentence_blocks(text: str, per_block: int = 3) -> list[str]:
    sentences = [text[a:b] for a, b in split_sentences(text)]
    return [
        " ".join(sentences[i : i + per_block]) for i in range(0, len(sentences), per_block)
    ] or [text]


def _align(human: list[str], ai: list[str]) -> list[tuple[str, str]]:
    """Pair human and AI blocks monotonically.

    The rewrite prompt preserves the order of ideas, so position is a good alignment.
    Where counts differ, blocks are matched proportionally rather than truncated, so no
    content is silently dropped from either side.
    """
    if not human or not ai:
        return []
    n = max(len(human), len(ai))
    pairs = []
    for i in range(n):
        pairs.append(
            (
                human[min(i * len(human) // n, len(human) - 1)],
                ai[min(i * len(ai) // n, len(ai) - 1)],
            )
        )
    # Collapse consecutive duplicates introduced by the proportional mapping.
    deduped: list[tuple[str, str]] = []
    for pair in pairs:
        if not deduped or pair != deduped[-1]:
            deduped.append(pair)
    return deduped


def _safe_to_cut_before(block: str) -> bool:
    words = collapse_whitespace(block).split(" ")
    return not any(w.strip(".,;:").lower() in _UNSAFE_OPENERS for w in words[:1])


def splice(
    human_text: str,
    ai_text: str,
    target_fraction: float,
    *,
    granularity: Granularity = "paragraph",
    seed: int = 0,
) -> MixedDocument | None:
    """Build a mixed document and measure what fraction of it is AI-authored."""
    human_blocks = _paragraphs(human_text)
    ai_blocks = _paragraphs(ai_text)
    pairs = _align(human_blocks, ai_blocks)
    if len(pairs) < 2:
        return None

    rng = random.Random(seed)
    total = len(pairs)

    if granularity == "prefix_suffix":
        cut = max(1, min(total - 1, round(total * target_fraction)))
        # AI writes either the opening or the closing, not always the same end.
        opening = rng.random() < 0.5
        chosen = set(range(cut)) if opening else set(range(total - cut, total))
        # Only cut where the following block does not open with an anaphor.
        boundary = max(chosen) + 1 if chosen and max(chosen) + 1 < total else None
        if boundary is not None and not _safe_to_cut_before(pairs[boundary][0]):
            return None
    else:
        order = list(range(total))
        rng.shuffle(order)
        chosen = set()
        human_chars = sum(len(h) for h, _ in pairs)
        ai_chars = 0.0
        for index in order:
            if ai_chars / max(1, human_chars) >= target_fraction:
                break
            chosen.add(index)
            ai_chars += len(pairs[index][1])
            human_chars += len(pairs[index][1]) - len(pairs[index][0])

    out: list[str] = []
    spans: list[tuple[int, int]] = []
    position = 0
    for index, (human_block, ai_block) in enumerate(pairs):
        block = ai_block if index in chosen else human_block
        if out:
            position += 1  # the joining space
        if index in chosen:
            spans.append((position, position + len(block)))
        out.append(block)
        position += len(block)

    text = " ".join(out)
    merged = _merge_spans(spans)
    covered = sum(end - start for start, end in merged)
    return MixedDocument(
        text=text,
        ai_spans=merged,
        ai_fraction=covered / len(text) if text else 0.0,
        ai_fraction_target=target_fraction,
        granularity=granularity,
        blocks_from_ai=len(chosen),
        blocks_total=total,
    )


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


# scope.md 1 lists "detecting light AI-assisted editing" as a non-goal. That is only
# enforceable if the corpus contains documents at low fractions the model is trained to
# score low; without them the model will confidently flag lightly edited human prose.
FRACTION_BUCKETS: tuple[tuple[float, float, float], ...] = (
    (0.15, 0.85, 0.40),  # keeps the regression head non-degenerate across the range
    (0.50, 0.90, 0.30),  # mass at the 0.7 binarization boundary
    (0.85, 0.97, 0.15),  # heavily AI-rewritten human text: explicitly in scope
    (0.03, 0.15, 0.15),  # lightly AI-touched: explicitly a non-goal
)


def draw_target_fraction(rng: random.Random) -> float:
    draw = rng.random()
    running = 0.0
    for low, high, weight in FRACTION_BUCKETS:
        running += weight
        if draw < running:
            return rng.uniform(low, high)
    return rng.uniform(*FRACTION_BUCKETS[0][:2])
