"""Reference implementation of scope.md 8: chunk logits -> highlighted runs.

This is a coupling point. The extension ports it and fixtures/aggregate.json pins the
expected output of both. Every ambiguity in scope.md 8 is resolved here, and the fixture
is the specification -- if you disagree with a choice, change it here and regenerate,
don't reinterpret it in TypeScript.

Resolved ambiguities:
  - "shrunk toward 0.5 by words/80" is linear in probability space, capped at 1.
  - "fewer than 20% of chunks exceed t_off" uses a strict >.
  - the document-prior bump is applied in log-odds space (see calibration.py).
  - a run's score is the unweighted mean of its chunks' log-odds, after the length
    penalty, converted back to a probability.
  - scope.md 8's "hysteresis" is spatial, not temporal: a run opens at a chunk >= t_on
    and extends through neighbours >= t_off. There is no state carried between calls, so
    re-scoring a page always gives the same answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .calibration import Calibration, logit, sigmoid


@dataclass(frozen=True)
class Chunk:
    logit: float
    words: int


@dataclass(frozen=True)
class Run:
    start: int  # index of the first chunk, inclusive
    end: int  # index of the last chunk, inclusive
    words: int
    score: float
    flagged: bool


@dataclass(frozen=True)
class AggregateResult:
    t_on_effective: float
    chunk_p: list[float]  # after temperature, before the length penalty
    chunk_p_penalized: list[float]  # what the thresholds are compared against
    runs: list[Run]

    @property
    def flagged_runs(self) -> list[Run]:
        return [r for r in self.runs if r.flagged]


def apply_length_penalty(p: float, words: int, pivot: int) -> float:
    """Shrink short chunks toward 0.5. Detectors are near chance on short spans."""
    if words >= pivot:
        return p
    return 0.5 + (p - 0.5) * (words / pivot)


def aggregate(chunks: list[Chunk], cal: Calibration) -> AggregateResult:
    params = cal.aggregate
    chunk_p = [sigmoid(c.logit / cal.temperature) for c in chunks]
    penalized = [
        apply_length_penalty(p, c.words, params.length_penalty_words)
        for p, c in zip(chunk_p, chunks, strict=True)
    ]

    # Document prior: on a page with little flagged material, demand more confidence.
    t_on = cal.t_on
    if chunks:
        above = sum(1 for p in penalized if p > cal.t_off)
        if above < params.doc_prior_min_fraction * len(chunks):
            t_on = sigmoid(logit(cal.t_on) + params.doc_prior_bump_logodds)

    runs: list[Run] = []
    i = 0
    while i < len(penalized):
        if penalized[i] < t_on:
            i += 1
            continue
        # A run opens here and extends over neighbours that clear the lower threshold.
        start = i
        end = i
        while end + 1 < len(penalized) and penalized[end + 1] >= cal.t_off:
            end += 1
        while start - 1 >= 0 and penalized[start - 1] >= cal.t_off:
            if runs and start - 1 <= runs[-1].end:
                break  # don't absorb chunks already claimed by the previous run
            start -= 1
        words = sum(c.words for c in chunks[start : end + 1])
        mean_logodds = sum(logit(p) for p in penalized[start : end + 1]) / (end - start + 1)
        runs.append(
            Run(
                start=start,
                end=end,
                words=words,
                score=sigmoid(mean_logodds),
                flagged=words >= params.run_min_words,
            )
        )
        i = end + 1

    return AggregateResult(
        t_on_effective=t_on,
        chunk_p=chunk_p,
        chunk_p_penalized=penalized,
        runs=runs,
    )
