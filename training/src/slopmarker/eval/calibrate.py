"""Temperature scaling and threshold selection (scope.md 4.5).

Three things about this are worth stating plainly, because the spec implies otherwise.

**Temperature barely affects the decision rule.** It is a one-parameter monotone
transform of a single logit, so it cannot change ranking, and therefore cannot change
the false-positive rate at a threshold chosen to hit a target FPR. Its real jobs are the
tooltip number the extension shows and the length penalty, which is not
monotone-preserving across chunks of different lengths. Fit it, ship it, but do not
believe it is setting the operating point.

**The per-genre bound is what binds.** A single global threshold can always satisfy it,
because per-genre FPR is monotone in the threshold -- there is no infeasibility here,
only cost. The cost is that the worst genre sets the operating point for every genre,
and since the genre head is dropped at export the extension cannot do better. So the
per-genre *oracle* recall is computed alongside, and the gap between them is the
measured price of that design decision rather than an opinion about it.

**Select with margin, verify separately.** A threshold at the 98th percentile of a few
thousand windows rests on very few tail points. Selecting exactly at the target means
missing it about half the time on fresh data, so selection uses a margin and the test
split verifies with a Clopper-Pearson upper bound.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import (
    auroc,
    clopper_pearson_upper,
    fpr_at_threshold,
    recall_at_threshold,
    split_by_label,
    threshold_at_fpr,
)

OVERALL_FPR_TARGET = 0.01
PER_GENRE_FPR_TARGET = 0.02
SELECTION_MARGIN = 0.7  # select at 70% of target, verify at target


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Fit T minimising soft-target BCE of sigmoid(logit / T). Scalar, so a 1-D scan."""
    finite = np.isfinite(logits)
    logits, targets = logits[finite], targets[finite]
    if logits.size == 0:
        return 1.0

    def nll(temperature: float) -> float:
        z = logits / temperature
        # log(1 + exp(z)) computed stably.
        softplus = np.logaddexp(0.0, z)
        return float(np.mean(softplus - targets * z))

    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
    best = min(grid, key=nll)
    # Refine locally around the grid winner.
    fine = np.linspace(best * 0.8, best * 1.25, 200)
    return float(min(fine, key=nll))


@dataclass
class GenreReport:
    genre: str
    n_human: int
    n_ai: int
    threshold: float
    fpr: float
    fpr_upper: float
    recall: float
    auc: float


@dataclass
class Calibrated:
    temperature: float
    t_on: float
    binding_genre: str
    overall_fpr: float
    recall: float
    oracle_recall: float
    per_genre: list[GenreReport]

    @property
    def oracle_gap(self) -> float:
        """Recall given up by shipping one global threshold instead of per-genre ones.

        This is the measured price of dropping the genre head at export.
        """
        return self.oracle_recall - self.recall


def _by_genre(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["genre"]].append(row)
    return grouped


def calibrate(
    rows: list[dict[str, Any]],
    *,
    overall_target: float = OVERALL_FPR_TARGET,
    genre_target: float = PER_GENRE_FPR_TARGET,
    margin: float = SELECTION_MARGIN,
) -> Calibrated:
    """Fit temperature and choose the global operating threshold."""
    logits = np.array([r["logit"] for r in rows], dtype=np.float64)
    fractions = np.array([r["ai_fraction"] for r in rows], dtype=np.float64)
    temperature = fit_temperature(logits, fractions)
    probabilities = 1.0 / (1.0 + np.exp(-logits / temperature))

    human, ai = split_by_label(probabilities, fractions)
    thresholds = {"__overall__": threshold_at_fpr(human, overall_target * margin)}

    per_genre: list[GenreReport] = []
    for genre, group in sorted(_by_genre(rows).items()):
        p = np.array(
            [1.0 / (1.0 + np.exp(-r["logit"] / temperature)) for r in group], dtype=np.float64
        )
        f = np.array([r["ai_fraction"] for r in group], dtype=np.float64)
        g_human, g_ai = split_by_label(p, f)
        if g_human.size < 20:
            continue
        thresholds[genre] = threshold_at_fpr(g_human, genre_target * margin)
        per_genre.append(
            GenreReport(
                genre=genre,
                n_human=int(g_human.size),
                n_ai=int(g_ai.size),
                threshold=thresholds[genre],
                fpr=0.0,
                fpr_upper=0.0,
                recall=0.0,
                auc=auroc(g_human, g_ai) if g_ai.size else float("nan"),
            )
        )

    # The worst genre sets the operating point for every genre.
    binding = max(thresholds, key=lambda k: thresholds[k])
    t_on = thresholds[binding]

    # Recompute each genre's realised numbers at the shipped threshold, and the recall
    # a per-genre oracle would have achieved.
    oracle_hits = oracle_total = 0
    finished: list[GenreReport] = []
    for report in per_genre:
        group = _by_genre(rows)[report.genre]
        p = np.array(
            [1.0 / (1.0 + np.exp(-r["logit"] / temperature)) for r in group], dtype=np.float64
        )
        f = np.array([r["ai_fraction"] for r in group], dtype=np.float64)
        g_human, g_ai = split_by_label(p, f)
        false_positives = int((g_human >= t_on).sum())
        finished.append(
            GenreReport(
                genre=report.genre,
                n_human=int(g_human.size),
                n_ai=int(g_ai.size),
                threshold=report.threshold,
                fpr=fpr_at_threshold(g_human, t_on),
                fpr_upper=clopper_pearson_upper(false_positives, int(g_human.size)),
                recall=recall_at_threshold(g_ai, t_on),
                auc=report.auc,
            )
        )
        if g_ai.size:
            oracle_hits += int((g_ai >= report.threshold).sum())
            oracle_total += int(g_ai.size)

    return Calibrated(
        temperature=temperature,
        t_on=float(t_on),
        binding_genre=binding,
        overall_fpr=fpr_at_threshold(human, t_on),
        recall=recall_at_threshold(ai, t_on),
        oracle_recall=(oracle_hits / oracle_total) if oracle_total else 0.0,
        per_genre=finished,
    )


def verify(rows: list[dict[str, Any]], temperature: float, t_on: float) -> dict[str, Any]:
    """Check a chosen threshold on a split that was not used to choose it."""
    logits = np.array([r["logit"] for r in rows], dtype=np.float64)
    fractions = np.array([r["ai_fraction"] for r in rows], dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-logits / temperature))
    human, ai = split_by_label(probabilities, fractions)

    genres = {}
    worst = 0.0
    for genre, group in sorted(_by_genre(rows).items()):
        p = np.array(
            [1.0 / (1.0 + np.exp(-r["logit"] / temperature)) for r in group], dtype=np.float64
        )
        f = np.array([r["ai_fraction"] for r in group], dtype=np.float64)
        g_human, g_ai = split_by_label(p, f)
        if g_human.size < 20:
            continue
        false_positives = int((g_human >= t_on).sum())
        upper = clopper_pearson_upper(false_positives, int(g_human.size))
        genres[genre] = {
            "n_human": int(g_human.size),
            "fpr": fpr_at_threshold(g_human, t_on),
            "fpr_upper": upper,
            "recall": recall_at_threshold(g_ai, t_on) if g_ai.size else None,
        }
        worst = max(worst, upper)

    return {
        "overall_fpr": fpr_at_threshold(human, t_on),
        "recall": recall_at_threshold(ai, t_on),
        "auroc": auroc(human, ai) if human.size and ai.size else float("nan"),
        "worst_genre_fpr_upper": worst,
        "per_genre": genres,
    }
