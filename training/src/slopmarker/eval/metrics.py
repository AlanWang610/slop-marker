"""Metrics for a high-precision detector (scope.md 4.5).

Hand-rolled in numpy rather than pulled from sklearn, for two reasons: the production
path stays dependency-light, and the definitions here are the ones the release gates
turn on, so they should be readable rather than inherited. Tests cross-check each
against sklearn on synthetic data with analytically known answers.

Label conventions, because scope.md leaves them ambiguous and it matters:

  human (the FPR denominator)  ai_fraction == 0.0 -- pure human, no AI editing
  AI (the recall denominator)  ai_fraction >= 0.7
  mid-band                     excluded from both, reported as a coverage curve

Counting a 60%-AI-rewritten document as a false positive would inflate FPR and push the
threshold up, costing real recall for no benefit.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

HUMAN_MAX_FRACTION = 0.0
AI_MIN_FRACTION = 0.7


def split_by_label(scores: np.ndarray, fractions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (human scores, AI scores), dropping the mid-band."""
    return scores[fractions <= HUMAN_MAX_FRACTION], scores[fractions >= AI_MIN_FRACTION]


def threshold_at_recall(ai_scores: np.ndarray, target_recall: float) -> float:
    """The largest threshold whose recall is still at least `target_recall`."""
    if ai_scores.size == 0:
        return float("inf")
    return float(np.quantile(ai_scores, 1.0 - target_recall, method="lower"))


def threshold_at_fpr(human_scores: np.ndarray, target_fpr: float) -> float:
    """The smallest threshold whose FPR is at most `target_fpr`.

    `method="higher"` is the conservative choice: with few tail points it errs toward a
    higher threshold, and therefore toward fewer false positives.
    """
    if human_scores.size == 0:
        return float("-inf")
    return float(np.quantile(human_scores, 1.0 - target_fpr, method="higher"))


def fpr_at_threshold(human_scores: np.ndarray, threshold: float) -> float:
    if human_scores.size == 0:
        return 0.0
    return float((human_scores >= threshold).mean())


def recall_at_threshold(ai_scores: np.ndarray, threshold: float) -> float:
    if ai_scores.size == 0:
        return 0.0
    return float((ai_scores >= threshold).mean())


def fpr_at_recall(
    human_scores: np.ndarray, ai_scores: np.ndarray, target_recall: float
) -> tuple[float, float]:
    threshold = threshold_at_recall(ai_scores, target_recall)
    return fpr_at_threshold(human_scores, threshold), threshold


def auroc(human_scores: np.ndarray, ai_scores: np.ndarray) -> float:
    """Rank-based AUROC, ties counted as half."""
    if human_scores.size == 0 or ai_scores.size == 0:
        return float("nan")
    combined = np.concatenate([human_scores, ai_scores])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty(combined.size, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1)
    # Average ranks within tied groups so ties contribute 0.5.
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]

    n_ai = ai_scores.size
    rank_sum = ranks[human_scores.size :].sum()
    return float((rank_sum - n_ai * (n_ai + 1) / 2) / (n_ai * human_scores.size))


def roc_curve(human_scores: np.ndarray, ai_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(fpr, tpr) at every distinct threshold, both ascending."""
    thresholds = np.unique(np.concatenate([human_scores, ai_scores]))[::-1]
    fpr = np.array([fpr_at_threshold(human_scores, t) for t in thresholds])
    tpr = np.array([recall_at_threshold(ai_scores, t) for t in thresholds])
    return np.concatenate([[0.0], fpr]), np.concatenate([[0.0], tpr])


def pauc(human_scores: np.ndarray, ai_scores: np.ndarray, max_fpr: float = 0.02) -> float:
    """AUROC restricted to FPR in [0, max_fpr], McClish-standardised to [0, 1].

    This, not AUROC, is the model-selection metric. AUROC integrates over regions of
    the curve this product will never operate in, and two checkpoints with identical
    AUROC can differ by many recall points at 1% FPR.
    """
    if human_scores.size == 0 or ai_scores.size == 0:
        return float("nan")
    fpr, tpr = roc_curve(human_scores, ai_scores)
    keep = fpr <= max_fpr
    fpr_c, tpr_c = fpr[keep], tpr[keep]
    if fpr_c.size < 2:
        return 0.5
    if fpr_c[-1] < max_fpr:  # extend flat to the boundary
        fpr_c = np.append(fpr_c, max_fpr)
        tpr_c = np.append(tpr_c, tpr_c[-1])
    area = float(np.trapezoid(tpr_c, fpr_c))
    # McClish: rescale so a random classifier is 0.5 and a perfect one 1.0.
    minimum = max_fpr**2 / 2
    maximum = max_fpr
    return 0.5 * (1 + (area - minimum) / (maximum - minimum))


def clopper_pearson_upper(successes: int, trials: int, alpha: float = 0.05) -> float:
    """Upper bound of the one-sided Clopper-Pearson interval.

    The release gate uses this rather than the point estimate. A threshold placed at
    the 98th percentile of a few thousand calibration windows rests on very few tail
    points, so the observed rate is not the rate to trust.
    """
    if trials == 0:
        return 1.0
    if successes >= trials:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(1 - alpha, successes + 1, trials - successes))


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in pairwise(edges):
        mask = (probabilities >= low) & (probabilities < high)
        if not mask.any():
            continue
        error += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(error)
