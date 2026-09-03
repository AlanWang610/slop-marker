"""Metrics, cross-checked against sklearn and against analytically known answers."""

from __future__ import annotations

import numpy as np
import pytest

from slopmarker.eval.metrics import (
    auroc,
    clopper_pearson_upper,
    expected_calibration_error,
    fpr_at_recall,
    fpr_at_threshold,
    pauc,
    recall_at_threshold,
    split_by_label,
    threshold_at_fpr,
    threshold_at_recall,
)

RNG = np.random.default_rng(20260903)


def separated(n: int = 4000, gap: float = 2.0):
    return RNG.normal(0.0, 1.0, n), RNG.normal(gap, 1.0, n)


class TestLabels:
    def test_mid_band_is_excluded_from_both(self) -> None:
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        fractions = np.array([0.0, 0.3, 0.69, 0.7])
        human, ai = split_by_label(scores, fractions)
        assert human.tolist() == [1.0]
        assert ai.tolist() == [4.0]

    def test_lightly_edited_human_is_not_a_false_positive(self) -> None:
        """A 30%-AI document flagged is not a false positive in any useful sense."""
        _, ai = split_by_label(np.array([9.0]), np.array([0.3]))
        assert ai.size == 0


class TestThresholds:
    def test_threshold_at_fpr_is_conservative(self) -> None:
        human = np.arange(100.0)
        t = threshold_at_fpr(human, 0.05)
        assert fpr_at_threshold(human, t) <= 0.05

    def test_threshold_at_recall_holds_the_target(self) -> None:
        ai = np.arange(100.0)
        t = threshold_at_recall(ai, 0.90)
        assert recall_at_threshold(ai, t) >= 0.90

    def test_perfect_separation(self) -> None:
        human, ai = np.zeros(100), np.ones(100)
        fpr, _ = fpr_at_recall(human, ai, 1.0)
        assert fpr == 0.0

    def test_empty_inputs_do_not_crash(self) -> None:
        empty = np.array([])
        assert fpr_at_threshold(empty, 0.5) == 0.0
        assert recall_at_threshold(empty, 0.5) == 0.0
        assert threshold_at_fpr(empty, 0.01) == float("-inf")
        assert threshold_at_recall(empty, 0.9) == float("inf")


class TestAuroc:
    def test_matches_sklearn(self) -> None:
        sklearn_metrics = pytest.importorskip("sklearn.metrics")
        human, ai = separated()
        y = np.concatenate([np.zeros(human.size), np.ones(ai.size)])
        s = np.concatenate([human, ai])
        assert auroc(human, ai) == pytest.approx(sklearn_metrics.roc_auc_score(y, s), abs=1e-9)

    def test_perfect_and_random(self) -> None:
        assert auroc(np.zeros(50), np.ones(50)) == 1.0
        assert auroc(np.ones(50), np.ones(50)) == pytest.approx(0.5)

    def test_ties_count_as_half(self) -> None:
        # Pairs: (1,2) (1,3) (2,3) are wins, (2,2) is a tie -> (3 + 0.5) / 4.
        assert auroc(np.array([1.0, 2.0]), np.array([2.0, 3.0])) == pytest.approx(0.875)


class TestPauc:
    def test_matches_sklearn_mcclish(self) -> None:
        sklearn_metrics = pytest.importorskip("sklearn.metrics")
        human, ai = separated()
        y = np.concatenate([np.zeros(human.size), np.ones(ai.size)])
        s = np.concatenate([human, ai])
        expected = sklearn_metrics.roc_auc_score(y, s, max_fpr=0.02)
        assert pauc(human, ai, 0.02) == pytest.approx(expected, abs=0.01)

    def test_bounded(self) -> None:
        human, ai = separated()
        assert 0.0 <= pauc(human, ai) <= 1.0

    def test_discriminates_where_auroc_does_not(self) -> None:
        """Why pAUC is the selection metric rather than AUROC.

        Two detectors with nearly the same AUROC can behave completely differently at
        a 2% false-positive rate, which is the only place this product operates. The
        bimodal one puts most of its AI mass far above any plausible threshold; the
        broad one is better on average and worse where it counts.
        """
        n = 40000
        human = RNG.normal(0, 1, n)
        broad = RNG.normal(1.5, 1.0, n)
        bimodal = np.where(RNG.random(n) < 0.6, RNG.normal(3.0, 0.3, n), RNG.normal(0.3, 0.5, n))
        assert auroc(human, broad) == pytest.approx(auroc(human, bimodal), abs=0.05)
        assert pauc(human, bimodal, 0.02) > pauc(human, broad, 0.02) + 0.05


class TestClopperPearson:
    def test_upper_bound_exceeds_point_estimate(self) -> None:
        assert clopper_pearson_upper(20, 1000) > 20 / 1000

    def test_tightens_with_more_data(self) -> None:
        assert clopper_pearson_upper(20, 1000) > clopper_pearson_upper(200, 10000)

    def test_edges(self) -> None:
        assert clopper_pearson_upper(0, 0) == 1.0
        assert clopper_pearson_upper(10, 10) == 1.0
        assert clopper_pearson_upper(0, 100) < 0.05

    def test_thin_calibration_sets_are_untrustworthy(self) -> None:
        """Selecting at a 2% target on 3000 samples can really be over 2%."""
        assert clopper_pearson_upper(60, 3000) > 0.02


def test_expected_calibration_error() -> None:
    probs = np.linspace(0.01, 0.99, 1000)
    perfect = (RNG.random(1000) < probs).astype(float)
    assert expected_calibration_error(probs, perfect) < 0.05
    # A model that always says 0.9 while being right half the time is badly calibrated.
    assert expected_calibration_error(np.full(1000, 0.9), np.zeros(1000)) > 0.8
