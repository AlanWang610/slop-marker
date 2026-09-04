"""Temperature fitting and threshold selection."""

from __future__ import annotations

import numpy as np
import pytest

from slopmarker.eval.calibrate import calibrate, fit_temperature, verify

RNG = np.random.default_rng(11)


def rows(genres: dict[str, tuple[float, float]], n: int = 3000, scale: float = 1.0):
    """Synthetic scores per genre, as (human_mean, ai_mean).

    A "hard" genre is one whose *human* text scores high -- press releases and
    marketing copy read like AI because they are templated, not because the AI text
    there is unusual. Varying only the AI mean would leave every genre with the same
    human distribution and therefore the same threshold.
    """
    out = []
    for genre, (human_mean, ai_mean) in genres.items():
        for _ in range(n):
            is_ai = RNG.random() < 0.5
            centre = ai_mean if is_ai else human_mean
            out.append(
                {
                    "genre": genre,
                    "logit": float(RNG.normal(centre, 1.0) * scale),
                    "ai_fraction": 1.0 if is_ai else 0.0,
                }
            )
    return out


class TestTemperature:
    def test_recovers_a_known_scaling(self) -> None:
        truth = 2.5
        latent = RNG.normal(0, 2.0, 20000)
        targets = (RNG.random(20000) < 1 / (1 + np.exp(-latent))).astype(float)
        assert fit_temperature(latent * truth, targets) == pytest.approx(truth, rel=0.15)

    def test_returns_one_for_calibrated_logits(self) -> None:
        latent = RNG.normal(0, 2.0, 20000)
        targets = (RNG.random(20000) < 1 / (1 + np.exp(-latent))).astype(float)
        assert fit_temperature(latent, targets) == pytest.approx(1.0, rel=0.15)

    def test_handles_empty_and_nonfinite(self) -> None:
        assert fit_temperature(np.array([]), np.array([])) == 1.0
        assert np.isfinite(fit_temperature(np.array([np.inf, 1.0]), np.array([1.0, 0.0])))


class TestCalibrate:
    def test_worst_genre_sets_the_threshold(self) -> None:
        """The genre where human text most resembles AI drives the operating point."""
        result = calibrate(
            rows({"news": (0.0, 3.0), "press_release": (1.8, 3.2), "encyclopedia": (0.0, 3.0)})
        )
        assert result.binding_genre in {"press_release", "__overall__"}
        hardest = next(g for g in result.per_genre if g.genre == "press_release")
        assert hardest.threshold >= max(
            g.threshold for g in result.per_genre if g.genre != "press_release"
        )

    def test_meets_the_per_genre_bound(self) -> None:
        result = calibrate(
            rows({"news": (0.0, 3.0), "press_release": (1.5, 3.0), "blog_personal": (0.5, 2.8)})
        )
        for genre in result.per_genre:
            assert genre.fpr <= 0.02 + 1e-9, genre

    def test_oracle_gap_is_measured_and_non_negative(self) -> None:
        """The price of shipping one global threshold instead of per-genre ones."""
        result = calibrate(rows({"news": (0.0, 3.5), "press_release": (2.0, 3.2)}))
        assert result.oracle_recall >= result.recall - 1e-9
        assert result.oracle_gap >= 0.0

    def test_uniform_genres_leave_almost_no_gap(self) -> None:
        result = calibrate(rows({"a": (0.0, 2.5), "b": (0.0, 2.5), "c": (0.0, 2.5)}))
        assert result.oracle_gap < 0.1

    def test_selection_margin_undershoots_the_target(self) -> None:
        """Selecting at the target means missing it half the time on fresh data."""
        data = rows({"news": (0.0, 2.0)}, n=6000)
        tight = calibrate(data, margin=1.0)
        conservative = calibrate(data, margin=0.5)
        assert conservative.t_on >= tight.t_on

    def test_temperature_does_not_change_the_decision_rule(self) -> None:
        """A monotone transform cannot change FPR at a threshold chosen to hit an FPR."""
        data = rows({"news": (0.0, 2.0)}, n=6000)
        result = calibrate(data)
        scaled = calibrate([{**r, "logit": r["logit"] * 3.0} for r in data])
        assert result.overall_fpr == pytest.approx(scaled.overall_fpr, abs=0.004)
        assert result.recall == pytest.approx(scaled.recall, abs=0.05)


class TestVerify:
    def test_reports_bounds_per_genre(self) -> None:
        data = rows({"news": (0.0, 3.0), "press_release": (1.5, 3.0)})
        result = calibrate(data)
        checked = verify(data, result.temperature, result.t_on)
        assert set(checked["per_genre"]) == {"news", "press_release"}
        for genre in checked["per_genre"].values():
            assert genre["fpr_upper"] >= genre["fpr"]
        assert checked["worst_genre_fpr_upper"] >= max(
            g["fpr"] for g in checked["per_genre"].values()
        )

    def test_upper_bound_exceeds_the_point_estimate_on_thin_data(self) -> None:
        data = rows({"news": (0.0, 2.5)}, n=200)
        result = calibrate(data)
        checked = verify(data, result.temperature, result.t_on)
        news = checked["per_genre"]["news"]
        assert news["fpr_upper"] > news["fpr"]
