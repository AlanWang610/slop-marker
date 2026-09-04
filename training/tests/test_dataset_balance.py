"""The sampler that removes the genre-implies-label shortcut."""

from __future__ import annotations

import numpy as np
import pytest

from slopmarker.data.dataset import balanced_weights, genre_ai_rates
from slopmarker.data.schema import WindowRow

# The shape the real corpus actually came out: academic prose dominates and every
# genre's AI rate sits well below half.
REAL_SKEW = {
    "academic_formal": (2156, 0.149),
    "blog_personal": (1211, 0.305),
    "product_marketing": (542, 0.106),
    "news": (364, 0.249),
    "forum_comment": (288, 0.167),
    "press_release": (107, 0.219),
    "encyclopedia": (84, 0.129),
    "technical_docs": (69, 0.080),
}


def corpus() -> list[WindowRow]:
    rows = []
    for genre, (n, ai_rate) in REAL_SKEW.items():
        for i in range(n):
            rows.append(
                WindowRow(
                    window_id=f"{genre}-{i}",
                    doc_id=f"{genre}-{i}",
                    k=0,
                    text="x",
                    n_words=100,
                    n_tokens=100,
                    truncated=False,
                    ai_fraction=1.0 if i < n * ai_rate else 0.0,
                    genre=genre,
                    genre_id=0,
                    split="train",
                )
            )
    return rows


def resample(rows: list[WindowRow], n: int = 120_000) -> list[WindowRow]:
    weights = balanced_weights(rows)
    picked = np.random.default_rng(0).choice(len(rows), size=n, p=weights)
    return [rows[i] for i in picked]


def test_ai_rate_is_balanced_within_every_genre() -> None:
    """Unbalanced, the model reads the label off the genre and no head prevents it."""
    rows = corpus()
    assert min(genre_ai_rates(rows).values()) < 0.11, "fixture should start skewed"
    for genre, rate in genre_ai_rates(resample(rows)).items():
        assert rate == pytest.approx(0.5, abs=0.03), f"{genre} at {rate:.3f}"


def test_genre_marginals_are_preserved() -> None:
    """Flattening genres too would upsample the rarest by ~30x and invite memorisation."""
    rows = corpus()
    before = genre_share(rows)
    after = genre_share(resample(rows))
    for genre, share in before.items():
        assert after[genre] == pytest.approx(share, abs=0.02), genre


def genre_share(rows: list[WindowRow]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.genre] = counts.get(row.genre, 0) + 1
    return {g: n / len(rows) for g, n in counts.items()}


def test_weights_are_a_probability_distribution() -> None:
    weights = balanced_weights(corpus())
    assert weights.sum() == pytest.approx(1.0)
    assert (weights > 0).all()


def test_single_class_genre_does_not_crash() -> None:
    rows = [
        WindowRow(
            window_id=f"w{i}",
            doc_id=f"d{i}",
            k=0,
            text="x",
            n_words=10,
            n_tokens=10,
            truncated=False,
            ai_fraction=0.0,
            genre="news",
            genre_id=0,
            split="train",
        )
        for i in range(5)
    ]
    assert balanced_weights(rows).sum() == pytest.approx(1.0)
