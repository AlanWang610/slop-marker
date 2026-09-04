"""The genre-share cap must hold on the corpus it produces, not the one it was given."""

from __future__ import annotations

from collections import Counter

import pytest

from slopmarker.corpus.build import _cap_genre_share
from slopmarker.data.schema import WindowRow

# Genre sizes as corpus v3 actually came out of windowing, before the cap ran.
V3_PRE_CAP = {
    "academic_formal": 439_063,
    "blog_personal": 236_157,
    "product_marketing": 92_464,
    "news": 68_987,
    "forum_comment": 58_067,
    "press_release": 20_466,
    "encyclopedia": 18_420,
    "technical_docs": 14_388,
}


def windows(sizes: dict[str, int], ai_rate: float = 0.2) -> list[WindowRow]:
    rows = []
    for genre, n in sizes.items():
        for i in range(n):
            rows.append(
                WindowRow(
                    window_id=f"{genre}-{i:07d}",
                    doc_id=f"{genre}-{i:07d}",
                    k=0,
                    text="x",
                    n_words=100,
                    n_tokens=100,
                    truncated=False,
                    ai_fraction=1.0 if i < n * ai_rate else 0.0,
                    genre=genre,  # type: ignore[arg-type]
                    genre_id=0,
                    split="train",
                )
            )
    return rows


@pytest.mark.parametrize("max_share", [0.25, 0.3, 0.5])
def test_cap_holds_on_the_output(max_share: float) -> None:
    """The share is measured against the surviving total, which the cap itself shrinks.

    Reading the allowance off the input size is what let v3 ship two genres at 0.317
    against a configured 0.25: the cap removed 200k windows and never re-measured.
    """
    kept = _cap_genre_share(windows({k: v // 100 for k, v in V3_PRE_CAP.items()}), max_share, 7)
    counts = Counter(w.genre for w in kept)
    assert counts
    for genre, count in counts.items():
        assert count / len(kept) <= max_share + 1e-9, f"{genre} at {count / len(kept):.4f}"


def test_uncapped_genres_are_untouched() -> None:
    sizes = {k: v // 100 for k, v in V3_PRE_CAP.items()}
    kept = _cap_genre_share(windows(sizes), 0.25, 7)
    counts = Counter(w.genre for w in kept)
    # Everything below the allowance keeps every window it had.
    for genre in ("news", "forum_comment", "press_release", "encyclopedia", "technical_docs"):
        assert counts[genre] == sizes[genre]


def test_label_rate_survives_the_cap() -> None:
    """Sampling is stratified by label, so capping must not move a genre's AI rate."""
    kept = _cap_genre_share(windows({k: v // 100 for k, v in V3_PRE_CAP.items()}, 0.2), 0.25, 7)
    ai = Counter(w.genre for w in kept if w.ai_fraction >= 0.7)
    total = Counter(w.genre for w in kept)
    for genre, n in total.items():
        assert abs(ai[genre] / n - 0.2) < 0.02, genre


def test_no_cap_is_a_passthrough() -> None:
    rows = windows({"news": 10, "blog_personal": 90})
    assert _cap_genre_share(rows, 1.0, 7) is rows
