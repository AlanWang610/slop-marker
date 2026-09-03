"""The genre taxonomy (scope.md 4.2), shared by the human and AI sides.

Genre is load-bearing in three places: the auxiliary training head (scope.md 4.4), the
per-genre FPR gate (scope.md 4.5), and the class-balance assertion that actually stops
the model learning "formal register => AI".

The label is always produced by one text-only classifier applied identically to both
classes. That is not an implementation convenience -- the AI side has no URL, so if
human genre came from URLs and AI genre came from generation prompts, the labelling
process itself would leak the class, and the auxiliary head would amplify the leak
rather than suppress it. What a generation prompt asked for is recorded separately as
`genre_intended`, and its disagreement with `genre` is a QA metric, never a label.
"""

from __future__ import annotations

from typing import Final, Literal

Genre = Literal[
    "news",
    "product_marketing",
    "press_release",
    "forum_comment",
    "blog_personal",
    "technical_docs",
    "encyclopedia",
    "academic_formal",
    "other",
]

# The eight scope.md 4.2 genres, in a fixed order. Index is the aux head's class id.
GENRES: Final[tuple[Genre, ...]] = (
    "news",
    "product_marketing",
    "press_release",
    "forum_comment",
    "blog_personal",
    "technical_docs",
    "encyclopedia",
    "academic_formal",
)

# Documents that fit none of the eight. Kept in the pool with a reason code so they can
# be inspected, but excluded from training windows.
OTHER: Final[Genre] = "other"

ALL_LABELS: Final[tuple[Genre, ...]] = (*GENRES, OTHER)
GENRE_ID: Final[dict[Genre, int]] = {g: i for i, g in enumerate(ALL_LABELS)}

# The genres where human text is most often mistaken for AI (scope.md 2). These are
# over-sampled on the human side and drive the per-genre FPR gate.
HardNegativeKind = Literal[
    "press_release",
    "seo_marketing",
    "corporate_blog",
    "product_template",
    "non_native_forum",
    "academic_formal",
]


def genre_id(genre: Genre) -> int:
    return GENRE_ID[genre]
