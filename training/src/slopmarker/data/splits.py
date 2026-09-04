"""Split assignment (scope.md 4.5).

Splits are by *group*, never by document and never by window. The grouping key is the
near-duplicate cluster first and the registered domain second. Cluster-first is the part
that matters: a press release is reprinted verbatim across dozens of domains, so
domain-level grouping alone puts the same text on both sides of the split and the test
number comes out flattering and wrong.

For AI rows the group is the seed cluster -- a human seed, every generation derived
from it, every attack on those, and every splice. Otherwise the model can memorize
topic -> label in train and be scored on the same topics in val.

Assignment is a hash of the group key, so it is deterministic, resumable, and monotone:
new documents in an existing group inherit its split, and a new group never moves
existing ones.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from ..corpus.config import SplitConfig
from .genre import Genre
from .schema import Split

SPLIT_ORDER: tuple[Split, ...] = ("train", "val", "calibration", "test")
_BUCKETS = 10_000


def group_key(
    *,
    seed_cluster_id: str | None = None,
    minhash_cluster: str | None = None,
    cluster_size: int = 1,
    host: str | None = None,
    doc_id: str = "",
) -> str:
    """The unit a split is assigned to. Order of precedence is deliberate."""
    if seed_cluster_id:
        return f"seed:{seed_cluster_id}"
    if minhash_cluster and cluster_size > 1:
        return f"cluster:{minhash_cluster}"
    if host:
        return f"host:{host}"
    return f"doc:{doc_id}"


def _bucket(key: str, salt: str) -> int:
    digest = hashlib.blake2b(f"{salt}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _BUCKETS


def assign_split(key: str, salt: str, cfg: SplitConfig) -> Split:
    bucket = _bucket(key, salt)
    edge = 0.0
    for name in SPLIT_ORDER:
        edge += getattr(cfg, name)
        if bucket < edge * _BUCKETS:
            return name
    return "test"


def held_out_domains(
    domains_by_genre: dict[Genre, Iterable[str]], salt: str, per_genre: int
) -> set[str]:
    """Reserve some domains for test only.

    This measures whether the model memorized host templates rather than learning to
    detect AI text -- a failure mode per-genre FPR alone will not reveal, because a
    memorized host appears in both train and test.
    """
    reserved: set[str] = set()
    for genre, domains in domains_by_genre.items():
        ranked = sorted(domains, key=lambda d: _bucket(f"heldout:{genre}:{d}", salt))
        reserved.update(ranked[:per_genre])
    return reserved


class SplitIntegrityError(AssertionError):
    """Raised when a split invariant is violated. Never downgraded to a warning."""


# Reweighting can equalize a genre's AI rate but cannot invent distinct windows. Below
# this many, the sampler is showing the model the same handful of documents over and
# over at whatever rate it likes, and the genre contributes variety it does not have.
MIN_MINORITY_WINDOWS_PER_GENRE = 1000


def check_split_integrity(
    *,
    group_to_split: dict[str, Split],
    doc_to_group: dict[str, str],
    human_windows_per_genre_in_calibration: dict[Genre, int],
    genre_ai_counts: dict[Genre, tuple[int, int]],
    cfg: SplitConfig,
) -> None:
    """Assert everything that would silently inflate a metric if it were false."""
    for doc_id, group in doc_to_group.items():
        if group not in group_to_split:
            raise SplitIntegrityError(f"document {doc_id} has unassigned group {group}")

    thin = {
        genre: n
        for genre, n in human_windows_per_genre_in_calibration.items()
        if n < cfg.min_human_windows_per_genre_in_calibration
    }
    if thin:
        raise SplitIntegrityError(
            "calibration set too thin to place a 2% per-genre threshold: "
            f"{thin} (need {cfg.min_human_windows_per_genre_in_calibration} each)"
        )

    # A genre that is mostly one class would teach the model genre => label. What
    # prevents that here is the balanced sampler, which equalizes P(AI | genre) in the
    # stream the model actually sees, so the corpus rate itself need not be 0.5 -- an
    # earlier version of this check asserted [0.4, 0.6] on the corpus and was checking
    # an object the design does not constrain. What the sampler cannot fix is a genre
    # whose minority class has too few *distinct* windows to reweight, so that is the
    # assertion.
    starved = {
        genre: counts
        for genre, counts in genre_ai_counts.items()
        if min(counts[1], counts[0] - counts[1]) < MIN_MINORITY_WINDOWS_PER_GENRE
    }
    if starved:
        raise SplitIntegrityError(
            "genres whose minority class has too few distinct windows for the balanced "
            f"sampler to work with: {starved} "
            f"(need {MIN_MINORITY_WINDOWS_PER_GENRE} each, as (total, ai))"
        )
