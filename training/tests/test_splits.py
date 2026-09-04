"""Split assignment invariants. Each of these, if violated, inflates a metric silently."""

from __future__ import annotations

import pytest

from slopmarker.corpus.config import SplitConfig
from slopmarker.data.splits import (
    SplitIntegrityError,
    assign_split,
    check_split_integrity,
    group_key,
    held_out_domains,
)

CFG = SplitConfig()
SALT = "slopmarker-v1"
GENRES = [
    "news",
    "product_marketing",
    "press_release",
    "forum_comment",
    "blog_personal",
    "technical_docs",
    "encyclopedia",
    "academic_formal",
]


class TestGroupKey:
    def test_syndicated_copies_group_together(self) -> None:
        """The leak this exists to close: same story, different domains."""
        a = group_key(minhash_cluster="c1", cluster_size=7, host="prnewswire.com", doc_id="a")
        b = group_key(minhash_cluster="c1", cluster_size=7, host="stockhouse.com", doc_id="b")
        assert a == b

    def test_singleton_falls_back_to_host(self) -> None:
        key = group_key(minhash_cluster="c9", cluster_size=1, host="example.com", doc_id="x")
        assert key == "host:example.com"

    def test_ai_rows_group_by_seed_cluster(self) -> None:
        """A seed, its generations, its attacks and its splices are one unit."""
        keys = {
            group_key(seed_cluster_id="s1", host="example.com", doc_id=f"gen-{i}") for i in range(5)
        }
        assert len(keys) == 1

    def test_seed_cluster_outranks_minhash_cluster(self) -> None:
        key = group_key(seed_cluster_id="s1", minhash_cluster="c1", cluster_size=9)
        assert key.startswith("seed:")

    def test_last_resort_is_the_document(self) -> None:
        assert group_key(doc_id="lonely") == "doc:lonely"


class TestAssignSplit:
    def test_deterministic(self) -> None:
        assert assign_split("host:a.com", SALT, CFG) == assign_split("host:a.com", SALT, CFG)

    def test_salt_changes_assignment(self) -> None:
        keys = [f"host:{i}.com" for i in range(500)]
        a = [assign_split(k, "salt-a", CFG) for k in keys]
        b = [assign_split(k, "salt-b", CFG) for k in keys]
        assert a != b

    def test_proportions_are_roughly_right(self) -> None:
        keys = [f"host:{i}.example" for i in range(40000)]
        counts: dict[str, int] = {}
        for k in keys:
            s = assign_split(k, SALT, CFG)
            counts[s] = counts.get(s, 0) + 1
        for name in ("train", "val", "calibration", "test"):
            observed = counts[name] / len(keys)
            assert observed == pytest.approx(getattr(CFG, name), abs=0.01), name

    def test_monotone_under_corpus_growth(self) -> None:
        """Adding documents must never move an existing group's split."""
        existing = {k: assign_split(k, SALT, CFG) for k in (f"host:{i}.com" for i in range(100))}
        for _ in range(1000):  # pretend the corpus grew
            assign_split("host:new.com", SALT, CFG)
        assert {k: assign_split(k, SALT, CFG) for k in existing} == existing


def test_held_out_domains_are_deterministic_and_sized() -> None:
    by_genre = {g: [f"{g}-{i}.com" for i in range(50)] for g in GENRES}
    reserved = held_out_domains(by_genre, SALT, per_genre=10)
    assert len(reserved) == 10 * len(GENRES)
    assert reserved == held_out_domains(by_genre, SALT, per_genre=10)


class TestIntegrityChecks:
    def _ok(self, **overrides: object) -> dict:
        base = dict(
            group_to_split={"host:a.com": "train"},
            doc_to_group={"d1": "host:a.com"},
            human_windows_per_genre_in_calibration={g: 6000 for g in GENRES},
            genre_ai_rate={g: 0.5 for g in GENRES},
            cfg=CFG,
        )
        base.update(overrides)
        return base

    def test_passes_when_healthy(self) -> None:
        check_split_integrity(**self._ok())

    def test_rejects_unassigned_group(self) -> None:
        with pytest.raises(SplitIntegrityError, match="unassigned group"):
            check_split_integrity(**self._ok(doc_to_group={"d1": "host:missing.com"}))

    def test_rejects_thin_calibration_genre(self) -> None:
        thin = {g: 6000 for g in GENRES} | {"press_release": 900}
        with pytest.raises(SplitIntegrityError, match="too thin"):
            check_split_integrity(**self._ok(human_windows_per_genre_in_calibration=thin))

    def test_rejects_genre_label_skew(self) -> None:
        """If marketing copy is 85% AI, the model learns marketing => AI."""
        skewed = {g: 0.5 for g in GENRES} | {"product_marketing": 0.85}
        with pytest.raises(SplitIntegrityError, match="label-leaking"):
            check_split_integrity(**self._ok(genre_ai_rate=skewed))


def test_seed_and_its_derivatives_share_a_split() -> None:
    """A human seed and every AI row generated from it must land together.

    Otherwise the model sees the same content, entities and topic on both sides of
    the split, and the test number measures memorization rather than detection.
    """
    seed_cluster = "seedcluster-1"
    keys = {
        "human_seed": group_key(seed_cluster_id=seed_cluster, host="example.com", doc_id="h1"),
        "ai_rewrite": group_key(seed_cluster_id=seed_cluster, doc_id="a1"),
        "ai_mixed": group_key(seed_cluster_id=seed_cluster, doc_id="a2"),
        "ai_continuation": group_key(seed_cluster_id=seed_cluster, doc_id="a3"),
    }
    assert len(set(keys.values())) == 1
    splits = {assign_split(k, SALT, CFG) for k in keys.values()}
    assert len(splits) == 1
