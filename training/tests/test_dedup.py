"""Near-duplicate clustering and RAID decontamination."""

from __future__ import annotations

import random

import pytest

from slopmarker.corpus.decontam import RaidIndex, word_ngrams
from slopmarker.corpus.dedup import (
    cap_per_cluster,
    cluster_near_duplicates,
    cluster_sizes,
    exact_duplicates,
    shingles,
)

# A word bank, so each seed yields genuinely distinct 13-grams.
WORDS = [
    "harbour",
    "lantern",
    "gravel",
    "meridian",
    "tundra",
    "copper",
    "salvage",
    "thicket",
    "bureau",
    "kettle",
    "quarry",
    "mantle",
    "drifting",
    "fathom",
    "pigment",
    "trellis",
    "cobalt",
    "marrow",
    "ledger",
    "tempest",
    "furrow",
    "beacon",
    "quiver",
    "saffron",
    "obsidian",
    "bramble",
    "cistern",
    "hollow",
    "verdant",
    "plinth",
]


def article(seed: int = 0, n: int = 30) -> str:
    """Genuinely distinct text per seed.

    Deliberately not a template with the seed substituted in: that would share
    almost every 13-gram between seeds and make the overlap tests meaningless.
    """
    rng = random.Random(seed)
    return " ".join(
        " ".join(rng.choice(WORDS) for _ in range(12)).capitalize() + "." for _ in range(n)
    )


class TestShingles:
    def test_counts(self) -> None:
        assert len(shingles("a b c d e f", size=5)) == 2

    def test_ignores_whitespace_differences(self) -> None:
        assert shingles("a b  c   d e f") == shingles("a b c d e f")

    def test_short_text_yields_one_shingle(self) -> None:
        assert len(shingles("a b", size=5)) == 1

    def test_empty(self) -> None:
        assert shingles("") == set()


class TestExactDuplicates:
    def test_detects_typographic_variants(self) -> None:
        """Documents differing only in quotes and dashes are the same document."""
        curly = "The report \u2014 released Tuesday \u2014 said \u201cyes\u201d."
        plain = 'The report - released Tuesday - said "yes".'
        assert exact_duplicates([("a", curly), ("b", plain)]) == {"b": "a"}

    def test_first_occurrence_wins(self) -> None:
        dupes = exact_duplicates([("a", "same"), ("b", "same"), ("c", "same")])
        assert dupes == {"b": "a", "c": "a"}

    def test_distinct_documents_are_not_duplicates(self) -> None:
        assert exact_duplicates([("a", article(1)), ("b", article(2))]) == {}


class TestNearDuplicateClustering:
    def test_syndicated_copies_share_a_cluster(self) -> None:
        """The press-release case: one story reprinted across many domains."""
        base = article(1)
        variants = [
            ("prnewswire", base),
            ("stockhouse", base + " Additional distribution notice appended."),
            ("marketbeat", "Reprinted with permission. " + base),
            ("unrelated", article(99)),
        ]
        clusters = cluster_near_duplicates(variants)
        syndicated = {clusters[k] for k in ("prnewswire", "stockhouse", "marketbeat")}
        assert len(syndicated) == 1, "syndicated copies must form one cluster"
        assert clusters["unrelated"] not in syndicated

    def test_singletons_map_to_themselves(self) -> None:
        clusters = cluster_near_duplicates([("a", article(1)), ("b", article(2))])
        assert clusters == {"a": "a", "b": "b"}

    def test_cluster_ids_are_order_independent(self) -> None:
        base = article(3)
        items = [("z", base), ("m", base + " tail"), ("a", base + " other tail")]
        forward = cluster_near_duplicates(items)
        backward = cluster_near_duplicates(list(reversed(items)))
        assert set(forward.values()) == set(backward.values())

    def test_cluster_sizes(self) -> None:
        base = article(4)
        clusters = cluster_near_duplicates([("a", base), ("b", base), ("c", article(5))])
        assert sorted(cluster_sizes(clusters).values()) == [1, 2]


class TestCapPerCluster:
    def test_caps_and_is_deterministic(self) -> None:
        clusters = {"a": "a", "b": "a", "c": "a", "d": "d"}
        kept = list(cap_per_cluster(clusters, 2))
        assert kept == ["a", "b", "d"]
        assert kept == list(cap_per_cluster(clusters, 2))


class TestRaidIndex:
    def test_ngrams(self) -> None:
        assert len(word_ngrams("a b c", n=13)) == 0
        assert len(word_ngrams(" ".join(str(i) for i in range(15)), n=13)) == 3

    def test_exact_match_is_contaminated(self) -> None:
        index = RaidIndex.build([article(1)])
        assert index.is_contaminated(article(1))

    def test_typographic_variant_of_a_raid_doc_is_caught(self) -> None:
        index = RaidIndex.build(["The report \u2014 released Tuesday \u2014 said \u201cyes\u201d."])
        assert index.is_contaminated('The report - released Tuesday - said "yes".')

    def test_unrelated_text_is_clean(self) -> None:
        index = RaidIndex.build([article(1)])
        assert not index.is_contaminated(article(999))
        assert index.overlap(article(999)) == pytest.approx(0.0)

    def test_partial_overlap_is_measured(self) -> None:
        """A document quoting a long passage from RAID is contaminated."""
        index = RaidIndex.build([article(1)])
        mixed = article(2) + " " + " ".join(article(1).split(" ")[:120])
        assert 0.0 < index.overlap(mixed) < 1.0
        assert index.is_contaminated(mixed)

    def test_empty_index_flags_nothing(self) -> None:
        assert not RaidIndex.build([]).is_contaminated(article(1))

    def test_formulaic_text_overlaps_heavily_by_nature(self) -> None:
        """A caveat worth pinning rather than discovering later.

        Templated genres -- press releases, product copy -- share long spans with
        each other by construction, so 13-gram overlap is a blunt instrument there.
        It is safe only because RAID's domains contain none of those genres; if that
        ever changes, this threshold will start deleting legitimate hard negatives.
        """
        boilerplate = (
            "About Acme Corporation. Acme Corporation is a leading global provider "
            "of integrated solutions serving customers in more than forty countries. "
            "For more information visit our website or contact investor relations. "
        )
        index = RaidIndex.build([boilerplate + article(1)])
        assert index.overlap(boilerplate + article(2)) > 0.02
