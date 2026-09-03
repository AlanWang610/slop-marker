"""Exact and near-duplicate detection.

Near-duplicate clusters are not just a filter. The cluster id becomes the split
grouping key (splits.py), which is what stops syndicated text straddling train and
test: a press release published on prnewswire and reprinted on forty other domains is
one cluster, so it lands wholly on one side of the split. Grouping by domain alone
would leak it.

datasketch rather than datatrove: at ~550k documents a single in-memory MinHashLSH is
a few hundred MB, and the band-partitioned multi-stage pipeline buys nothing at this
scale.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..data.normalize import collapse_whitespace, content_hash

DEFAULT_SHINGLE_WORDS = 5
DEFAULT_PERMUTATIONS = 112
DEFAULT_THRESHOLD = 0.8


def shingles(text: str, size: int = DEFAULT_SHINGLE_WORDS) -> set[bytes]:
    """Word n-grams, on collapsed text so whitespace differences cannot matter."""
    words = collapse_whitespace(text).split(" ")
    if len(words) < size:
        return {" ".join(words).encode("utf-8")} if words != [""] else set()
    return {" ".join(words[i : i + size]).encode("utf-8") for i in range(len(words) - size + 1)}


def signature(text: str, num_perm: int = DEFAULT_PERMUTATIONS, shingle_words: int = 5):  # type: ignore[no-untyped-def]
    from datasketch import MinHash

    minhash = MinHash(num_perm=num_perm)
    minhash.update_batch(sorted(shingles(text, shingle_words)))
    return minhash


def exact_duplicates(items: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Map doc_id -> the doc_id it exactly duplicates. First occurrence wins.

    Exactness is over the hash-normalized form, so documents differing only in
    typography or whitespace collapse together.
    """
    seen: dict[str, str] = {}
    duplicates: dict[str, str] = {}
    for doc_id, text in items:
        digest = content_hash(text)
        if digest in seen:
            duplicates[doc_id] = seen[digest]
        else:
            seen[digest] = doc_id
    return duplicates


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:  # path compression
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Keep the lexicographically smaller root so cluster ids are stable
            # regardless of the order documents arrive in.
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def cluster_near_duplicates(
    items: Iterable[tuple[str, str]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    num_perm: int = DEFAULT_PERMUTATIONS,
    shingle_words: int = DEFAULT_SHINGLE_WORDS,
) -> dict[str, str]:
    """Map every doc_id to a cluster id (the smallest doc_id in its cluster).

    Singletons map to themselves, so callers can treat the result as total.
    """
    from datasketch import MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    union = _UnionFind()
    signatures = {}

    for doc_id, text in items:
        minhash = signature(text, num_perm, shingle_words)
        signatures[doc_id] = minhash
        for match in lsh.query(minhash):
            union.union(doc_id, str(match))
        lsh.insert(doc_id, minhash)
        union.find(doc_id)

    return {doc_id: union.find(doc_id) for doc_id in signatures}


def cluster_sizes(clusters: dict[str, str]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for cluster_id in clusters.values():
        sizes[cluster_id] = sizes.get(cluster_id, 0) + 1
    return sizes


def cap_per_cluster(clusters: dict[str, str], max_per_cluster: int) -> Iterator[str]:
    """Yield doc_ids to keep, at most `max_per_cluster` from each cluster.

    Sorted so the choice is deterministic rather than dependent on dict order.
    """
    kept: dict[str, int] = {}
    for doc_id in sorted(clusters):
        cluster_id = clusters[doc_id]
        if kept.get(cluster_id, 0) < max_per_cluster:
            kept[cluster_id] = kept.get(cluster_id, 0) + 1
            yield doc_id
