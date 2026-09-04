"""Turn harvested documents into training windows.

Order is forced by data dependencies, not preference:

  exact dedup -> near-duplicate clustering -> RAID decontamination -> per-cluster cap
  -> split assignment -> windowing -> manifest

Clustering must precede splitting, because the cluster id *is* the grouping key that
stops syndicated text straddling train and test. Decontamination must precede capping,
so a cluster is not filled with documents that are about to be removed.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..data.genre import GENRES, Genre, genre_id
from ..data.schema import DocumentRow, Split, WindowRow
from ..data.splits import assign_split, group_key
from ..data.windows import sample_windows
from .config import CorpusConfig
from .decontam import RaidIndex
from .dedup import cap_per_cluster, cluster_near_duplicates, cluster_sizes, exact_duplicates
from .heuristics import genre_from_text
from .state import read_shard, shard_path, write_shard

WINDOW_SHARD_ROWS = 50_000


def load_documents(root: Path, stage: str = "interim") -> list[DocumentRow]:
    directory = root / stage
    if not directory.exists():
        return []
    documents: list[DocumentRow] = []
    for path in sorted(directory.glob("*.jsonl.zst")):
        documents.extend(DocumentRow.from_json(line) for line in read_shard(path))
    return documents


def load_raid_index(root: Path) -> RaidIndex | None:
    path = root / "external" / "raid_human.json"
    if not path.exists():
        return None
    return RaidIndex.build(json.loads(path.read_text(encoding="utf-8")))


def build(root: Path, cfg: CorpusConfig, version: str, *, log: Any = print) -> dict[str, Any]:
    started = time.time()

    def step(name: str) -> None:
        """Stage timing. Without it a slow stage is indistinguishable from a hang."""
        log(f"[{time.time() - started:7.1f}s] {name}")

    # Both halves of the corpus. The cross-class dedup below is why they are loaded
    # together: rewrite and continuation rows are seeded from human documents, so
    # near-duplicate pairs straddling the class boundary are guaranteed otherwise.
    step("loading documents")
    documents = load_documents(root, "interim") + load_documents(root, "generated")
    stats: dict[str, Any] = {"version": version, "harvested": len(documents)}
    if not documents:
        return stats

    by_id = {doc.doc_id: doc for doc in documents}

    step("genre labelling")
    # 0. Genre for anything no URL rule matched, from the text alone. Text-only is
    #    required rather than convenient: the AI side has no URL, so a URL-derived
    #    human label and a prompt-derived AI label would make the labelling process
    #    itself carry the class.
    relabelled = 0
    for doc in by_id.values():
        if doc.genre == "other":
            doc.genre, doc.genre_confidence = genre_from_text(doc.text)  # type: ignore[assignment]
            doc.genre_source = "rule"
            relabelled += 1
    stats["genre_relabelled_from_text"] = relabelled

    step("exact dedup")
    # 1. Exact duplicates, over the hash-normalized form.
    duplicates = exact_duplicates((doc.doc_id, doc.text) for doc in documents)
    for doc_id in duplicates:
        by_id.pop(doc_id, None)
    stats["exact_duplicates_removed"] = len(duplicates)

    step("minhash clustering")
    # 2. Near-duplicate clusters. The cluster id becomes the split grouping key.
    clusters = cluster_near_duplicates(
        ((doc_id, doc.text) for doc_id, doc in by_id.items()),
        threshold=cfg.dedup.get("jaccard_threshold", 0.8),
        num_perm=cfg.dedup.get("num_permutations", 112),
        shingle_words=cfg.dedup.get("shingle_words", 5),
    )
    sizes = cluster_sizes(clusters)
    for doc_id, cluster_id in clusters.items():
        by_id[doc_id].minhash_cluster = cluster_id
    stats["clusters"] = len(set(clusters.values()))

    step("raid decontamination")
    # 3. RAID decontamination. Reported, because that number is what makes the
    #    external evaluation credible.
    raid = load_raid_index(root)
    removed_by_raid = 0
    if raid is not None:
        threshold = cfg.decontamination.get("raid_ngram_overlap", 0.02)
        for doc_id in list(by_id):
            overlap = raid.overlap(by_id[doc_id].text)
            by_id[doc_id].raid_overlap = overlap
            if overlap >= threshold:
                by_id.pop(doc_id)
                removed_by_raid += 1
    stats["raid_contaminated_removed"] = removed_by_raid
    stats["raid_index_present"] = raid is not None

    step("cluster caps")
    # 4. Cap per cluster, so one viral document cannot dominate a genre.
    max_per_cluster = cfg.caps.get("max_documents_per_cluster", 3)
    keep = set(cap_per_cluster({d: clusters[d] for d in by_id}, max_per_cluster))
    by_id = {doc_id: doc for doc_id, doc in by_id.items() if doc_id in keep}
    stats["after_cluster_cap"] = len(by_id)

    step("split assignment")
    # 5. Split assignment, on groups rather than documents.
    #
    #    A human seed must share its derivatives' group. AI rows carry
    #    seed_cluster_id, but the seed itself does not know it was used as one, so
    #    without this it would group by cluster-or-host and could land in train while
    #    its own AI rewrite lands in test -- the same content on both sides of the
    #    split, which is exactly what grouping exists to prevent.
    seed_clusters = {
        doc.seed_doc_id: doc.seed_cluster_id
        for doc in by_id.values()
        if doc.seed_doc_id and doc.seed_cluster_id
    }
    adopted = 0
    for doc_id, cluster in seed_clusters.items():
        seed = by_id.get(doc_id)
        if seed is not None and not seed.seed_cluster_id:
            seed.seed_cluster_id = cluster
            adopted += 1
    stats["seeds_adopted_into_generation_clusters"] = adopted

    for doc in by_id.values():
        doc.group_key = group_key(
            seed_cluster_id=doc.seed_cluster_id,
            minhash_cluster=doc.minhash_cluster,
            cluster_size=sizes.get(doc.minhash_cluster or "", 1),
            host=doc.host,
            doc_id=doc.doc_id,
        )
        doc.split = assign_split(doc.group_key, cfg.split_salt, cfg.splits)

    step("windowing")
    # 6. Windows. The per-window AI fraction comes from the document's spans.
    windows: list[WindowRow] = []
    for doc in by_id.values():
        spans = [tuple(s) for s in doc.ai_spans] if doc.ai_spans else None
        for index, window in enumerate(
            sample_windows(
                doc.text,
                doc.doc_id,
                corpus_seed=cfg.corpus_seed,
                ai_spans=spans,  # type: ignore[arg-type]
                buckets=cfg.window_buckets,
                max_windows=cfg.max_windows_per_document,
            )
        ):
            windows.append(
                WindowRow(
                    window_id=f"{doc.doc_id}:{index:02d}",
                    doc_id=doc.doc_id,
                    k=index,
                    text=window.text,
                    n_words=window.n_words,
                    n_tokens=0,  # filled in by pretokenization
                    truncated=False,
                    ai_fraction=window.ai_fraction,
                    genre=doc.genre,
                    genre_id=genre_id(doc.genre),
                    split=doc.split or "train",
                    doc_class=doc.doc_class,
                    source=doc.source,
                    host=doc.host,
                    hard_negative_kind=doc.hard_negative_kind,
                    generator=doc.generator,
                    prompt_style=doc.prompt_style,
                    attack=doc.attack,
                )
            )

    step(f"writing {len(windows)} windows")
    _write_windows(root, windows, version)
    stats.update(_summarize(by_id.values(), windows))
    (root / "processed" / version / "manifest.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    step("done")
    return stats


def _write_windows(root: Path, windows: list[WindowRow], version: str) -> None:
    by_split: dict[Split, list[str]] = defaultdict(list)
    for window in windows:
        by_split[window.split].append(window.to_json())
    for split, rows in by_split.items():
        for shard_index in range(0, len(rows), WINDOW_SHARD_ROWS):
            chunk = rows[shard_index : shard_index + WINDOW_SHARD_ROWS]
            name = f"{split}-{shard_index // WINDOW_SHARD_ROWS:05d}"
            write_shard(shard_path(root / "processed" / version, "windows", name), chunk)


def _summarize(documents: Any, windows: list[WindowRow]) -> dict[str, Any]:
    genre_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    hard_negatives: Counter[str] = Counter()
    genre_ai: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for window in windows:
        genre_counts[window.genre] += 1
        split_counts[window.split] += 1
        class_counts[window.doc_class] += 1
        if window.hard_negative_kind:
            hard_negatives[window.hard_negative_kind] += 1
        bucket = genre_ai[window.genre]
        bucket[0] += 1
        bucket[1] += int(window.ai_fraction >= 0.7)

    return {
        "documents": sum(1 for _ in documents),
        "windows": len(windows),
        "windows_by_genre": dict(genre_counts),
        "windows_by_split": dict(split_counts),
        "windows_by_class": dict(class_counts),
        "hard_negative_windows": dict(hard_negatives),
        "genre_ai_rate": {g: (n_ai / n if n else 0.0) for g, (n, n_ai) in genre_ai.items()},
        "genres_missing": [g for g in GENRES if g not in genre_counts],
    }


def calibration_windows_per_genre(windows: list[WindowRow]) -> dict[Genre, int]:
    counts: Counter[Genre] = Counter()
    for window in windows:
        if window.split == "calibration" and window.ai_fraction == 0.0:
            counts[window.genre] += 1
    return dict(counts)
