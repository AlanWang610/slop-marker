"""Human-side harvest (scope.md 4.2).

Every schema here was verified against the live datasets rather than taken from
documentation, and three of the plan's intended sources did not survive that check:

  allenai/peS2o            script-based; `datasets` 5.x refuses to load it.
                           Replaced by common-pile/peS2o, which is parquet and carries
                           the `created` date the pre-2022 filter needs.
  barilan/blog_authorship  script-based, same failure. Dropped: personal blogs are
                           mined from FineWeb by host instead, which is both simpler
                           and consistent with how every other genre is routed.
  bigcode/the-stack-dedup  gated. Dropped to avoid a credential dependency; the Pile's
                           StackExchange subset covers technical prose.

`date` semantics differ per source and are recorded per row. A crawl date is only an
upper bound on authorship, which is the guarantee the pre-2022 cutoff actually needs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..data.genre import Genre
from ..data.schema import DocumentRow
from .hosts import classify_url, registered_domain

# Pre-2022 CommonCrawl dumps. One dump already oversamples the hard negatives we need
# by ~10x; a second buys temporal diversity. More than that triples cost for nothing.
FINEWEB_DUMPS = ("CC-MAIN-2021-49", "CC-MAIN-2019-51")

# Pile subsets we take, and the genre each feeds. USPTO Backgrounds and NIH ExPorter
# are the best formal hard negatives available anywhere: maximally templated human
# prose written to a house style.
PILE_GENRES: dict[str, Genre] = {
    "Wikipedia (en)": "encyclopedia",
    "StackExchange": "technical_docs",
    "HackerNews": "forum_comment",
    "PubMed Abstracts": "academic_formal",
    "NIH ExPorter": "academic_formal",
    "USPTO Backgrounds": "academic_formal",
    "FreeLaw": "academic_formal",
    "PhilPapers": "academic_formal",
    "ArXiv": "academic_formal",
}

# Subreddits with substantial prose rather than one-line replies.
REDDIT_SUBREDDITS = (
    "AskHistorians",
    "DIY",
    "EatCheapAndHealthy",
    "Fitness",
    "Foodforthought",
    "Documentaries",
    "GetMotivated",
    "Fantasy",
    "Games",
    "IAmA",
)
REDDIT_CUTOFF_UTC = 1640995200  # 2022-01-01

# A routed URL already implies a genre. Unrouted documents stay "other" and are
# resolved later by the text-only classifier, which is what both classes share.
HARD_NEGATIVE_GENRE: dict[str, Genre] = {
    "press_release": "press_release",
    "seo_marketing": "product_marketing",
    "product_template": "product_marketing",
    "corporate_blog": "blog_personal",
    "non_native_forum": "forum_comment",
    "academic_formal": "academic_formal",
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo: str
    config: str | None = None
    split: str = "train"


def doc_id_for(source: str, key: str) -> str:
    return hashlib.blake2b(f"{source}:{key}".encode(), digest_size=16).hexdigest()


Shard = tuple[int, int] | None


def _shard(dataset: Any, shard: Shard) -> Any:
    """Split a streaming dataset across containers, by underlying file.

    Modulo-on-row-index would make every container read every row; file-level
    sharding means each container reads only its own share of the bytes.
    """
    if shard is None:
        return dataset
    index, num_shards = shard
    if num_shards <= 1:
        return dataset
    # A dataset with fewer files than shards raises "list index out of range" on the
    # high indices rather than yielding nothing. Those shards have no work to do, so
    # hand them an empty stream; the lower shards still cover every file.
    available = getattr(dataset, "num_shards", None)
    if available is not None and index >= available:
        return dataset.take(0)
    return dataset.shard(num_shards=num_shards, index=index)


def _before_cutoff(date: str | None, cutoff: str) -> bool:
    """Compare ISO-ish date prefixes as strings. Malformed dates are rejected."""
    if not date or len(date) < 4 or not date[:4].isdigit():
        return False
    year = int(date[:4])
    if not 1900 <= year <= 2100:  # peS2o carries a few year-0001 rows
        return False
    return date[:10] < cutoff


def harvest_fineweb(
    *,
    dumps: tuple[str, ...] = FINEWEB_DUMPS,
    cutoff: str = "2022-01-01",
    limit: int | None = None,
    corporate_hosts: frozenset[str] = frozenset(),
    sample: str | None = None,
    shard: Shard = None,
    keep_natural_rate: float = 0.25,
) -> Iterator[DocumentRow]:
    """FineWeb, routed by URL. The only source that can supply hard negatives at scale.

    `sample` selects a named subset (e.g. "sample-10BT") for pilot runs; the full
    harvest passes None and filters on `dump` instead.

    `keep_natural_rate` subsamples documents that matched no routing rule. Emitting
    every one of them would swamp the hard negatives -- routed URLs are ~6.5% of the
    crawl -- but keeping none would be worse: scope.md's mix needs at least 15% plain
    web text, or the model's notion of "human" drifts toward the adversarial end and
    ordinary prose starts scoring oddly.
    """
    import random

    from datasets import load_dataset

    rng = random.Random(0xC0FFEE + (shard[0] if shard else 0))

    kwargs: dict[str, Any] = {"streaming": True, "split": "train"}
    if sample:
        kwargs["name"] = sample
    dataset = _shard(load_dataset("HuggingFaceFW/fineweb", **kwargs), shard)

    emitted = 0
    for row in dataset:
        if not sample and row["dump"] not in dumps:
            continue
        if not _before_cutoff(row.get("date"), cutoff):
            continue
        url = row.get("url") or ""
        host = registered_domain(url)
        kind, rule = classify_url(url, corporate_host=host in corporate_hosts)
        if kind is None and rng.random() >= keep_natural_rate:
            continue
        yield DocumentRow(
            doc_id=doc_id_for("fineweb", row["id"]),
            text=row["text"],
            source="fineweb",
            source_config=row["dump"],
            doc_class="human",
            ai_fraction=0.0,
            genre=HARD_NEGATIVE_GENRE.get(kind or "", "other"),
            url=url,
            host=host,
            date=row.get("date"),
            date_kind="crawl",
            license="ODC-BY-1.0",
            genre_source="rule",
            sampling_stratum="targeted" if kind else "natural",
            hard_negative_kind=kind,
            mining_rule=rule or None,
        )
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def harvest_cc_news(
    *, cutoff: str = "2022-01-01", limit: int | None = None, shard: Shard = None
) -> Iterator[DocumentRow]:
    """CC-News 2017-2019. Entirely pre-cutoff, and carries exact publication dates.

    Press releases reach it through newswire syndication, so the same host router runs
    over `domain` -- with a real publication date attached rather than a crawl date.
    """
    from datasets import load_dataset

    dataset = _shard(load_dataset("vblagoje/cc_news", split="train", streaming=True), shard)
    emitted = 0
    for row in dataset:
        if not _before_cutoff(row.get("date"), cutoff):
            continue
        url = row.get("url") or ""
        kind, rule = classify_url(url)
        yield DocumentRow(
            doc_id=doc_id_for("cc_news", url or row["title"]),
            text=row["text"],
            source="cc_news",
            doc_class="human",
            ai_fraction=0.0,
            genre="press_release" if kind == "press_release" else "news",
            url=url,
            host=registered_domain(url),
            date=row.get("date"),
            date_kind="published",
            genre_source="rule",
            sampling_stratum="targeted" if kind else "natural",
            hard_negative_kind=kind,
            mining_rule=rule or None,
        )
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def harvest_pile(
    *,
    subsets: dict[str, Genre] | None = None,
    limit_per_subset: int | None = None,
    shard: Shard = None,
) -> Iterator[DocumentRow]:
    """The Pile, routed by subset. Collected in 2020, so entirely pre-cutoff."""
    from datasets import load_dataset

    subsets = subsets or PILE_GENRES
    dataset = _shard(
        load_dataset("monology/pile-uncopyrighted", split="train", streaming=True), shard
    )
    counts: dict[str, int] = {}
    for index, row in enumerate(dataset):
        name = row["meta"].get("pile_set_name", "")
        genre = subsets.get(name)
        if genre is None:
            continue
        if limit_per_subset is not None:
            if counts.get(name, 0) >= limit_per_subset:
                if all(counts.get(s, 0) >= limit_per_subset for s in subsets):
                    return
                continue
            counts[name] = counts.get(name, 0) + 1
        yield DocumentRow(
            doc_id=doc_id_for("pile", f"{name}:{index}"),
            text=row["text"],
            source="pile",
            source_config=name,
            doc_class="human",
            ai_fraction=0.0,
            genre=genre,
            date="2020-01-01",
            date_kind="dump",
            genre_source="rule",
            hard_negative_kind="academic_formal" if genre == "academic_formal" else None,
            sampling_stratum="targeted" if genre == "academic_formal" else "natural",
            mining_rule=f"pile_set:{name}",
        )


def harvest_pes2o(
    *, cutoff: str = "2022-01-01", limit: int | None = None, shard: Shard = None
) -> Iterator[DocumentRow]:
    """Academic full text, filtered to pre-cutoff by its own `created` field.

    Body text, not abstracts: RAID's abstracts domain comes from a Kaggle arXiv dump,
    and using bodies sidesteps that overlap entirely.
    """
    from datasets import load_dataset

    dataset = _shard(load_dataset("common-pile/peS2o", split="train", streaming=True), shard)
    emitted = 0
    for row in dataset:
        if not _before_cutoff(row.get("created"), cutoff):
            continue
        yield DocumentRow(
            doc_id=doc_id_for("pes2o", row["id"]),
            text=row["text"],
            source="pes2o",
            source_config=row.get("source"),
            doc_class="human",
            ai_fraction=0.0,
            genre="academic_formal",
            date=row.get("created"),
            date_kind="created",
            license="ODC-BY-1.0",
            genre_source="rule",
            sampling_stratum="targeted",
            hard_negative_kind="academic_formal",
            mining_rule="source:pes2o",
        )
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def harvest_reddit(
    *,
    subreddits: tuple[str, ...] = REDDIT_SUBREDDITS,
    min_score: int = 2,
    limit_per_subreddit: int | None = None,
    shard: Shard = None,
) -> Iterator[DocumentRow]:
    """Reddit comments. Note the split name *is* the subreddit -- there is no
    subreddit column -- and `created_utc` and `score` are strings, not numbers.
    """
    from datasets import load_dataset

    # Reddit shards by subreddit rather than by file: the split name is the subreddit,
    # so each container simply takes its own subset of them.
    if shard is not None and shard[1] > 1:
        index, num_shards = shard
        subreddits = subreddits[index::num_shards]
    for subreddit in subreddits:
        dataset = load_dataset("HuggingFaceGECLM/REDDIT_comments", split=subreddit, streaming=True)
        emitted = 0
        for row in dataset:
            try:
                created = int(row.get("created_utc") or 0)
                score = int(row.get("score") or 0)
            except (TypeError, ValueError):
                continue
            if created >= REDDIT_CUTOFF_UTC or created == 0 or score < min_score:
                continue
            body = row.get("body") or ""
            if body in ("[deleted]", "[removed]"):
                continue
            yield DocumentRow(
                doc_id=doc_id_for("reddit", row["id"]),
                text=body,
                source="reddit",
                source_config=subreddit,
                doc_class="human",
                ai_fraction=0.0,
                genre="forum_comment",
                date=datetime.fromtimestamp(created, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                date_kind="published",
                genre_source="rule",
                mining_rule=f"subreddit:{subreddit}",
            )
            emitted += 1
            if limit_per_subreddit is not None and emitted >= limit_per_subreddit:
                break


HARVESTERS = {
    "fineweb": harvest_fineweb,
    "cc_news": harvest_cc_news,
    "pile": harvest_pile,
    "pes2o": harvest_pes2o,
    "reddit": harvest_reddit,
}
