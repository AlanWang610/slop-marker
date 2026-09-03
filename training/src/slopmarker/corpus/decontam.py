"""RAID decontamination (scope.md 4.6).

RAID is the external eval, and its train split is both labelled and public, so its
human side must never reach our training data. Direct reuse is not the risk -- our AI
side is self-generated. The risk is *source overlap*: RAID's human documents come from
Reddit (Webis TL;DR-17), arXiv abstracts, Wikipedia, BBC news, IMDb, recipes, book
summaries and poetry, and scope.md 4.2 has us drawing Reddit, Wikipedia and news.

Three layers, cheapest first. Source-level exclusion does most of the work and is the
only one that is airtight; the other two catch what leaks in through a different route.
The removal count is reported, because that number is what makes the external number
credible.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..data.normalize import collapse_whitespace, content_hash

# Datasets RAID draws its human side from. Never harvest these, whatever else changes.
RAID_HUMAN_SOURCES = frozenset(
    {
        "webis/tldr-17",
        "stanfordnlp/imdb",
        "recipe_nlg",
        "RecipeNLG",
        "cmu_book_summaries",
        "bbc_news_2006",
        "kaggle_arxiv",
        "kaggle_poetry",
    }
)

DEFAULT_NGRAM = 13
DEFAULT_OVERLAP = 0.02
DEFAULT_JACCARD = 0.7


def word_ngrams(text: str, n: int = DEFAULT_NGRAM) -> set[str]:
    words = collapse_whitespace(text).split(" ")
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


@dataclass
class RaidIndex:
    """Exact hashes and n-grams of RAID's human documents."""

    hashes: set[str]
    ngrams: set[str]
    ngram_size: int = DEFAULT_NGRAM

    @classmethod
    def build(cls, texts: Iterable[str], ngram_size: int = DEFAULT_NGRAM) -> RaidIndex:
        hashes: set[str] = set()
        ngrams: set[str] = set()
        for text in texts:
            hashes.add(content_hash(text))
            ngrams |= word_ngrams(text, ngram_size)
        return cls(hashes=hashes, ngrams=ngrams, ngram_size=ngram_size)

    def overlap(self, text: str) -> float:
        """Fraction of the document's n-grams that also appear in RAID."""
        grams = word_ngrams(text, self.ngram_size)
        if not grams:
            return 0.0
        return len(grams & self.ngrams) / len(grams)

    def is_contaminated(self, text: str, max_overlap: float = DEFAULT_OVERLAP) -> bool:
        if content_hash(text) in self.hashes:
            return True
        return self.overlap(text) >= max_overlap


def load_raid_human_texts(limit: int | None = None) -> list[str]:
    """Human rows from RAID's labelled splits, via the HF dataset.

    The raid-bench package is deliberately not a dependency: it pins numpy<1.27 and
    scikit-learn<1.4, which would drag the whole project onto a 2023 stack. RAID is
    just a dataset, so it is read as one.
    """
    from datasets import load_dataset

    texts: list[str] = []
    dataset = load_dataset("liamdugan/raid", split="train", streaming=True)
    seen: set[str] = set()
    for row in dataset:
        if row.get("model") != "human":
            continue
        key = str(row.get("source_id") or row.get("id") or len(texts))
        if key in seen:
            continue
        seen.add(key)
        texts.append(str(row["generation"]))
        if limit is not None and len(texts) >= limit:
            break
    return texts
