"""Corpus config loading, and the content hash that drives resume.

The hash of the config is part of every stage's completion-marker path, so changing a
setting invalidates exactly the stages that depend on it and leaves the rest cached.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..data.genre import Genre
from ..data.windows import LengthBuckets

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "corpus.yaml"


@dataclass(frozen=True)
class CleaningConfig:
    min_words: int = 120
    max_words: int = 20000
    min_sentences: int = 3
    language: str = "en"
    min_language_score: float = 0.85
    l2_language_score_range: tuple[float, float] = (0.65, 0.90)


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.80
    val: float = 0.05
    calibration: float = 0.075
    test: float = 0.075
    min_human_windows_per_genre_in_calibration: int = 5000
    held_out_domains_per_genre: int = 10

    def __post_init__(self) -> None:
        total = self.train + self.val + self.calibration + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split proportions must sum to 1, got {total}")


@dataclass(frozen=True)
class CorpusConfig:
    corpus_seed: int
    split_salt: str
    window_buckets: LengthBuckets
    max_windows_per_document: int
    genre_targets: dict[Genre, int]
    min_natural_fraction: float
    max_date: str
    human_windows: int
    ai_windows: int
    budget_usd: float
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    dedup: dict[str, Any] = field(default_factory=dict)
    decontamination: dict[str, Any] = field(default_factory=dict)
    caps: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    @property
    def short_hash(self) -> str:
        """First 8 hex chars, used in resume-marker paths."""
        return self.content_hash[:8]


def load_config(path: Path = DEFAULT_CONFIG) -> CorpusConfig:
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    targets = raw["targets"]
    return CorpusConfig(
        corpus_seed=raw["corpus_seed"],
        split_salt=raw["split_salt"],
        window_buckets=tuple((lo, hi, w) for lo, hi, w in raw["window_buckets"]),
        max_windows_per_document=raw["max_windows_per_document"],
        genre_targets=raw["genre_targets"],
        min_natural_fraction=raw["min_natural_fraction"],
        max_date=str(raw["max_date"]),
        human_windows=targets["human_windows"],
        ai_windows=targets["ai_windows"],
        budget_usd=raw["budget_usd"],
        cleaning=CleaningConfig(
            **{
                **raw.get("cleaning", {}),
                "l2_language_score_range": tuple(
                    raw.get("cleaning", {}).get("l2_language_score_range", (0.65, 0.90))
                ),
            }
        ),
        splits=SplitConfig(**raw.get("splits", {})),
        dedup=raw.get("dedup", {}),
        decontamination=raw.get("decontamination", {}),
        caps=raw.get("caps", {}),
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
