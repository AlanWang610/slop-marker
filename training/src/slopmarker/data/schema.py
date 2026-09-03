"""Row schemas for the corpus. One shape for both classes; AI fields are None on human
rows (data/README.md).

Two files come out of the pipeline: documents.jsonl.zst (the canonical record) and
windows.jsonl.zst (the training unit, with the fields the trainer needs denormalized
onto it so the dataloader never has to join).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

from .genre import Genre, HardNegativeKind

SCHEMA_VERSION = "1.0"

Split = Literal["train", "val", "calibration", "test"]
DocClass = Literal["human", "ai"]
# What `date` actually means. A crawl date is only an upper bound on authorship, which
# is exactly the guarantee the pre-2022 cutoff needs, and nothing more.
DateKind = Literal["crawl", "published", "created", "dump", "unknown"]
PromptStyle = Literal[
    "continuation", "rewrite", "topic_prompt", "persona_instruction", "structured"
]
Attack = Literal[
    "paraphrase",
    "humanizer_prompt",
    "humanizer_chain",
    "shuffle_local",
    "shuffle_para",
    "typo_light",
    "typo_heavy",
]
# Which document supplied the unchanged ordering. Provenance only: at the token level
# "human draft then AI edit" and "AI draft then human edit" are the same object, so
# there is one splicer, not two.
MixDirection = Literal["human_base", "ai_base"]


@dataclass
class DocumentRow:
    doc_id: str
    text: str
    source: str  # fineweb | cc_news | pile | pes2o | reddit | wdc | the_stack | generated
    doc_class: DocClass

    # Labels. ai_fraction is the regression target and is always measured, never
    # requested: it is the fraction of the document's tokens emitted by a model.
    ai_fraction: float
    genre: Genre
    # Character spans of AI-authored text, on the collapsed text. Required whenever
    # ai_fraction is strictly between 0 and 1, because the window sampler recomputes a
    # per-window fraction from them -- a document-level fraction is not a valid label
    # for a window cut from the document's human half.
    ai_spans: list[tuple[int, int]] | None = None

    # Provenance
    url: str | None = None
    host: str | None = None  # registered domain, via the pinned public suffix list
    date: str | None = None  # ISO 8601
    date_kind: DateKind = "unknown"
    source_config: str | None = None  # CC dump, pile subset, subreddit, ...
    license: str | None = None

    # Genre labelling
    genre_source: Literal["rule", "model", "llm", "prompt"] = "model"
    genre_confidence: float | None = None
    genre_intended: Genre | None = None  # AI side: what the prompt asked for

    # Sampling and hard negatives
    sampling_stratum: Literal["targeted", "natural"] = "natural"
    hard_negative_kind: HardNegativeKind | None = None
    mining_rule: str | None = None  # audit trail for why this row was selected

    # Dedup, decontamination and splitting
    minhash_cluster: str | None = None
    group_key: str | None = None  # cluster first, registered domain second
    split: Split | None = None
    raid_overlap: float = 0.0
    # Diagnostic only. A test asserts no filter ever reads this: screening the human
    # class for "sounds like AI" would strip exactly the hard negatives scope.md 4.2
    # asks for and ship an optimistic threshold.
    slop_lexicon_score: float | None = None

    # Counts
    n_words: int = 0

    # AI side only
    seed_doc_id: str | None = None  # human document this was generated from
    seed_cluster_id: str | None = None  # THE SPLIT UNIT: seed plus all its derivatives
    generator: str | None = None
    generator_provider: Literal["anthropic", "openai", "google", "vllm"] | None = None
    prompt_style: PromptStyle | None = None
    prompt_version: str | None = None
    sampling_params: dict[str, Any] | None = None
    attack: Attack | None = None
    attack_model: str | None = None
    mix_direction: MixDirection | None = None
    ai_fraction_target: float | None = None  # requested; the label is the measured one
    hygiene_ops: list[str] = field(default_factory=list)

    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> DocumentRow:
        raw = json.loads(line)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class WindowRow:
    """One training example: a window cut from a document (scope.md 4.3)."""

    window_id: str  # f"{doc_id}:{k:02d}"
    doc_id: str
    k: int
    text: str
    n_words: int
    n_tokens: int
    truncated: bool

    # Recomputed from the document's ai_spans, not copied from the document.
    ai_fraction: float
    genre: Genre
    genre_id: int
    split: Split

    # Denormalized for per-slice reporting without a join.
    doc_class: DocClass = "human"
    source: str = ""
    host: str | None = None
    hard_negative_kind: HardNegativeKind | None = None
    generator: str | None = None
    prompt_style: PromptStyle | None = None
    attack: Attack | None = None

    schema_version: str = SCHEMA_VERSION

    @property
    def label_binary(self) -> int:
        """Binary label for evaluation (scope.md 4.2). Not a training target."""
        return int(self.ai_fraction >= 0.7)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> WindowRow:
        raw = json.loads(line)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})
