"""Window dataset and collator for training.

Windows arrive already cut and labelled (corpus/build.py), so this only tokenizes and
batches. Two details matter:

Padding is dynamic, to the longest sequence in the batch. Most windows are far shorter
than 512 tokens -- a 165-word window is about 215 -- so padding everything to 512 would
more than double the compute for nothing. Right-padding with a correct attention mask
leaves the [CLS] output unchanged, so this costs no accuracy.

The model is fed `collapse_whitespace(text)`, the same normalization the extension
applies before inference. Training on raw text and inferring on normalized text would
be a silent, small, permanent accuracy loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..corpus.state import read_shard
from .genre import GENRE_ID
from .normalize import collapse_whitespace
from .schema import WindowRow

MAX_LENGTH = 512


def load_windows(root: Path, version: str, split: str) -> list[WindowRow]:
    directory = root / "processed" / version / "windows"
    if not directory.exists():
        return []
    rows: list[WindowRow] = []
    for path in sorted(directory.glob(f"{split}-*.jsonl.zst")):
        rows.extend(WindowRow.from_json(line) for line in read_shard(path))
    return rows


@dataclass
class Batch:
    input_ids: Any
    attention_mask: Any
    ai_fraction: Any
    genre_id: Any


class WindowDataset:
    """Tokenizes lazily so a shard of a million windows need not be held twice."""

    def __init__(self, rows: list[WindowRow], tokenizer: Any, max_length: int = MAX_LENGTH):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        encoded = self.tokenizer(
            collapse_whitespace(row.text),
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": encoded["input_ids"],
            "ai_fraction": float(row.ai_fraction),
            "genre_id": GENRE_ID.get(row.genre, GENRE_ID["other"]),
        }


@dataclass
class WindowCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        padded = self.tokenizer.pad(
            [{"input_ids": f["input_ids"]} for f in features], return_tensors="pt"
        )
        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "ai_fraction": torch.tensor([f["ai_fraction"] for f in features], dtype=torch.float32),
            "genre_id": torch.tensor([f["genre_id"] for f in features], dtype=torch.long),
        }


def genre_ai_rates(rows: list[WindowRow]) -> dict[str, float]:
    """Per-genre share of windows labelled AI.

    This is the number that decides whether the model can learn "genre implies label".
    It matters more than the auxiliary head, which does not remove that shortcut.
    """
    totals: dict[str, list[int]] = {}
    for row in rows:
        bucket = totals.setdefault(row.genre, [0, 0])
        bucket[0] += 1
        bucket[1] += int(row.ai_fraction >= 0.7)
    return {genre: ai / n for genre, (n, ai) in totals.items() if n}


def balanced_weights(rows: list[WindowRow]) -> np.ndarray:
    """Sampling weights that equalise the AI rate *within* each genre.

    This is the mechanism that removes the "formal register implies AI" shortcut: if
    marketing copy is 10% AI and blogs are 30%, the model can read the label off the
    genre, and no auxiliary head will stop it.

    Genre marginals are deliberately preserved rather than flattened. Equalising them
    too would upsample the rarest genre by a factor of thirty against the most common
    one -- on this corpus, technical documentation against academic prose -- which
    trades a label shortcut for memorisation of a few thousand windows.
    """
    genre_totals: dict[str, int] = {}
    cell_totals: dict[tuple[str, int], int] = {}
    for row in rows:
        label = int(row.ai_fraction >= 0.7)
        genre_totals[row.genre] = genre_totals.get(row.genre, 0) + 1
        key = (row.genre, label)
        cell_totals[key] = cell_totals.get(key, 0) + 1

    weights = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        label = int(row.ai_fraction >= 0.7)
        # genre share preserved, label balanced inside it
        weights[index] = genre_totals[row.genre] / cell_totals[(row.genre, label)]
    return weights / weights.sum()
