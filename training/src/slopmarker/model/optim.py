"""Layerwise learning-rate decay for ModernBERT (scope.md 4.4).

Lower layers move less than the head. Depth 0 is the embeddings, 1..22 the encoder
layers, 23 the heads, and each group gets `base_lr * decay ** (top - depth)`.

One trap worth knowing about: HuggingFace schedulers are LambdaLR, which *scales* each
group's stored base_lr by a shared factor, so per-group rates survive scheduling. Swap
in a scheduler that *sets* lr instead and layerwise decay silently disappears with no
error anywhere. The test asserts the group ratios both at step 0 and mid-warmup.

ModernBERT has no biases anywhere -- attention, MLP, classifier and norm biases are all
disabled -- so the no-decay group only ever catches norm weights.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import torch

NUM_LAYERS = 22
HEAD_DEPTH = NUM_LAYERS + 1

_LAYER = re.compile(r"\blayers\.(\d+)\.")


def parameter_depth(name: str) -> int:
    """0 = embeddings, 1..22 = encoder layers, 23 = heads and final norm."""
    if "embeddings." in name:
        return 0
    match = _LAYER.search(name)
    if match:
        return int(match.group(1)) + 1
    return HEAD_DEPTH


def _no_decay(name: str) -> bool:
    return name.endswith(".bias") or "norm" in name.lower()


def build_llrd_param_groups(
    model: torch.nn.Module,
    base_lr: float,
    *,
    decay: float = 0.9,
    weight_decay: float = 0.01,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, bool], list[torch.nn.Parameter]] = defaultdict(list)
    for name, param in model.named_parameters():
        if param.requires_grad:
            buckets[(parameter_depth(name), _no_decay(name))].append(param)
    return [
        {
            "params": params,
            "lr": base_lr * decay ** (HEAD_DEPTH - depth),
            "weight_decay": 0.0 if skip_decay else weight_decay,
            "depth": depth,
        }
        for (depth, skip_decay), params in sorted(buckets.items())
    ]
