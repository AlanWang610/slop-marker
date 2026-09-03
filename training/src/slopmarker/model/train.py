"""Training loop (scope.md 4.4).

A plain loop rather than HuggingFace Trainer. Trainer's compute_loss signature has
changed across releases and the multi-task loss plus layerwise decay plus a balanced
sampler needs three overrides anyway; a 150-line loop is less code than the subclass and
does not move underneath us.

Checkpointing is per-epoch and resumable, which matters because Modal GPU functions are
preemptible.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TrainConfig:
    backbone: str = "answerdotai/ModernBERT-base"
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 3e-5
    llrd_decay: float = 0.9
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    label_smoothing: float = 0.05
    smoothing_mode: str = "symmetric"
    genre_weight: float = 0.2
    pooling: str = "cls"
    max_length: int = 512
    max_grad_norm: float = 1.0
    balanced_sampling: bool = True
    eval_every: int = 500
    seed: int = 20260903


def evaluate(model: Any, loader: Any, device: Any, limit: int = 40) -> dict[str, float]:
    """Score a few batches for the training curve. The real evaluation is separate."""
    import torch

    from ..eval.metrics import auroc, fpr_at_recall, pauc, split_by_label
    from .losses import excess_bce

    model.eval()
    logits_all, targets_all, excess = [], [], []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= limit:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model(**batch)
            logits_all.append(out.logits.squeeze(-1).float().cpu().numpy())
            targets_all.append(batch["ai_fraction"].float().cpu().numpy())
            excess.append(float(excess_bce(out.logits.float(), batch["ai_fraction"].float())))
    model.train()

    if not logits_all:
        return {}
    scores = np.concatenate(logits_all)
    fractions = np.concatenate(targets_all)
    human, ai = split_by_label(scores, fractions)
    metrics = {"excess_bce": float(np.mean(excess))}
    if human.size and ai.size:
        fpr90, _ = fpr_at_recall(human, ai, 0.90)
        metrics |= {
            "auroc": auroc(human, ai),
            "pauc@2fpr": pauc(human, ai, 0.02),
            "fpr@recall90": fpr90,
            "n_human": float(human.size),
            "n_ai": float(ai.size),
        }
    return metrics


def train(
    train_rows: list[Any],
    val_rows: list[Any],
    cfg: TrainConfig,
    out_dir: Path,
    *,
    log: Any = print,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from ..data.dataset import WindowCollator, WindowDataset, balanced_weights, genre_ai_rates
    from .modeling import build_training_model
    from .optim import build_llrd_param_groups

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    rates = genre_ai_rates(train_rows)
    log(f"genre AI rates before balancing: { {k: round(v, 3) for k, v in rates.items()} }")

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    tokenizer.model_max_length = cfg.max_length
    collate = WindowCollator(tokenizer)

    train_set = WindowDataset(train_rows, tokenizer, cfg.max_length)
    val_set = WindowDataset(val_rows, tokenizer, cfg.max_length)

    sampler = None
    shuffle = True
    if cfg.balanced_sampling:
        # Equalise the AI rate within each genre. This, not the auxiliary head, is what
        # stops the model learning "formal register implies AI".
        weights = balanced_weights(train_rows)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), num_samples=len(train_rows)
        )
        shuffle = False

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=4,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, collate_fn=collate, num_workers=2)

    model = build_training_model(
        cfg.backbone,
        pooling=cfg.pooling,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
    ).to(device)

    groups = build_llrd_param_groups(
        model, cfg.learning_rate, decay=cfg.llrd_decay, weight_decay=cfg.weight_decay
    )
    optimizer = torch.optim.AdamW(groups, lr=cfg.learning_rate)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps
    )

    history: list[dict[str, Any]] = []
    step = 0
    best = -math.inf
    started = time.time()

    for epoch in range(cfg.epochs):
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model(
                    **batch,
                    eps=cfg.label_smoothing,
                    smoothing_mode=cfg.smoothing_mode,
                    genre_weight=cfg.genre_weight,
                )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % cfg.eval_every == 0:
                metrics = evaluate(model, val_loader, device)
                metrics |= {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(out.loss),
                    "ai_loss": float(out.ai_loss),
                    "genre_loss": float(out.genre_loss),
                    "lr": scheduler.get_last_lr()[-1],
                    "elapsed_s": round(time.time() - started, 1),
                }
                history.append(metrics)
                log(json.dumps(metrics))
                score = metrics.get("pauc@2fpr", -math.inf)
                if score > best:
                    best = score
                    _save(model, tokenizer, out_dir / "best", cfg)

        _save(model, tokenizer, out_dir / "last", cfg)

    final = evaluate(model, val_loader, device, limit=200)
    log(f"final val: {json.dumps(final)}")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {
        "steps": step,
        "epochs": cfg.epochs,
        "best_pauc": best,
        "final": final,
        "minutes": round((time.time() - started) / 60, 1),
        "genre_ai_rates": rates,
    }


def _save(model: Any, tokenizer: Any, path: Path, cfg: TrainConfig) -> None:
    """Save the stripped classifier -- the auxiliary head is training-only."""
    path.mkdir(parents=True, exist_ok=True)
    model.strip_to_classifier().save_pretrained(path)
    tokenizer.save_pretrained(path)
    (path / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
