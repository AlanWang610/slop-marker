"""The training model: ModernBERT with an AI-fraction head and an auxiliary genre head.

The single most important structural decision here is that the *trained checkpoint is a
stock HuggingFace classifier*. A custom `model_type` would mean optimum-onnx has no
exporter for it and we would be writing an OnnxConfig by hand. Instead a stock
`ModernBertForSequenceClassification` is wrapped as `self.core` and the genre head hangs
off the pooled vector *outside* it, so stripping for export is a prefix filter on the
state dict and the export path is the fully supported text-classification task.

On the auxiliary head: scope.md 4.4 says its purpose is to discourage the "formal
register implies AI" shortcut. It does not do that. A multi-task cross-entropy head
*encourages* the backbone to encode genre linearly -- it is a reasonable generic
regulariser, but the thing that actually removes the shortcut is per-genre class
balance in the corpus, which splits.py asserts at load time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SlopMarkerOutput:
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None  # [B, 1] AI-fraction logit
    genre_logits: torch.Tensor | None = None  # [B, n_genres]
    ai_loss: torch.Tensor | None = None
    genre_loss: torch.Tensor | None = None


class GenreHead(nn.Module):
    """Deliberately not routed through the classifier's own prediction head.

    The auxiliary task must share only the pooled backbone representation. Letting its
    gradient flow through the AI head's parameters would regularise the wrong thing.
    """

    def __init__(self, hidden_size: int, num_genres: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, num_genres)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(self.drop(pooled))


def build_training_model(
    backbone: str = "answerdotai/ModernBERT-base",
    *,
    num_genres: int = 9,
    dropout: float = 0.1,
    pooling: str = "cls",
    attn_implementation: str | None = None,
):
    """Load the stock classifier and attach the auxiliary head beside it."""
    from transformers import AutoConfig, AutoModelForSequenceClassification

    config = AutoConfig.from_pretrained(backbone)
    config.num_labels = 1
    config.problem_type = "regression"
    config.classifier_pooling = pooling
    config.classifier_dropout = dropout
    # Must be off for export, and it is a recompilation storm under dynamic padding.
    config.reference_compile = False

    kwargs = {"config": config}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    core = AutoModelForSequenceClassification.from_pretrained(backbone, **kwargs)
    return SlopMarkerModel(core, num_genres=num_genres, dropout=dropout)


class SlopMarkerModel(nn.Module):
    def __init__(self, core: nn.Module, *, num_genres: int = 9, dropout: float = 0.1) -> None:
        super().__init__()
        self.core = core
        hidden = core.config.hidden_size
        self.genre_head = GenreHead(hidden, num_genres, dropout)

    @property
    def config(self):
        return self.core.config

    def _pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.core.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        if getattr(self.core.config, "classifier_pooling", "cls") == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return hidden[:, 0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ai_fraction: torch.Tensor | None = None,
        genre_id: torch.Tensor | None = None,
        *,
        eps: float = 0.05,
        smoothing_mode: str = "symmetric",
        genre_weight: float = 0.2,
        **_: object,
    ) -> SlopMarkerOutput:
        from .losses import multitask_loss

        pooled = self._pooled(input_ids, attention_mask)
        # Reuse the stock head so the stripped checkpoint is byte-identical to what
        # ModernBertForSequenceClassification.from_pretrained expects.
        logits = self.core.classifier(self.core.drop(self.core.head(pooled)))
        genre_logits = self.genre_head(pooled)

        loss = ai_loss = genre_loss = None
        if ai_fraction is not None:
            loss, ai_loss, genre_loss = multitask_loss(
                logits,
                ai_fraction,
                genre_logits,
                genre_id,
                eps=eps,
                mode=smoothing_mode,
                genre_weight=genre_weight,
            )
        return SlopMarkerOutput(
            loss=loss,
            logits=logits,
            genre_logits=genre_logits,
            ai_loss=ai_loss,
            genre_loss=genre_loss,
        )

    def strip_to_classifier(self) -> nn.Module:
        """Drop the auxiliary head, leaving a stock HF classifier ready for export."""
        return self.core
