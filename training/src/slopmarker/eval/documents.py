"""Document-level scoring: the whole scope.md 8 path, end to end.

Everything else in `eval/` works on chunk scores. This module is what a reader
actually experiences -- chunk the text the way the extension does, score each chunk on
the shipped artifact, then aggregate to runs. A document is flagged when a run survives
the length penalty, the document prior and the 150-word minimum, which is a stricter
condition than any single chunk crossing a threshold.

Both the export gate and the standalone evaluation call this, so the number that gates
a bundle is produced by the same code as the number that gets reported.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..data.chunking import chunk_block
from ..data.normalize import collapse_whitespace
from .aggregate import Chunk, aggregate
from .calibration import Calibration
from .metrics import clopper_pearson_upper


class ChunkScorer(Protocol):
    """Maps chunk texts to logits. An ONNX session or a torch model, wrapped."""

    def __call__(self, texts: list[str]) -> list[float]: ...


def score_document(
    text: str, scorer: ChunkScorer, cal: Calibration, batch: int = 32
) -> dict[str, Any] | None:
    """Chunk, score and aggregate one document. None when it is too short to chunk."""
    chunks = chunk_block(collapse_whitespace(text))
    if not chunks:
        return None

    scored: list[Chunk] = []
    for start in range(0, len(chunks), batch):
        piece = chunks[start : start + batch]
        for value, chunk in zip(scorer(piece), piece, strict=True):
            scored.append(Chunk(logit=float(value), words=len(chunk.split())))

    result = aggregate(scored, cal)
    flagged = result.flagged_runs
    return {
        "n_chunks": len(scored),
        "words": sum(c.words for c in scored),
        "flagged": bool(flagged),
        "flagged_words": sum(r.words for r in flagged),
        "max_run_score": max((r.score for r in result.runs), default=0.0),
        # The ranking score, and it must not be max_run_score. Runs exist only above
        # t_on, so every document without one takes the same default and the classes
        # pile up tied at a single value -- which makes AUROC a statement about the ties
        # rather than about the model. The strongest penalized chunk is always defined.
        "max_chunk_p": max(result.chunk_p_penalized, default=0.0),
    }


def document_fpr(texts: list[str], scorer: ChunkScorer, cal: Calibration) -> dict[str, Any]:
    """False-positive rate over human documents, with its Clopper-Pearson upper bound."""
    scored = [score_document(text, scorer, cal) for text in texts]
    rows = [row for row in scored if row is not None]
    if not rows:
        return {"n_human": 0, "doc_level_fpr": float("nan"), "doc_level_fpr_upper": 1.0}

    flagged = sum(1 for row in rows if row["flagged"])
    return {
        "n_human": len(rows),
        "n_skipped_too_short": len(scored) - len(rows),
        "flagged": flagged,
        "doc_level_fpr": flagged / len(rows),
        "doc_level_fpr_upper": clopper_pearson_upper(flagged, len(rows)),
        "median_words": sorted(row["words"] for row in rows)[len(rows) // 2],
        "median_chunks": sorted(row["n_chunks"] for row in rows)[len(rows) // 2],
    }
