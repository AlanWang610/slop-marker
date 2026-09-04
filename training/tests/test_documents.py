"""Document-level scoring and the gate it feeds.

The chunk-level bound in scope.md 4.5 is not the number a reader experiences. These
tests pin the difference: a document flags only when a run survives the length penalty,
the document prior and the 150-word minimum.
"""

from __future__ import annotations

import pytest

from slopmarker.eval.calibration import AggregateParams, Calibration
from slopmarker.eval.documents import ChunkScorer, document_fpr, score_document
from slopmarker.export.gates import MAX_DOCUMENT_FPR_UPPER, release_gate

CAL = Calibration(
    version="test",
    temperature=1.0,
    t_on=0.84,
    t_off=0.74,
    aggregate=AggregateParams(),
)


def prose(sentences: int) -> str:
    """Distinct sentences of ~10 words, so the chunker sees real boundaries."""
    return " ".join(
        f"The committee reviewed proposal number {i} during the spring session carefully."
        for i in range(sentences)
    )


def constant(value: float) -> ChunkScorer:
    """A scorer that returns one fixed logit, so the aggregation is what is tested."""

    def scorer(texts: list[str]) -> list[float]:
        return [value] * len(texts)

    return scorer


def test_short_document_is_not_scored() -> None:
    """Under the 40-word chunk minimum there is nothing to score."""
    assert score_document("Three words only.", constant(9.0), CAL) is None


def test_confident_long_document_flags() -> None:
    result = score_document(prose(40), constant(9.0), CAL)
    assert result is not None
    assert result["flagged"]
    assert result["words"] >= CAL.aggregate.run_min_words


def test_confident_but_short_document_does_not_flag() -> None:
    """A run under 150 words is not shown, however certain the model is.

    This is the gap between the chunk-level bound and what a reader sees, and it is
    also why RAID's abstracts are a worst case for this pipeline.
    """
    result = score_document(prose(6), constant(9.0), CAL)
    assert result is not None
    assert result["words"] < CAL.aggregate.run_min_words
    assert not result["flagged"]


def test_human_document_does_not_flag() -> None:
    result = score_document(prose(40), constant(-9.0), CAL)
    assert result is not None
    assert not result["flagged"]
    assert result["max_run_score"] == 0.0


def test_ranking_score_is_defined_without_a_run() -> None:
    """max_chunk_p must separate documents that never fire; max_run_score cannot.

    Both of these documents produce no run, so both take the same max_run_score and
    would tie. AUROC over such ties describes the ties, not the model -- which is what
    the first RAID run reported.
    """
    quiet = score_document(prose(40), constant(-9.0), CAL)
    louder = score_document(prose(40), constant(-1.0), CAL)
    assert quiet is not None and louder is not None
    assert quiet["max_run_score"] == louder["max_run_score"] == 0.0
    assert louder["max_chunk_p"] > quiet["max_chunk_p"]


def test_document_fpr_counts_and_bounds() -> None:
    report = document_fpr([prose(40)] * 50, constant(-9.0), CAL)
    assert report["n_human"] == 50
    assert report["doc_level_fpr"] == 0.0
    # Zero observed still carries a bound, and 50 documents cannot establish 3%.
    assert report["doc_level_fpr_upper"] > MAX_DOCUMENT_FPR_UPPER


def test_document_fpr_skips_unscorable_documents() -> None:
    report = document_fpr([prose(40)] * 10 + ["too short"] * 5, constant(-9.0), CAL)
    assert report["n_human"] == 10
    assert report["n_skipped_too_short"] == 5


def healthy_report() -> dict:
    return {
        "parity": {"passed": True},
        "quantization": {"passed": True, "problems": []},
        "graph": {"passed": True, "problems": []},
        "int8_vs_fp32": {
            "spearman": 0.9819,
            "auroc_drop": 0.0009,
            "pauc_drop": 0.0016,
            "decision_flip_rate": 0.00725,
            "fp32": {"auroc": 0.9905},
            "int8": {"auroc": 0.9896},
        },
        "test_verification": {
            "per_genre": {
                g: {"n_human": 900, "fpr": 0.0, "fpr_upper": 0.004}
                for g in ("news", "blog_personal", "technical_docs")
            }
        },
        "document_fpr": {"n_human": 1000, "doc_level_fpr": 0.002, "doc_level_fpr_upper": 0.0063},
    }


def test_gate_passes_with_document_evidence() -> None:
    assert release_gate(healthy_report())["passed"]


def test_gate_fails_without_document_evidence() -> None:
    """scope.md's chunk-level bound was never the user-visible one."""
    report = healthy_report()
    del report["document_fpr"]
    result = release_gate(report)
    assert not result["passed"]
    assert any("document-level FPR did not run" in f for f in result["failures"])


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ({"n_human": 1000, "doc_level_fpr": 0.05, "doc_level_fpr_upper": 0.065}, "exceeds"),
        ({"n_human": 100, "doc_level_fpr": 0.0, "doc_level_fpr_upper": 0.03}, "not informative"),
    ],
)
def test_gate_rejects_bad_document_evidence(document: dict, expected: str) -> None:
    report = healthy_report()
    report["document_fpr"] = document
    result = release_gate(report)
    assert not result["passed"]
    assert any(expected in f for f in result["failures"])
