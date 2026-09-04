"""The release gate, written against the bundle that should never have been built."""

from __future__ import annotations

from typing import Any

import pytest

from slopmarker.export.bundle import assemble
from slopmarker.export.gates import release_gate


def healthy() -> dict[str, Any]:
    return {
        "parity": {"passed": True},
        "quantization": {"passed": True, "problems": []},
        "graph": {"passed": True, "problems": []},
        "int8_vs_fp32": {
            "spearman": 0.9991,
            "auroc_drop": 0.0021,
            "pauc_drop": 0.004,
            "fp32": {"auroc": 0.9928},
            "int8": {"auroc": 0.9907},
        },
        "test_verification": {
            "per_genre": {
                g: {"n_human": 900, "fpr": 0.0, "fpr_upper": 0.004}
                for g in ("news", "blog_personal", "technical_docs")
            }
        },
        # The user-visible bound. See test_documents.py for what it gates.
        "document_fpr": {"n_human": 1000, "doc_level_fpr": 0.002, "doc_level_fpr_upper": 0.0063},
    }


def test_healthy_report_passes() -> None:
    assert release_gate(healthy())["passed"]


def test_the_r1_bundle_would_have_been_refused() -> None:
    """The numbers the first export actually produced, which shipped a bundle anyway.

    fp32 parity, node counts, file size and opsets all passed; the model had lost
    twenty points of AUROC. This is the case the gate exists for.
    """
    report = healthy()
    report["int8_vs_fp32"] = {
        "spearman": 0.71,
        "auroc_drop": 0.2079,
        "pauc_drop": 0.35,
        "fp32": {"auroc": 0.9928},
        "int8": {"auroc": 0.7849},
    }
    result = release_gate(report)
    assert not result["passed"]
    assert any("Spearman" in f for f in result["failures"])
    assert any("AUROC drop" in f for f in result["failures"])


def test_a_missing_comparison_is_a_failure_not_a_pass() -> None:
    """Absent evidence must not read as passing evidence -- that was the original bug."""
    report = healthy()
    del report["int8_vs_fp32"]
    result = release_gate(report)
    assert not result["passed"]
    assert any("did not run" in f for f in result["failures"])


def test_a_thin_genre_fails_even_at_zero_observed_fpr() -> None:
    """57 human windows cannot bound a 2% tail, however clean the point estimate."""
    report = healthy()
    report["test_verification"]["per_genre"]["technical_docs"] = {
        "n_human": 57,
        "fpr": 0.0,
        "fpr_upper": 0.0805,
    }
    result = release_gate(report)
    assert not result["passed"]
    assert any("technical_docs" in f and "below" in f for f in result["failures"])


def test_assemble_refuses_a_failing_gate(tmp_path: Any) -> None:
    with pytest.raises(RuntimeError, match="release gate failed"):
        assemble(
            int8_model=tmp_path / "m.onnx",
            tokenizer_dir=tmp_path,
            calibration_path=tmp_path / "c.json",
            out_dir=tmp_path / "out",
            gate={"passed": False, "failures": ["int8 reorders against fp32"]},
        )
