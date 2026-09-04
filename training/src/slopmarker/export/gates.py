"""Release gates: the checks a bundle must pass before it may be assembled.

These are separate from the structural checks in `onnx_export`. Those confirm the
export *ran* -- node counts, file size, opsets, graph inputs -- and every one of them
passed on a bundle whose model had lost 20 points of AUROC. A correctly-executed
quantization that destroys the model looks identical, from the graph, to one that does
not. Only scoring both artifacts on the same rows can tell them apart, so that
comparison is a gate rather than a report, and `assemble` refuses without it.
"""

from __future__ import annotations

from typing import Any

# Quantization must preserve the ranking, because the ranking is what the threshold is
# selected against. Spearman rather than a logit tolerance: int8 may shift the scale
# freely as long as it does not reorder.
MIN_SPEARMAN = 0.995
MAX_PAUC_DROP = 0.01
MAX_AUROC_DROP = 0.01

# scope.md 4.5. The bound is on the upper end of the interval, not the point estimate:
# a 2% tail measured from a few dozen windows is one observation wide.
MAX_GENRE_FPR_UPPER = 0.02
MIN_HUMAN_WINDOWS_PER_GENRE = 500


def release_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Decide whether the artifact described by `report` may ship."""
    failures: list[str] = []

    for name in ("parity", "quantization", "graph"):
        section = report.get(name)
        if section is None:
            failures.append(f"{name} check did not run")
        elif not section.get("passed"):
            failures.append(f"{name} check failed: {section.get('problems') or section}")

    comparison = report.get("int8_vs_fp32")
    if comparison is None:
        failures.append("int8-vs-fp32 comparison did not run")
    else:
        if comparison["spearman"] < MIN_SPEARMAN:
            failures.append(
                f"int8 reorders against fp32: Spearman {comparison['spearman']:.4f}"
                f" below {MIN_SPEARMAN}"
            )
        if comparison["auroc_drop"] > MAX_AUROC_DROP:
            failures.append(
                f"int8 AUROC drop {comparison['auroc_drop']:.4f} exceeds {MAX_AUROC_DROP}"
                f" ({comparison['fp32']['auroc']:.4f} -> {comparison['int8']['auroc']:.4f})"
            )
        if comparison["pauc_drop"] > MAX_PAUC_DROP:
            failures.append(
                f"int8 pAUC@2%FPR drop {comparison['pauc_drop']:.4f} exceeds {MAX_PAUC_DROP}"
            )

    verification = report.get("test_verification")
    if verification is None:
        failures.append("test-split verification did not run")
    else:
        for genre, row in sorted(verification.get("per_genre", {}).items()):
            if row["n_human"] < MIN_HUMAN_WINDOWS_PER_GENRE:
                failures.append(
                    f"{genre} has {row['n_human']} human test windows,"
                    f" below {MIN_HUMAN_WINDOWS_PER_GENRE}; its FPR bound is not informative"
                )
            elif row["fpr_upper"] > MAX_GENRE_FPR_UPPER:
                failures.append(
                    f"{genre} FPR upper bound {row['fpr_upper']:.4f} exceeds {MAX_GENRE_FPR_UPPER}"
                )

    return {"passed": not failures, "failures": failures}


def compare_scores(fp32: list[dict[str, Any]], int8: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare two artifacts scored on the same rows, in the same order.

    Spearman is the headline rather than a logit tolerance because the threshold is
    selected on the artifact that ships: int8 may move the whole scale without costing
    anything, but it may not reorder, since the ordering is the decision rule.
    """
    import numpy as np
    from scipy.stats import spearmanr

    from ..eval.metrics import auroc, pauc, split_by_label

    if len(fp32) != len(int8):
        raise ValueError(f"score lists differ in length: {len(fp32)} vs {len(int8)}")
    fractions = np.array([row["ai_fraction"] for row in fp32], dtype=np.float64)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        values = np.array([row["logit"] for row in rows], dtype=np.float64)
        human, ai = split_by_label(values, fractions)
        return {
            "n_human": int(human.size),
            "n_ai": int(ai.size),
            "auroc": auroc(human, ai),
            "pauc_2pct": pauc(human, ai, 0.02),
            "mean_human_logit": float(human.mean()) if human.size else float("nan"),
            "mean_ai_logit": float(ai.mean()) if ai.size else float("nan"),
        }

    left, right = summarize(fp32), summarize(int8)
    a = np.array([row["logit"] for row in fp32], dtype=np.float64)
    b = np.array([row["logit"] for row in int8], dtype=np.float64)
    return {
        "n": len(fp32),
        "fp32": left,
        "int8": right,
        "spearman": float(spearmanr(a, b).statistic),
        "max_abs_logit_diff": float(np.max(np.abs(a - b))) if a.size else 0.0,
        "auroc_drop": left["auroc"] - right["auroc"],
        "pauc_drop": left["pauc_2pct"] - right["pauc_2pct"],
    }
