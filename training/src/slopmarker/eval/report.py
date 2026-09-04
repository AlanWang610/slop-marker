"""Render measured numbers as markdown for docs/measurements/.

The README reserves that directory for numbers we actually measured rather than
estimated, so everything here comes from a scores file or a run summary.
"""

from __future__ import annotations

from typing import Any


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def per_genre_table(per_genre: list[dict[str, Any]]) -> str:
    rows = [
        [
            str(g["genre"]),
            str(g["n_human"]),
            str(g["n_ai"]),
            f"{g['fpr']:.4f}",
            f"{g['fpr_upper']:.4f}",
            f"{g['recall']:.3f}",
            f"{g['auc']:.3f}" if g.get("auc") == g.get("auc") else "n/a",
        ]
        for g in sorted(per_genre, key=lambda g: -g["fpr"])
    ]
    return _table(
        ["genre", "human", "AI", "FPR", "FPR upper", "recall", "AUROC"],
        rows,
    )


def calibration_report(report: dict[str, Any]) -> str:
    cal = report.get("calibration", {})
    verification = report.get("test_verification", {})
    parity = report.get("parity", {})
    quant = report.get("quantization", {})

    parts = [
        f"# Calibration and release gates -- {report.get('version', 'unknown')}",
        "",
        "## Operating point",
        "",
        _table(
            ["quantity", "value"],
            [
                ["temperature", f"{cal.get('temperature', float('nan')):.4f}"],
                ["t_on", f"{cal.get('t_on', float('nan')):.4f}"],
                ["t_off", f"{cal.get('t_off', float('nan')):.4f}"],
                ["binding genre", str(cal.get("binding_genre"))],
                ["recall (shipped global threshold)", f"{cal.get('recall_global', 0):.3f}"],
                ["recall (per-genre oracle)", f"{cal.get('recall_oracle', 0):.3f}"],
                ["**oracle gap**", f"**{cal.get('oracle_gap', 0):.3f}**"],
            ],
        ),
        "",
        "The oracle gap is the recall given up by shipping one global threshold instead",
        "of a per-genre vector. It is the measured price of dropping the genre head at",
        "export (scope.md 4.4), not an argument about it.",
        "",
        "## Per-genre false-positive rate (calibration split)",
        "",
        per_genre_table(cal.get("per_genre", [])),
        "",
        "## Verification on the test split",
        "",
        _table(
            ["quantity", "value"],
            [
                ["overall FPR", f"{verification.get('overall_fpr', 0):.4f}"],
                ["recall", f"{verification.get('recall', 0):.3f}"],
                ["AUROC", f"{verification.get('auroc', 0):.4f}"],
                [
                    "worst genre FPR (Clopper-Pearson upper)",
                    f"{verification.get('worst_genre_fpr_upper', 0):.4f}",
                ],
            ],
        ),
        "",
        "## Export",
        "",
        _table(
            ["check", "value"],
            [
                ["fp32 vs PyTorch max |delta|", f"{parity.get('max_abs_diff', 0):.2e}"],
                ["parity across shapes", "pass" if parity.get("passed") else "FAIL"],
                ["fp32 size (MB)", str(quant.get("fp32_mb"))],
                ["int8 size (MB)", str(quant.get("int8_mb"))],
                ["compression", str(quant.get("compression"))],
                ["MatMulInteger nodes", str(quant.get("matmul_integer_nodes"))],
                ["quantization verified", "pass" if quant.get("passed") else "FAIL"],
            ],
        ),
    ]
    if parity.get("by_shape"):
        parts += [
            "",
            "Parity by shape (a single-length check would miss a frozen attention mask):",
            "",
            _table(
                ["shape", "max |delta|"],
                [[k, f"{v:.2e}"] for k, v in sorted(parity["by_shape"].items())],
            ),
        ]
    return "\n".join(parts) + "\n"


def corpus_report(manifest: dict[str, Any], probe: dict[str, Any] | None = None) -> str:
    parts = [
        f"# Corpus -- {manifest.get('version', 'unknown')}",
        "",
        _table(
            ["quantity", "value"],
            [
                ["documents harvested", str(manifest.get("harvested"))],
                ["exact duplicates removed", str(manifest.get("exact_duplicates_removed"))],
                ["near-duplicate clusters", str(manifest.get("clusters"))],
                ["genre assigned from text", str(manifest.get("genre_relabelled_from_text"))],
                ["RAID-contaminated removed", str(manifest.get("raid_contaminated_removed"))],
                ["documents after caps", str(manifest.get("after_cluster_cap"))],
                ["training windows", str(manifest.get("windows"))],
            ],
        ),
        "",
        "## Windows by genre",
        "",
        _table(
            ["genre", "windows", "AI rate"],
            [
                [genre, str(n), f"{manifest.get('genre_ai_rate', {}).get(genre, 0):.3f}"]
                for genre, n in sorted(
                    manifest.get("windows_by_genre", {}).items(), key=lambda kv: -kv[1]
                )
            ],
        ),
        "",
        "An AI rate far from 0.5 in any genre lets the model learn genre implies label,",
        "which is what actually produces the 'formal register implies AI' shortcut.",
        "",
        "## Windows by split",
        "",
        _table(
            ["split", "windows"],
            [[k, str(v)] for k, v in sorted(manifest.get("windows_by_split", {}).items())],
        ),
    ]
    if probe:
        parts += [
            "",
            "## Corpus gate",
            "",
            f"Result: **{'pass' if probe.get('passed') else 'FAIL'}**",
            "",
            _table(
                ["probe", "AUC", "band", "verdict", "why it matters"],
                [
                    [
                        p["name"],
                        f"{p['auc']:.3f}",
                        f"[{p['low']:.2f}, {p['high']:.2f}]",
                        "pass" if p["passed"] else "**FAIL**",
                        p.get("note", ""),
                    ]
                    for p in probe.get("probes", [])
                ],
            ),
            "",
            "Bands, not ceilings: a bag-of-words model at 0.99 means the corpus is",
            "trivially separable, and at 0.55 it means the classes were preprocessed",
            "differently or over-scrubbed. Both are failures.",
        ]
    return "\n".join(parts) + "\n"
