"""Drive the post-corpus pipeline: gate, train, export, calibrate, report.

    uv run python scripts/run_pipeline.py --stage gate    --version v1
    uv run python scripts/run_pipeline.py --stage train   --version v1 --run-id r1
    uv run python scripts/run_pipeline.py --stage export  --version v1 --run-id r1

The gate is separate and comes first on purpose: no training run should start against
a corpus that is separable for reasons unrelated to AI writing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

CORPUS_APP = "slopmarker-corpus"
TRAIN_APP = "slopmarker-train"
MEASUREMENTS = Path(__file__).resolve().parents[2] / "docs" / "measurements"


def _write(name: str, text: str) -> None:
    MEASUREMENTS.mkdir(parents=True, exist_ok=True)
    (MEASUREMENTS / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote docs/measurements/{name}")


def gate(version: str) -> None:
    from slopmarker.eval.report import corpus_report

    probe = modal.Function.from_name(CORPUS_APP, "probe_corpus").remote(version)
    print(json.dumps({k: v for k, v in probe.items() if k != "probes"}, indent=2))
    for row in probe.get("probes", []):
        mark = "pass" if row["passed"] else "FAIL"
        print(
            f"  {row['name']:34} auc={row['auc']:.3f} [{row['low']:.2f},{row['high']:.2f}] {mark}"
        )

    manifest_path = Path("manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    _write(f"corpus-{version}.md", corpus_report({**manifest, "version": version}, probe))
    if not probe.get("passed"):
        raise SystemExit("corpus gate FAILED -- do not train against this corpus")


def train(version: str, run_id: str, overrides: dict | None) -> None:
    result = modal.Function.from_name(TRAIN_APP, "train_model").remote(
        version, run_id, overrides or {}
    )
    print(json.dumps(result, indent=2, default=str))


def export(version: str, run_id: str, bundle_version: str) -> None:
    from slopmarker.eval.report import calibration_report

    report = modal.Function.from_name(TRAIN_APP, "export_and_calibrate").remote(
        run_id, version, bundle_version
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "calibration"}, indent=2, default=str)
    )
    cal = report.get("calibration", {})
    if cal:
        print(f"\nt_on={cal['t_on']:.4f} temperature={cal['temperature']:.4f}")
        print(f"binding genre: {cal['binding_genre']}")
        print(
            f"recall global={cal['recall_global']:.3f} oracle={cal['recall_oracle']:.3f} "
            f"gap={cal['oracle_gap']:.3f}"
        )
    _write(f"calibration-{bundle_version}.md", calibration_report(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["gate", "train", "export"])
    parser.add_argument("--version", default="v1")
    parser.add_argument("--run-id", default="run1")
    # No default. It was mb-base-0.1.0-dev, which has been withdrawn -- there is no bundle
    # and no release for it -- so exporting without the flag assembled an artifact under a
    # dead version and overwrote that version's calibration report. A bundle version is an
    # identity, and naming it is the one thing an export must not guess.
    parser.add_argument("--bundle-version")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-every", type=int)
    args = parser.parse_args()

    if args.stage == "gate":
        gate(args.version)
    elif args.stage == "train":
        overrides = {
            k: v
            for k, v in (
                ("epochs", args.epochs),
                ("batch_size", args.batch_size),
                ("eval_every", args.eval_every),
            )
            if v is not None
        }
        train(args.version, args.run_id, overrides)
    else:
        if args.bundle_version is None:
            parser.error("--bundle-version is required for --stage export")
        export(args.version, args.run_id, args.bundle_version)


if __name__ == "__main__":
    main()
