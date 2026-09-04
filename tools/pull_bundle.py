#!/usr/bin/env python
"""Copy a model bundle off the Modal Volume into artifacts/bundles/ and verify it.

The bundle is the unit of release (see artifacts/README.md). It never enters git; this
script is how a developer machine gets a local copy to benchmark, to test the extension
against, and to publish from.

    uv run --extra modal python ../tools/pull_bundle.py --version mb-base-0.2.0-dev

Verification is not optional: every file is checked against the bundle's own SHA256SUMS
before the script reports success, which is the same check the extension performs on the
downloaded copy (scope.md 6.4).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VOLUME = "slopmarker-data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="bundle version, e.g. mb-base-0.2.0-dev")
    ap.add_argument("--volume", default=VOLUME)
    ap.add_argument(
        "--out", type=Path, default=None, help="defaults to <repo>/artifacts/bundles/<version>"
    )
    args = ap.parse_args()

    out: Path = args.out or (REPO / "artifacts" / "bundles" / args.version)
    out.mkdir(parents=True, exist_ok=True)

    remote = f"artifacts/bundles/{args.version}"
    print(f"pulling {args.volume}:{remote} -> {out}")
    # modal prints a U+2713 on success; the Windows console default codec cannot encode
    # it and the CLI dies on its own status line. Force UTF-8 rather than lose the download.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        ["modal", "volume", "get", "--force", args.volume, remote, str(out.parent)],
        cwd=REPO,
        env=env,
    )
    if proc.returncode != 0:
        print("modal volume get failed", file=sys.stderr)
        return proc.returncode

    sys.path.insert(0, str(REPO / "training" / "src"))
    from slopmarker.export.bundle import BUNDLE_FILES, verify_sha256sums

    missing = [f for f in (*BUNDLE_FILES, "SHA256SUMS") if not (out / f).is_file()]
    if missing:
        print(f"bundle is incomplete, missing: {', '.join(missing)}", file=sys.stderr)
        return 1

    problems = verify_sha256sums(out)
    if problems:
        for p in problems:
            print(f"  CHECKSUM: {p}", file=sys.stderr)
        return 1

    total = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(f"verified {len(BUNDLE_FILES) + 1} files, {total / 1e6:.1f} MB, all checksums match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
