"""Publish a model bundle to the static host the extension fetches from (scope.md 6.4).

    # show exactly what would happen, change nothing:
    uv run --project training python tools/publish_bundle.py --version mb-base-0.2.0-dev

    # actually publish:
    uv run --project training python tools/publish_bundle.py --version mb-base-0.2.0-dev --yes

Dry run is the default because this is a public, hard-to-reverse action: it creates a
GitHub release and uploads a 137 MB model that anyone can then download.

The host is GitHub Releases with the release tag set to the model version, so the layout
the extension expects falls out for free:

    https://github.com/<owner>/<repo>/releases/download/<version>/<file>

model-host/README.md states the contract this implements, and the last step is the one
that matters: after uploading, every file is downloaded back from its public URL and
hashed. A bundle is not published until the *served* bytes match SHA256SUMS. Uploading and
assuming is how a corrupted or truncated asset ships, and the extension would then fail
closed on every user's first run with nothing to point at.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STAGING = REPO / "model-host"

# The whole bundle is published, not just what the extension downloads: SHA256SUMS is only
# useful to a third party if the files it names are all there.
PUBLISH_FILES = (
    "model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "calibration.json",
    "SHA256SUMS",
)

NOTES = """Model bundle `{version}` for the slop-marker extension.

Downloaded on first run and verified against the `SHA256SUMS` built into the extension
(scope.md 6.4). Not intended to be installed by hand.

| | |
|---|---|
| architecture | ModernBERT-base, single AI-fraction logit |
| quantization | weight-only, int8 encoder + int4 embedding table |
| `model.onnx` | {size_mb:.1f} MB |
| temperature | {temperature} |
| `t_on` / `t_off` | {t_on} / {t_off} |
| max sequence length | {max_length} |

Measurements are in [`docs/measurements/`](../../tree/main/docs/measurements): per-genre
FPR with Clopper-Pearson bounds, the global-vs-oracle recall gap, int8 parity, RAID, and
document-level FPR.

Verify a download against the checksums in this release:

```
sha256sum -c SHA256SUMS
```
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gh(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gh", *args], cwd=REPO, capture_output=True, text=True, check=check, encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--repo", default=None, help="owner/name; defaults to the origin remote")
    ap.add_argument("--yes", action="store_true", help="actually publish (default is a dry run)")
    ap.add_argument("--bundle", type=Path, default=None)
    args = ap.parse_args()

    bundle: Path = args.bundle or (REPO / "artifacts" / "bundles" / args.version)
    if not bundle.is_dir():
        print(f"no bundle at {bundle}", file=sys.stderr)
        return 1

    repo = args.repo
    if repo is None:
        info = json.loads(gh("repo", "view", "--json", "nameWithOwner").stdout)
        repo = info["nameWithOwner"]

    # ---- verify locally before anything leaves the machine -------------------------
    sys.path.insert(0, str(REPO / "training" / "src"))
    from slopmarker.export.bundle import verify_sha256sums

    problems = verify_sha256sums(bundle)
    if problems:
        for p in problems:
            print(f"  CHECKSUM: {p}", file=sys.stderr)
        print("refusing to publish an unverified bundle", file=sys.stderr)
        return 1

    missing = [f for f in PUBLISH_FILES if not (bundle / f).is_file()]
    if missing:
        print(f"bundle is incomplete, missing: {', '.join(missing)}", file=sys.stderr)
        return 1

    expected = {f: sha256_file(bundle / f) for f in PUBLISH_FILES if f != "SHA256SUMS"}
    total_mb = sum((bundle / f).stat().st_size for f in PUBLISH_FILES) / 1e6
    base_url = f"https://github.com/{repo}/releases/download"

    calibration = json.loads((bundle / "calibration.json").read_text(encoding="utf-8"))
    if calibration["version"] != args.version:
        print(
            f"calibration.json says {calibration['version']}, not {args.version}", file=sys.stderr
        )
        return 1

    print(f"bundle    {bundle}")
    print(f"repo      {repo}")
    print(f"tag       {args.version}")
    print(f"files     {len(PUBLISH_FILES)}, {total_mb:.1f} MB")
    print(f"base URL  {base_url}")
    for name, digest in expected.items():
        print(f"            {digest[:16]}...  {name}")

    if not args.yes:
        print("\nDRY RUN. Nothing was uploaded. Re-run with --yes to publish.")
        print("This creates a PUBLIC release; the model becomes downloadable by anyone.")
        return 0

    # ---- stage, per model-host/README.md -------------------------------------------
    staged = STAGING / args.version
    staged.mkdir(parents=True, exist_ok=True)
    for name in PUBLISH_FILES:
        shutil.copy2(bundle / name, staged / name)
    print(f"\nstaged to {staged}")

    # ---- create the release and upload ---------------------------------------------
    exists = gh("release", "view", args.version, "--repo", repo, check=False).returncode == 0
    if not exists:
        notes = NOTES.format(
            version=args.version,
            size_mb=(bundle / "model.onnx").stat().st_size / 1e6,
            temperature=calibration["temperature"],
            t_on=calibration["t_on"],
            t_off=calibration["t_off"],
            max_length=calibration["max_length"],
        )
        notes_file = staged / "RELEASE_NOTES.md"
        notes_file.write_text(notes, encoding="utf-8", newline="\n")
        gh(
            "release", "create", args.version,
            "--repo", repo,
            "--title", f"Model bundle {args.version}",
            "--notes-file", str(notes_file),
        )
        print(f"created release {args.version}")
    else:
        print(f"release {args.version} already exists; uploading over it")

    gh(
        "release", "upload", args.version,
        *[str(staged / f) for f in PUBLISH_FILES],
        "--repo", repo, "--clobber",
    )
    print("uploaded")

    # ---- the step that makes this trustworthy --------------------------------------
    print("\nverifying the SERVED bytes:")
    bad = 0
    for name, want in expected.items():
        url = f"{base_url}/{args.version}/{name}"
        digest = hashlib.sha256()
        with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https host
            for block in iter(lambda: response.read(1 << 20), b""):
                digest.update(block)
        got = digest.hexdigest()
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BAD '} {name}")
    if bad:
        print(f"\n{bad} file(s) do not match what was uploaded. Do not ship this.", file=sys.stderr)
        return 1

    print(f"\npublished and verified. Point the extension at it:")
    print(f"  uv run --project training python tools/sync_extension_assets.py \\")
    print(f"      --bundle {bundle} --base-url {base_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
