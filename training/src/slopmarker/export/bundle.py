"""Assemble the shipped model bundle (scope.md 5, artifacts/README.md).

Exactly six files. Provenance is written *beside* the directory rather than inside it,
because the extension verifies every file it finds against SHA256SUMS and an extra file
is an invitation to a mismatch.

A bundle is never edited in place -- a change means a new version string, which is what
invalidates cached scores on both browsers. The directory is made read-only after
assembly so that convention is enforced by the filesystem rather than by memory.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
from pathlib import Path
from typing import Any

BUNDLE_FILES = (
    "model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "calibration.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sha256sums(directory: Path) -> Path:
    """coreutils format, LF endings, sorted, excluding itself."""
    lines = [
        f"{sha256_file(directory / name)}  {name}"
        for name in sorted(BUNDLE_FILES)
        if (directory / name).exists()
    ]
    path = directory / "SHA256SUMS"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def verify_sha256sums(directory: Path) -> list[str]:
    """Return mismatches. Empty means the bundle is intact."""
    sums = directory / "SHA256SUMS"
    if not sums.exists():
        return ["SHA256SUMS missing"]
    problems = []
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split("  ", 1)
        path = directory / name
        if not path.exists():
            problems.append(f"{name}: missing")
        elif sha256_file(path) != expected:
            problems.append(f"{name}: checksum mismatch")
    return problems


def model_version(semver: str, git_sha: str) -> str:
    return f"mb-base-{semver}-{git_sha[:7]}"


def assemble(
    *,
    int8_model: Path,
    tokenizer_dir: Path,
    calibration_path: Path,
    out_dir: Path,
    provenance: dict[str, Any] | None = None,
    read_only: bool = True,
    gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the six-file bundle. Refuses unless the release gate passed.

    The gate is a required argument in practice rather than a courtesy check: the
    first bundle this function produced held a model that had lost twenty points of
    AUROC to quantization, and every structural check upstream of here passed on it.
    """
    if gate is None:
        raise RuntimeError("assemble requires a release gate result; see export.gates")
    if not gate.get("passed"):
        raise RuntimeError(f"release gate failed: {gate.get('failures', [])}")

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(int8_model, out_dir / "model.onnx")
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        source = tokenizer_dir / name
        if source.exists():
            shutil.copy2(source, out_dir / name)
    shutil.copy2(calibration_path, out_dir / "calibration.json")

    missing = [name for name in BUNDLE_FILES if not (out_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"bundle incomplete: {missing}")
    extra = sorted(p.name for p in out_dir.iterdir() if p.name not in {*BUNDLE_FILES, "SHA256SUMS"})
    if extra:
        raise RuntimeError(f"bundle must contain exactly six files; found extra: {extra}")

    write_sha256sums(out_dir)
    if provenance is not None:
        # Beside, not inside: the extension checksums everything in the directory.
        sibling = out_dir.parent / f"{out_dir.name}.provenance.json"
        sibling.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    size_mb = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6
    if read_only:
        for path in out_dir.iterdir():
            path.chmod(path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)

    return {
        "path": str(out_dir),
        "files": sorted(p.name for p in out_dir.iterdir()),
        "size_mb": round(size_mb, 1),
    }
