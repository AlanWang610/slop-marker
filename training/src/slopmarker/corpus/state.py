"""Shard I/O and resume markers.

Every mapped stage is a pure function of (input shard, config hash). A stage checks for
its marker first and returns immediately if present, so a crashed run costs only the
shards that were in flight. Because the config hash is in the marker path, editing the
config invalidates exactly the affected stages.

Writes go to a temporary file and are renamed into place, so a killed container can
never leave a half-written shard that a later run would treat as complete.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import zstandard


def shard_path(root: Path, stage: str, name: str) -> Path:
    return root / stage / f"{name}.jsonl.zst"


def marker_path(root: Path, stage: str, config_hash: str, name: str) -> Path:
    return root / "_state" / stage / config_hash / f"{name}.done"


def is_done(root: Path, stage: str, config_hash: str, name: str) -> bool:
    return marker_path(root, stage, config_hash, name).exists()


def mark_done(root: Path, stage: str, config_hash: str, name: str, summary: dict[str, Any]) -> None:
    path = marker_path(root, stage, config_hash, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary), encoding="utf-8")


def read_summary(root: Path, stage: str, config_hash: str, name: str) -> dict[str, Any]:
    text = marker_path(root, stage, config_hash, name).read_text(encoding="utf-8")
    summary: dict[str, Any] = json.loads(text)
    return summary


def write_shard(path: Path, rows: Iterable[str]) -> int:
    """Write JSONL lines, zstd-compressed, atomically. Returns the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    compressor = zstandard.ZstdCompressor(level=10)
    with tmp.open("wb") as raw, compressor.stream_writer(raw) as out:
        for row in rows:
            out.write(row.encode("utf-8"))
            out.write(b"\n")
            count += 1
    os.replace(tmp, path)
    return count


def read_shard(path: Path) -> Iterator[str]:
    decompressor = zstandard.ZstdDecompressor()
    with path.open("rb") as raw, decompressor.stream_reader(raw) as stream:
        buffer = b""
        while True:
            block = stream.read(1 << 20)
            if not block:
                break
            buffer += block
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                if line:
                    yield line.decode("utf-8")
        if buffer:
            yield buffer.decode("utf-8")


def clear_stage(root: Path, stage: str, config_hash: str) -> int:
    """Drop a stage's markers so it reruns. Returns how many were removed."""
    directory = root / "_state" / stage / config_hash
    if not directory.exists():
        return 0
    markers = list(directory.glob("*.done"))
    for marker in markers:
        marker.unlink()
    return len(markers)
