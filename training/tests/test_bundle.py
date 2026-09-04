"""Bundle assembly and checksums."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slopmarker.eval.calibration import AggregateParams, Calibration
from slopmarker.export.bundle import (
    BUNDLE_FILES,
    assemble,
    model_version,
    sha256_file,
    verify_sha256sums,
    write_sha256sums,
)


@pytest.fixture
def parts(tmp_path: Path) -> dict[str, Path]:
    tokenizer = tmp_path / "ckpt"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    (tokenizer / "tokenizer_config.json").write_text('{"model_max_length":512}', encoding="utf-8")
    (tokenizer / "config.json").write_text('{"num_labels":1}', encoding="utf-8")
    model = tmp_path / "model.int8.onnx"
    model.write_bytes(b"\x08\x01onnx-bytes")
    cal = tmp_path / "calibration.json"
    Calibration(
        version="mb-base-1.0.0-abc1234",
        temperature=1.18,
        t_on=0.96,
        t_off=0.90,
        aggregate=AggregateParams(),
    ).save(cal)
    return {"model": model, "tokenizer": tokenizer, "cal": cal, "root": tmp_path}


PASSING = {"passed": True, "failures": []}


def test_model_version_format() -> None:
    assert model_version("1.0.0", "abc1234def") == "mb-base-1.0.0-abc1234"


def test_assemble_produces_exactly_six_files(parts: dict[str, Path]) -> None:
    out = parts["root"] / "bundles" / "v1"
    result = assemble(
        int8_model=parts["model"],
        tokenizer_dir=parts["tokenizer"],
        calibration_path=parts["cal"],
        out_dir=out,
        gate=PASSING,
        read_only=False,
    )
    assert set(result["files"]) == {*BUNDLE_FILES, "SHA256SUMS"}
    assert len(result["files"]) == 6


def test_provenance_is_written_beside_not_inside(parts: dict[str, Path]) -> None:
    """The extension checksums every file it finds; an extra one is a mismatch."""
    out = parts["root"] / "bundles" / "v1"
    assemble(
        int8_model=parts["model"],
        tokenizer_dir=parts["tokenizer"],
        calibration_path=parts["cal"],
        out_dir=out,
        gate=PASSING,
        provenance={"run_id": "r1"},
        read_only=False,
    )
    assert not (out / "provenance.json").exists()
    sibling = out.parent / "v1.provenance.json"
    assert json.loads(sibling.read_text(encoding="utf-8"))["run_id"] == "r1"


def test_checksums_round_trip(parts: dict[str, Path]) -> None:
    out = parts["root"] / "bundles" / "v1"
    assemble(
        int8_model=parts["model"],
        tokenizer_dir=parts["tokenizer"],
        calibration_path=parts["cal"],
        out_dir=out,
        gate=PASSING,
        read_only=False,
    )
    assert verify_sha256sums(out) == []


def test_checksums_catch_a_truncated_copy(parts: dict[str, Path]) -> None:
    """A truncated 150MB copy is a plausible failure only the checksum catches."""
    out = parts["root"] / "bundles" / "v1"
    assemble(
        int8_model=parts["model"],
        tokenizer_dir=parts["tokenizer"],
        calibration_path=parts["cal"],
        out_dir=out,
        gate=PASSING,
        read_only=False,
    )
    (out / "model.onnx").write_bytes(b"\x08\x01onnx")  # truncated
    assert any("model.onnx" in p for p in verify_sha256sums(out))


def test_missing_file_is_reported(parts: dict[str, Path]) -> None:
    out = parts["root"] / "bundles" / "v1"
    assemble(
        int8_model=parts["model"],
        tokenizer_dir=parts["tokenizer"],
        calibration_path=parts["cal"],
        out_dir=out,
        gate=PASSING,
        read_only=False,
    )
    (out / "config.json").unlink()
    assert any("config.json" in p for p in verify_sha256sums(out))


def test_incomplete_input_raises(parts: dict[str, Path]) -> None:
    (parts["tokenizer"] / "config.json").unlink()
    with pytest.raises(FileNotFoundError, match="bundle incomplete"):
        assemble(
            int8_model=parts["model"],
            tokenizer_dir=parts["tokenizer"],
            calibration_path=parts["cal"],
            out_dir=parts["root"] / "bundles" / "v1",
            gate=PASSING,
            read_only=False,
        )


def test_sha256sums_uses_lf_and_two_spaces(parts: dict[str, Path], tmp_path: Path) -> None:
    directory = tmp_path / "b"
    directory.mkdir()
    for name in BUNDLE_FILES:
        (directory / name).write_text(name, encoding="utf-8")
    raw = write_sha256sums(directory).read_bytes()
    assert b"\r\n" not in raw
    first = raw.decode().splitlines()[0]
    assert "  " in first
    assert first.split("  ")[0] == sha256_file(directory / sorted(BUNDLE_FILES)[0])


def test_bundle_is_read_only_after_assembly(parts: dict[str, Path]) -> None:
    """artifacts/README.md says a bundle is never edited in place; enforce it."""
    out = parts["root"] / "bundles" / "v1"
    assemble(
        int8_model=parts["model"],
        tokenizer_dir=parts["tokenizer"],
        calibration_path=parts["cal"],
        out_dir=out,
        gate=PASSING,
    )
    import os

    assert not os.access(out / "model.onnx", os.W_OK)
