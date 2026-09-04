"""The node-count assertion, which passed on a model that had lost 19 points of AUROC.

The original check was `matmul_integer >= 80` with no upper bound. ModernBERT-base has
22 layers x 4 weight MatMuls plus a two-MatMul head, so 90 is correct; the shipped
export produced 136 because MatMulConstBOnly was False and the attention products got
quantized too. A floor cannot tell those apart, so both ends are now bounded.

Sizes here are proportional rather than realistic: verify_quantized only stats the fp32
path, so what it checks is the ratio, and a test does not need to write 599MB to
exercise it.
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper

from slopmarker.export.onnx_export import (
    MAX_MATMUL_INTEGER_NODES,
    MIN_MATMUL_INTEGER_NODES,
    verify_quantized,
)


def graph_with(n_nodes: int, path: Path) -> Path:
    """A structurally valid graph carrying a chosen number of MatMulInteger nodes."""
    nodes = [
        helper.make_node("MatMulInteger", ["a", "b"], [f"out{i}"], name=f"mmi{i}")
        for i in range(n_nodes)
    ]
    graph = helper.make_graph(
        nodes,
        "g",
        [
            helper.make_tensor_value_info("a", TensorProto.UINT8, [1, 8]),
            helper.make_tensor_value_info("b", TensorProto.UINT8, [8, 8]),
        ],
        [helper.make_tensor_value_info("out0", TensorProto.INT32, [1, 8])],
    )
    onnx.save(helper.make_model(graph), str(path))
    return path


def big(path: Path, size: int = 4_000_000) -> Path:
    """Stands in for the fp32 model. Never parsed, only stat()ed."""
    path.write_bytes(b"\0" * size)
    return path


def test_the_correct_node_count_passes(tmp_path: Path) -> None:
    report = verify_quantized(
        big(tmp_path / "fp32.bin"),
        graph_with(90, tmp_path / "int8.onnx"),
        ["MatMul", "Gather"],
    )
    assert report["passed"], report["problems"]
    assert report["matmul_integer_nodes"] == 90


def test_too_few_nodes_is_a_silent_skip(tmp_path: Path) -> None:
    report = verify_quantized(
        big(tmp_path / "fp32.bin"),
        graph_with(MIN_MATMUL_INTEGER_NODES - 1, tmp_path / "int8.onnx"),
        ["MatMul"],
    )
    assert not report["passed"]
    assert any("skipped silently" in p for p in report["problems"])


def test_too_many_nodes_is_the_r1_failure(tmp_path: Path) -> None:
    """136 nodes: the count the first export produced and the old check waved through."""
    assert MAX_MATMUL_INTEGER_NODES < 136
    report = verify_quantized(
        big(tmp_path / "fp32.bin"),
        graph_with(136, tmp_path / "int8.onnx"),
        ["MatMul", "Gather"],
    )
    assert not report["passed"]
    assert any("MatMulConstBOnly" in p for p in report["problems"])


def test_no_shrink_is_caught(tmp_path: Path) -> None:
    """A quantized file the same size as its input means nothing was quantized."""
    int8 = graph_with(90, tmp_path / "int8.onnx")
    fp32 = big(tmp_path / "fp32.bin", size=int8.stat().st_size)
    report = verify_quantized(fp32, int8, ["MatMul"])
    assert not report["passed"]
    assert any("no shrink" in p for p in report["problems"])


def test_const_b_only_defaults_to_true() -> None:
    """The default is the whole fix; a caller must opt in to the destructive setting."""
    import inspect

    from slopmarker.export.onnx_export import quantize_int8

    assert inspect.signature(quantize_int8).parameters["const_b_only"].default is True
