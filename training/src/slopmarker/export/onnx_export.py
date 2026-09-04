"""ONNX export, parity checking and int8 quantization (scope.md 5).

Two things here are load-bearing and neither is obvious.

**Parity must be checked at several sequence lengths, not just at 512.** ModernBERT
builds its sliding-window attention mask from `torch.arange(seq_len)`, which is a strong
constant-folding candidate under the tracer. If it folds, the local mask freezes at the
export length and the model returns silently wrong logits at every *other* length. A
512-only check cannot see that, and nothing else would catch it before release.

**The quantization op list decides the bundle size, and scope.md's arithmetic was
wrong.** ModernBERT-base is 149M parameters, so fp32 is ~596MB -- that part is right.
But its 38.7M embedding parameters appear in the graph as a `Gather`, not a `MatMul`, so
quantizing MatMul alone leaves them in fp32 and lands at ~266MB rather than the ~150MB
the spec quotes. `Gather` is included here, gated on a measured accuracy delta, because
per-tensor int8 across a 50k-row embedding table is the riskiest step in this pipeline
for a detector whose signal is lexical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

OPSET = 17
PARITY_SEQ_LENS = (64, 128, 256, 384, 512)
PARITY_BATCH_SIZES = (1, 4)
PARITY_TOLERANCE = 1e-3
# 22 layers x {Wqkv, Wo, Wi, Wo} -- if far fewer survive, quantization silently skipped.
MIN_MATMUL_INTEGER_NODES = 80


@dataclass
class ParityReport:
    max_abs_diff: float
    tolerance: float
    by_shape: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.max_abs_diff <= self.tolerance

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs_diff": self.max_abs_diff,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "by_shape": self.by_shape,
        }


def export_fp32(checkpoint: Path, out_dir: Path, opset: int = OPSET) -> Path:
    """Export the stripped classifier with dynamic batch and sequence axes."""
    from optimum.exporters.onnx import main_export

    out_dir.mkdir(parents=True, exist_ok=True)
    main_export(
        model_name_or_path=str(checkpoint),
        output=str(out_dir),
        task="text-classification",
        opset=opset,
        device="cpu",
        framework="pt",
        do_validation=False,  # we run our own, across shapes
        no_dynamic_axes=False,
        # eager, and reference_compile off: ModernBERT only auto-disables its own
        # compile path on CPU/MPS, and the tracer cannot follow a compiled encoder.
        model_kwargs={"attn_implementation": "eager", "reference_compile": False},
    )
    return out_dir / "model.onnx"


def check_parity(
    checkpoint: Path,
    onnx_path: Path,
    texts: list[str],
    *,
    seq_lens: tuple[int, ...] = PARITY_SEQ_LENS,
    batch_sizes: tuple[int, ...] = PARITY_BATCH_SIZES,
    tolerance: float = PARITY_TOLERANCE,
) -> ParityReport:
    """Compare ONNX logits with PyTorch across shapes.

    The shape sweep is the point. A single-length check would pass even if the
    sliding-window mask had been frozen at that length.
    """
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, attn_implementation="eager"
    ).eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    worst = 0.0
    by_shape: dict[str, float] = {}
    for seq_len in seq_lens:
        for batch in batch_sizes:
            sample = texts[:batch] or ["placeholder text"] * batch
            while len(sample) < batch:
                sample = sample + sample
            sample = sample[:batch]
            encoded = tokenizer(
                sample,
                padding="max_length",
                truncation=True,
                max_length=seq_len,
                return_tensors="np",
            )
            feeds = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
            onnx_logits = session.run(None, feeds)[0]
            with torch.no_grad():
                torch_logits = model(
                    input_ids=torch.tensor(feeds["input_ids"]),
                    attention_mask=torch.tensor(feeds["attention_mask"]),
                ).logits.numpy()
            diff = float(np.max(np.abs(onnx_logits - torch_logits)))
            by_shape[f"seq{seq_len}_batch{batch}"] = diff
            worst = max(worst, diff)

    return ParityReport(max_abs_diff=worst, tolerance=tolerance, by_shape=by_shape)


def quantize_int8(
    fp32_path: Path,
    int8_path: Path,
    *,
    quantize_embeddings: bool = True,
    per_channel: bool = False,
) -> dict[str, Any]:
    """Dynamic int8 quantization, then assert it actually happened.

    There is a known ONNX Runtime regression where transformer MatMuls silently stop
    being quantized and the output file comes back the same size as the input, with no
    error raised anywhere. Trusting the call is not enough.

    Note that these assertions establish only that quantization *ran*. Whether the
    result still separates the classes is a question no property of the graph can
    answer; `gates.compare_scores` answers it, and `gates.release_gate` enforces it.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    op_types = ["MatMul", "Gather"] if quantize_embeddings else ["MatMul"]
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=op_types,
        per_channel=per_channel,
        extra_options={"MatMulConstBOnly": False},
    )
    report = verify_quantized(fp32_path, int8_path, op_types)
    report["per_channel"] = per_channel
    return report


def verify_quantized(fp32_path: Path, int8_path: Path, op_types: list[str]) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(int8_path), load_external_data=False)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1

    fp32_mb = fp32_path.stat().st_size / 1e6
    int8_mb = int8_path.stat().st_size / 1e6
    matmul_integer = counts.get("MatMulInteger", 0)
    report: dict[str, Any] = {
        "fp32_mb": round(fp32_mb, 1),
        "int8_mb": round(int8_mb, 1),
        "compression": round(fp32_mb / int8_mb, 2) if int8_mb else 0.0,
        "op_types_requested": op_types,
        "matmul_integer_nodes": matmul_integer,
        "dynamic_quantize_nodes": counts.get("DynamicQuantizeLinear", 0),
        "quantized_gather": counts.get("Gather", 0),
    }
    problems: list[str] = []
    if matmul_integer < MIN_MATMUL_INTEGER_NODES:
        problems.append(
            f"only {matmul_integer} MatMulInteger nodes "
            f"(expected >= {MIN_MATMUL_INTEGER_NODES}); quantization was skipped silently"
        )
    if int8_mb > fp32_mb * 0.75:
        problems.append(f"int8 file is {int8_mb:.0f}MB against fp32 {fp32_mb:.0f}MB; no shrink")
    report["problems"] = problems
    report["passed"] = not problems
    return report


def verify_graph(onnx_path: Path, tokenizer_dir: Path) -> dict[str, Any]:
    """Structural assertions on the exported graph and the tokenizer beside it."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    problems: list[str] = []

    opsets = {imp.domain or "ai.onnx": imp.version for imp in model.opset_import}
    if opsets.get("ai.onnx", 0) < OPSET:
        problems.append(f"opset {opsets.get('ai.onnx')} below {OPSET}")
    foreign = sorted(d for d in opsets if d not in ("ai.onnx", "ai.onnx.ml", "com.microsoft"))
    if foreign:
        problems.append(f"unexpected operator domains: {foreign}")

    inputs = {i.name for i in model.graph.input}
    if inputs != {"input_ids", "attention_mask"}:
        problems.append(f"unexpected graph inputs: {sorted(inputs)}")

    # Dynamic axes are what let short chunks cost less than 512 tokens.
    for graph_input in model.graph.input:
        dims = graph_input.type.tensor_type.shape.dim
        if not all(d.dim_param for d in dims):
            problems.append(f"{graph_input.name} has a static axis: {dims}")

    config_path = tokenizer_dir / "tokenizer_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model_max_length") != 512:
            problems.append(
                f"tokenizer model_max_length is {config.get('model_max_length')}, not 512; "
                "the browser could feed 8k-token inputs to a 512-token graph"
            )
    return {"opsets": opsets, "problems": problems, "passed": not problems}
