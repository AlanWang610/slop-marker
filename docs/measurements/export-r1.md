# Export and quantization -- run r1

## Summary

The first export produced a complete, checksummed six-file bundle containing a model
that had lost twenty points of AUROC. It was assembled because nothing in the export
path scores the model, and the release gate that would have caught it was specified,
written into the plan, and never implemented.

| artifact | split | windows | AUROC |
|---|---|---|---|
| PyTorch checkpoint | test (full) | 55,715 | **0.9939** |
| PyTorch checkpoint | val (full) | 36,825 | 0.9924 |
| int8 ONNX (`MatMul` + `Gather`) | test (3,500 sampled) | 3,500 | **0.7849** |

The bundle has been withdrawn. `export.gates.release_gate` now runs before
`export.bundle.assemble`, which raises without a passing gate result.

## What passed on the broken artifact

Every structural check, none of which scores anything:

| check | result |
|---|---|
| fp32 vs PyTorch parity, 5 sequence lengths x 2 batch sizes | max abs diff 5.48e-06, tolerance 1e-03 |
| `MatMulInteger` node count | 136 (floor 80) |
| `DynamicQuantizeLinear` nodes | 181 |
| file size | 150.7 MB from 599.0 MB, 3.98x |
| opset / domains | `ai.onnx` 17, no foreign domains |
| graph inputs | `input_ids`, `attention_mask`, both dynamic |
| `tokenizer_config.model_max_length` | 512 |

The fp32 parity result is the one worth dwelling on. It is genuinely good, and it
proves the export is faithful -- the ModernBERT sliding-window mask did not constant-fold,
which was the failure this project spent the most design effort guarding against. It
says nothing whatever about the quantized graph, because it was measured on the fp32
graph, before quantization ran.

## Diagnosis

Two variables moved between the 0.9939 and the 0.7849: the artifact (PyTorch to int8)
and the split (val to calibration/test). Scoring both full splits with PyTorch
separated them -- test at 0.9939 is if anything higher than val, so the split is not
the explanation and the model is intact.

That leaves quantization. The op list is `["MatMul", "Gather"]`, and the `Gather` is
the 50,368 x 768 embedding table. scope.md 5.4 adds it to reach the ~150 MB the spec
quotes: `MatMul` alone leaves the embeddings in fp32 and lands at ~266 MB. Per-tensor
int8 across an embedding table whose rows have very different norms is the most
destructive step available in this pipeline, and it was chosen to hit a file size.

`onnx_export.py` said so in its own module docstring -- "the riskiest step in this
pipeline for a detector whose signal is lexical" -- and then included it anyway,
"gated on a measured accuracy delta" that was never measured.

## Calibration on the broken artifact

Recorded for completeness. Every number below is a property of a model that does not
work, and none of it should be carried forward.

| quantity | value |
|---|---|
| temperature | 2.0037 |
| t_on | 0.4286 |
| t_off | 0.2916 |
| binding genre | technical_docs |
| recall at global threshold | 0.029 |
| recall at per-genre oracle | 0.232 |
| oracle gap | 0.203 |

A recall of 2.9% at the shipped threshold is the model failing, not the threshold
being conservative. The oracle gap of 0.203 was the number this project most wanted --
the empirical cost of scope.md 4.4's decision to ship one global threshold -- and it
cannot be read off a broken artifact. It remains unmeasured.

`technical_docs` bound the threshold off 57 human windows. Two causes compounded:
the subsample was uniform rather than genre-stratified because a `modal deploy` had
failed silently (`modal` on PATH resolved to an unrelated virtualenv that cannot
import `slopmarker`), so the stratified code never shipped; and the genre is only
1.9% of the corpus because `genre_targets` is enforced nowhere. In the full test
split technical_docs has 974 human windows, not 57.

## Gates added

`export.gates.release_gate` refuses a bundle unless:

| condition | threshold |
|---|---|
| Spearman, int8 vs fp32 on identical rows | >= 0.995 |
| AUROC drop | <= 0.01 |
| pAUC@2%FPR drop | <= 0.01 |
| per-genre FPR upper bound (Clopper-Pearson) on test | <= 0.02 |
| human windows per genre backing that bound | >= 500 |
| parity / quantization / graph sections | present and passing |

A missing comparison is a failure, not a pass. That is the specific way the original
path was wrong: absence of evidence read as evidence of correctness.
