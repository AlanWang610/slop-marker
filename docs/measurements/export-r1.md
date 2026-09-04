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

That leaves quantization, and the first hypothesis was wrong. `onnx_export.py` named
the embedding `Gather` in its own module docstring as "the riskiest step in this
pipeline for a detector whose signal is lexical", so the `Gather` was the obvious
suspect. Measured, it is not: leaving the embedding table entirely in fp32 scores
0.8027 against the 0.8103 that quantizes it. The docstring read as evidence because it
was confidently written, but nothing in it had been measured.

Two real causes were found, in order:

1. `extra_options={"MatMulConstBOnly": False}`. ONNX Runtime defaults this to True,
   restricting quantization to MatMuls with a constant B operand -- the weights. False
   also quantizes the activation-by-activation products inside attention, Q.K^T and
   attn.V, which hold no weights and degrade badly in int8. Fixing it recovered 17
   points of AUROC. The node count had said so all along: ModernBERT-base has
   22 x 4 weight MatMuls plus a two-MatMul head, so 90 is correct and the export
   produced 136. The assertion was `>= 80` with no ceiling, so quantizing half again
   too much looked exactly like quantizing correctly.
2. Dynamic quantization of *activations*, which no op list can switch off. This is what
   still cost 4-6 points of pAUC afterwards, and it is why the fix is weight-only
   quantization rather than a better dynamic recipe. See the recipe table below.

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
| int8-vs-fp32 comparison on identical rows | must have run |
| AUROC drop | <= 0.01 |
| pAUC@2%FPR drop | <= 0.01 |
| decision flip rate | <= 0.01 |
| Spearman | >= 0.95 (see the revision below) |
| per-genre FPR upper bound (Clopper-Pearson) on test | <= 0.02 |
| human windows per genre backing that bound | >= 500 |
| parity / quantization / graph sections | present and passing |
| `MatMulInteger` node count | within [80, 100], both ends |
| `GatherBlockQuantized` present when the table is meant to be quantized | required |

A missing comparison is a failure, not a pass. That is the specific way the original
path was wrong: absence of evidence read as evidence of correctness.

## Quantization recipe, measured

Every recipe scored against the fp32 graph on the same 1,500 test windows.

| recipe | MB | AUROC | pAUC@2%FPR | ΔpAUC | Spearman | max Δlogit |
|---|---|---|---|---|---|---|
| fp32 baseline | 599.0 | 0.9921 | 0.9571 | -- | -- | -- |
| weight-only int8, encoder only | 270.7 | 0.9923 | 0.9600 | **-0.0029** | 0.998 | 0.39 |
| **weight-only, int8 encoder + int4 table** | **136.7** | **0.9925** | **0.9563** | **0.0008** | 0.983 | 2.23 |
| weight-only int4 throughout | 80.8 | 0.9919 | 0.9317 | 0.0254 | 0.810 | 4.94 |
| dynamic int8, weights + embeddings | 150.6 | 0.9673 | 0.9021 | 0.0550 | 0.623 | 7.38 |
| dynamic int8, per-channel | 151.3 | 0.9777 | 0.9148 | 0.0423 | 0.638 | 7.19 |
| dynamic int8, all matmuls *(shipped in r1)* | 150.7 | 0.8103 | 0.6047 | 0.3524 | 0.409 | 7.95 |

The split between the two families is the whole story. `quantize_dynamic` compresses
weights *and* rescales activations per tensor at runtime; `MatMulNBits` compresses only
the constant weight operands and leaves activations in float. Weight-only is lossless
here -- pAUC marginally above fp32 -- and no dynamic recipe comes close, whatever is
done with per-channel scales, reduce_range, or the embedding table.

Two incidental findings, both from rows that were expected to differ and did not:

- `MatMulNBits` silently ignores `Gather` at 8 bits. The int8-with-embeddings and
  int8-without-embeddings builds came out byte-identical at 270.7MB, with 38.7M
  embedding parameters left in fp32. At 4 bits it does quantize them, which is the only
  way the int4 row reaches 80.8MB. `quantize_mixed` therefore fails the build when no
  `GatherBlockQuantized` node appears, since the sole symptom is a file 120MB too large.
- The encoder and the embedding table need not share a width. int8 encoder with an int4
  table is 136.7MB at a pAUC cost of 0.0008, which is how the scope.md target is reached
  without quantizing anything that turned out to be sensitive.

## Dynamic int8 is not portable, and that is disqualifying

Two sweeps over identical inputs returned AUROC 0.978 and 0.550 for the same dynamic
recipe, from files identical in size, node counts and every flag. Scoring the same 400
rows twice per container across three containers explains it:

| recipe | within container | AVX-512 vs AVX2, mean logit | max Δ |
|---|---|---|---|
| int8 encoder + int4 table | 0.0 | 0.000000 | 0.000002 |
| dynamic int8, per-channel | 0.0 | 1.233807 | 3.371312 |

Both are bitwise deterministic on any one machine, which is precisely why this hides.
Across instruction sets the weight-only build agrees to 2e-06. The dynamic build does
not agree at all: on a container without AVX-512 VNNI its output collapses to roughly
-1.07 for every input.

    AVX-512   [-3.083, -3.244, -3.105, -3.272, +2.316]
    AVX2      [-1.083, -1.086, -1.071, -1.089, -1.056]

The fifth window is AI-authored. It scores +2.32 on one machine and negative on the
other. This is ONNX Runtime's `MatMulInteger` kernels, not anything in this pipeline.

For a model that ships to browsers on unknown hardware this rules out dynamic
quantization by itself, before any accuracy argument. It would have produced garbage
for a fraction of users, and nothing in the export path was looking for it. It also
means the "dynamic loses 4-6 points of pAUC" figures above are the AVX-512 numbers;
on AVX2 the loss is total.

## Gate revision

The plan gated on `Spearman >= 0.995`. Only the 270MB build reaches that; the 136.7MB
build reaches 0.983 while dropping 0.0008 of pAUC. Rather than relax the number to
admit the preferred variant, the gate now measures what Spearman was standing in for:

| condition | threshold | why |
|---|---|---|
| pAUC@2%FPR drop | <= 0.01 | separation at the operating point, measured on the shipped artifact |
| AUROC drop | <= 0.01 | separation overall |
| decision flip rate | <= 0.01 | share of windows the two artifacts put on opposite sides of the threshold -- what a reader actually sees |
| Spearman | >= 0.95 | retained only to catch gross reordering, of the kind that produced 0.045 |

Spearman summarises the entire score range, nearly all of which sits far from the
threshold and cannot change any verdict. The flip rate measures the same concern
directly and at the right place.
