# Measured model sizes and the quantization arithmetic

Measured by constructing the model, not estimated.

| quantity | value |
|---|---|
| total training parameters | 149,612,554 |
| shipped classifier (`core`) | 149,605,633 |
| auxiliary genre head (dropped at export) | 6,921 |
| fp32 size of the shipped classifier | **598 MB** |
| embedding table (50368 x 768) | 38,682,624 params = **155 MB** at fp32 |

## Why the op list matters

scope.md's "~600 MB fp32" is correct — 598 MB measured. The int8 figure was not.

ModernBERT's embedding table appears in the ONNX graph as a `Gather`, not a `MatMul`.
Quantizing `MatMul` alone therefore leaves 155 MB of embeddings at fp32:

| variant | approximate size |
|---|---|
| fp32 | 598 MB |
| int8, `MatMul` only (as scope.md originally specified) | ~266 MB |
| int8, `MatMul` + `Gather` | ~150 MB |

The ~150 MB the spec quotes is only reachable with the embedding `Gather` included, so
that is what `export/onnx_export.py` requests. It is the riskiest step in the pipeline
for a detector whose signal is lexical, so it is gated on a measured accuracy delta
rather than assumed safe.

## Strip equality

The auxiliary head is attached beside the stock classifier rather than inside it, so
exporting drops it without touching the AI-fraction path. Measured on random inputs in
eval mode:

    max |logits_before_strip - logits_after_strip| = 0.00e+00

Exactly zero, not merely within tolerance. That equality is the whole guarantee that the
exported model scores identically to the trained one.

## Layerwise learning-rate decay

48 parameter groups, learning rates spanning 2.66e-06 (embeddings) to 3.00e-05 (heads)
at base 3e-5 with decay 0.9 over 22 layers.
