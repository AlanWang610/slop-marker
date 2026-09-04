# Export, parity and quantization -- measured

Run against a real ModernBERT-base checkpoint saved through the same strip-and-save
path training uses. Numbers below are measured, not projected.

## fp32 export

| quantity | value |
|---|---|
| export time | 7 s |
| fp32 `model.onnx` | **599 MB** |
| opset | 17 |
| foreign operator domains | none |
| dynamic axes on both inputs | present |
| tokenizer `model_max_length` | 512 |

Optimum emits a warning that opset 18 is its recommended minimum for ModernBERT. The
export succeeds at 17 and parity below is 5e-06, so 17 is kept -- it is what scope.md
specifies and it is the more conservative choice for ONNX Runtime Web's WASM kernels.

## Parity across shapes -- the frozen-mask risk did not materialise

ModernBERT builds its sliding-window attention mask from `torch.arange(seq_len)`, which
is a constant-folding candidate under the tracer. Had it folded, the mask would be
frozen at the export length and logits would be silently wrong at every *other* length
-- a failure a 512-only check cannot see.

Measured max |ONNX - PyTorch| per shape, tolerance 1e-3:

| shape | max abs delta |
|---|---|
| seq64 batch1 | 5.33e-06 |
| seq64 batch4 | 5.45e-06 |
| seq128 batch1 | 5.33e-06 |
| seq128 batch4 | 5.04e-06 |
| seq256 batch1 | 5.33e-06 |
| seq256 batch4 | 5.04e-06 |
| seq384 batch1 | 4.14e-06 |
| seq384 batch4 | 5.04e-06 |
| seq512 batch1 | 4.14e-06 |
| seq512 batch4 | 5.04e-06 |

Flat across every length, ~200x inside tolerance. The dynamic sequence axis is real, so
a 215-token chunk costs what a 215-token chunk should rather than being padded to 512.

## int8 quantization

| quantity | value |
|---|---|
| op types quantized | `MatMul`, `Gather` |
| int8 `model.onnx` | **150.7 MB** |
| compression | 3.98x |
| `MatMulInteger` nodes | 136 |
| `DynamicQuantizeLinear` nodes | 181 |
| quantized `Gather` nodes | 95 |

150.7 MB confirms the correction in `export/onnx_export.py`: scope.md's ~150 MB target
is reachable only with the embedding `Gather` included. Quantizing `MatMul` alone would
have left the 155 MB embedding table at fp32 and landed near 266 MB.

The node counts matter as an assertion rather than a statistic. There is a known ONNX
Runtime regression in which transformer MatMuls silently stop being quantized and the
output file comes back the same size as the input, raising nothing. 136 `MatMulInteger`
nodes against a floor of 80 is what proves it actually happened.

Accuracy impact of quantizing the embedding table is not measured here -- that is gated
on the calibration split against a trained checkpoint, since a randomly initialised head
would say nothing useful.
