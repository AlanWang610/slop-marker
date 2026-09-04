# ONNX Runtime Web latency -- bundle `mb-base-0.2.0-dev`

> These figures were measured on `mb-base-0.2.0-dev` and carry over unchanged to
> `mb-base-0.3.0-dev`, which ships a byte-identical `model.onnx` (sha256 `42e850ac...`
> in both). The later bundle differs only in `calibration.json`, which records the
> document-level gate. Latency was not re-measured, because the graph did not change.

The scope.md 10 budget was an estimate. These are measurements, and they change one
conclusion: **the shipped artifact runs about 2x slower in the browser than the estimate
assumed, for a reason that is structural rather than incidental.**

Measured with `extension/scripts/bench-ort.mjs`, `onnxruntime-web` 1.29.0, WASM execution
provider with SIMD, on an Intel Core Ultra 9 285H (16 cores, 64 GB), Node 24. Median of 3
runs after a warm-up, synthetic token ids at fixed sequence lengths -- cost depends on
sequence length, not token identity.

## The finding

ONNX Runtime's MLAS assigns `QNBitGemmDispatch` only under `MLAS_TARGET_AMD64`,
`MLAS_TARGET_ARM64`, `MLAS_TARGET_RISCV64` and `MLAS_TARGET_LARCH64`
(`onnxruntime/core/mlas/lib/platform.cpp`). **There is no WASM branch**, although
`MLAS_TARGET_WASM_SIMD` is a real target used elsewhere in `mlasi.h`.

So in a WASM build `MlasIsQNBitGemmAvailable()` returns false
(`mlas/lib/qnbitgemm.cpp:122`), `PrePack` returns early without packing, and every one of
the artifact's 89 `MatMulNBits` nodes falls through to
`MatMulNBits<float>::ComputeBUnpacked` (`contrib_ops/cpu/quantization/matmul_nbits.cc:1192`).
That path allocates `K x N` floats, dequantizes the entire weight matrix, transposes it
into a second `K x N` buffer, and runs a plain SGEMM -- per node, per forward pass.

The result is correct. It is not free.

## Measured

Milliseconds per chunk, single `run()`, batch 1.

| artifact | ops | threads | 128 | 215 | 256 | 384 | 512 |
|---|---|---|---|---|---|---|---|
| `mix_e8_emb4` **(ships)** | `MatMulNBits` | 1 | 685 | 1122 | 1340 | 2044 | 2806 |
| `mix_e8_emb4` **(ships)** | `MatMulNBits` | 4 | 191 | 308 | 360 | 553 | 763 |
| `dyn_weights_gather` | `MatMulInteger` | 1 | 289 | 509 | 628 | 1020 | 1491 |
| `dyn_weights_gather` | `MatMulInteger` | 4 | 87 | 147 | 179 | 284 | 420 |

215 tokens is the median chunk in `document-eval-r1.md`, so that is the column that
describes real pages.

**The missing kernel costs 2.2x single-threaded and 2.0x threaded.** `MatMulInteger` is the
one quantized path MLAS does have WASM kernels for, and it roughly meets the scope.md 10
budget; the weight-only artifact is roughly 2x over it. Threading recovers 3.6x, so a
cross-origin-isolated context matters more than the estimate implied.

Against scope.md 10 directly:

| | budget | `mix_e8_emb4` | verdict |
|---|---|---|---|
| single thread, 512 | 600-1200 ms | 2806 ms | 2.3-4.7x over |
| single thread, 256 | 300-600 ms | 1340 ms | 2.2-4.5x over |
| 4 threads, 512 | 200-400 ms | 763 ms | 1.9-3.8x over |
| 4 threads, 256 | 100-200 ms | 360 ms | 1.8-3.6x over |

Session creation is 0.6 s from bytes already in memory, comfortably inside the "low
single-digit seconds" that scope.md 6.2's Firefox warm-restore path assumes.

## Memory is not the recipe's fault

Resident set is 553-657 MB across *every* configuration measured, `MatMulInteger` included.
It tracks the ORT WASM heap for a ~150 MB model, not the dequantize fallback, so it is not
an argument for changing recipe. It is roughly 2x the ~250 MB scope.md 10 projects, and it
is resident in the Chrome offscreen document or the Firefox event page for as long as the
session lives.

## Why we ship the slower artifact anyway

Because the faster op family cannot be made accurate. From `quantization_sweep.json`, pAUC
drop against the 0.01 release gate in `export/gates.py`:

| recipe | MB | pAUC drop | gate |
|---|---|---|---|
| `mix_e8_emb4` (`MatMulNBits`) | 136.7 | 0.0008 | pass |
| `wo_int8` (`MatMulNBits`) | 270.7 | -0.0029 | pass |
| `dyn_weights_gather_reduce_range` | 151.3 | 0.0440 | **fail** |
| `dyn_weights_gather` | 150.6 | 0.3717 | **fail** |
| `dyn_all_matmul_gather` | 150.7 | 0.4444 | **fail** |
| `dyn_weights_gather_per_channel` | 151.3 | 0.4490 | **fail** |

No dynamic recipe comes within 4x of the accuracy gate, and `export-r1.md` separately
disqualifies the whole family for non-portability across instruction sets. Trading four
points of pAUC for 2x latency is not a trade this project makes -- scope.md 1 puts false
flags on human text first. **The 2x is accepted and mitigated in the extension instead.**

Mitigations, in scope.md 10's own order of effect: viewport-first queue so on-screen text
scores first, the content-keyed score cache so syndicated and revisited text is scored
once, and capping `max_length`. Note the third knob is weaker than it looks here: the
median chunk is 215 tokens, so a 256 cap only trims the tail.

## Observed in real browsers

The Node numbers above are per-`run()`. What each browser actually gets is decided by
thread availability, and the two differ exactly as scope.md 12 predicted -- now measured
rather than assumed, by `extension/e2e/run.mjs` and `extension/e2e/firefox.mjs`:

| | cross-origin isolated | threads | 215-token chunk |
|---|---|---|---|
| Chrome (offscreen document) | **yes**, via the manifest COOP/COEP keys | 4 | ~308 ms |
| Firefox (event page) | **no** | 1 | ~1122 ms |

Firefox is roughly **3.6x slower per chunk**, and the cause is structural rather than
incidental: `cross_origin_embedder_policy` and `cross_origin_opener_policy` are Chrome-only
manifest keys, and the Firefox build strips them because the AMO validator rejects unknown
keys rather than ignoring them. Without cross-origin isolation `SharedArrayBuffer` is
unavailable and ORT falls back to a single thread.

That is a real difference in what a user experiences, and it is the one place where
scope.md 1's "identical detection behaviour on both browsers" needs reading carefully: the
*scores* are identical, and the fixtures prove it. The *latency* is not, and cannot be
without a Firefox mechanism for isolating an extension page.

The observed mode is written to `storage.local` and shown on the options page, which is
what scope.md 6.3 asks for.

## What this does not cover

- **Logit parity.** Latency only. That the WASM graph reproduces the Python scores is
  measured separately, in `extension/tests/model-parity.test.ts` against
  `fixtures/logits.json`: max |delta| 2.72e-5 against a 1e-3 tolerance.
- **A layer-dropped backbone**, which scope.md 10 names as the fallback if the budget
  cannot be met. Not attempted; it is a model change, not an extension change.
