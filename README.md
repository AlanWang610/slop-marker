# slop-marker

A self-trained ModernBERT-base classifier that flags likely-AI-generated prose, plus a
Chrome/Firefox extension that runs it on-device (ONNX Runtime Web) and de-emphasizes the
text it flags. One repo: the model pipeline and the extension that consumes its output.

Full spec: [docs/scope.md](docs/scope.md). Section numbers below refer to it.

## Layout

```
docs/            scope.md (the spec), decisions/ (ADRs), measurements/ (numbers we
                 actually measured — latency, per-genre FPR — not estimates)

training/        Python. Everything that produces a model bundle.
  configs/       corpus / train / export / eval configs (yaml, checked in)
  src/slopmarker/
    corpus/      human harvest, AI generation, attack variants, mixed authorship (§4.2)
    data/        datasets, windowing (§4.3), genre labels, collators
    model/       backbone + AI-fraction head + auxiliary genre head (§4.4)
    export/      ONNX export, int8 quantization, PyTorch-parity check (§5)
    eval/        per-genre FPR, calibration/threshold selection (§4.5), RAID (§4.6)
  scripts/       thin CLI entry points
  tests/

extension/       TypeScript. One source tree, two builds (§13).
  src/
    content/     extraction, language gate, segmentation, hashing, rendering (§7, §9)
    router/      chrome.ts (MV3 service worker) / firefox.ts (event page) — the only
                 browser-specific runtime code (§6.2)
    host/        host.html + host.js: score cache, priority queue, worker mgmt (§6.3)
    worker/      ORT Web session, single dedicated Web Worker
    shared/      scoring & aggregation (§8), text normalization + hashing (§7.4),
                 port protocol types, calibration loader
    pages/       options page, first-run page (§6.5)
    assets/      manifest source, bundled tokenizer, icons
  scripts/       build: per-browser manifest post-processing, ORT wasm copy, packaging
  tests/         unit
  e2e/           browser-driven parity checks

fixtures/        Small JSON fixtures consumed by BOTH training/tests and extension/tests.
                 This is how we prove the two sides agree (see Coupling below).

data/            Corpora. Gitignored.
artifacts/       Checkpoints, runs, exports, assembled bundles. Gitignored.
model-host/      Staging for what gets uploaded to the static model host. Gitignored.
tools/           Cross-cutting scripts: build both browsers, bump model version, publish bundle.
```

## What is and isn't tracked

The model never enters git. `data/`, `artifacts/` and `model-host/` are ignored except
their READMEs, and `*.onnx` / `*.safetensors` / checkpoints are ignored everywhere.
The shipped model bundle (§5) lives on a static host and is fetched at first run (§6.4).

Two small files from each bundle release *are* tracked, inside `extension/src/assets/`:
the tokenizer (§6.3 bundles it in the extension so browser tokenization can't drift) and
`calibration.json` + `SHA256SUMS` (the extension verifies downloads against them). They
are copied in from a bundle release by a `tools/` script, not hand-edited.

## Coupling between the two halves

Three things must stay in lockstep; everything else is independent.

1. **Calibration constants.** `temperature`, `t_on`, `t_off`, `min_words` are produced by
   `training/` on the *quantized* artifact (§5.5) and consumed by the extension. Single
   source: `calibration.json`. No constant is retyped in TypeScript.
2. **Text handling parity.** `normalize()` + hashing (§7.4) and window/chunk boundaries
   (§4.3, §7.3) exist in Python and TypeScript. Fixtures in `fixtures/` pin the expected
   output of both; both test suites read the same files.
3. **Model version string.** Lives in `calibration.json`, keys the score cache (§11), and
   invalidates old scores on both browsers when the model is refreshed (§4.7).

## Toolchain

Python is pinned to 3.12 via `.python-version` (uv), not the 3.14 on this machine — the
training stack's wheels lag. Node 24 / npm for the extension.
