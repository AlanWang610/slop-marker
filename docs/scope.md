# AI-Generated Text De-emphasis Extension — Scope (Chrome + Firefox)

Browser extension for Chrome and Firefox, built from one codebase, that visually de-emphasizes page text likely to be AI-generated, using a self-trained ModernBERT-base classifier run in-extension via ONNX Runtime Web. Single release per browser; the only artifact updated after release is the model bundle.

---

## 1. Goals and non-goals

**Goals**

- De-emphasize (not hide) contiguous runs of likely-AI prose in the main content of English web pages.
- All inference on-device. No text leaves the browser.
- High-precision operating point: false flags on human text are the primary failure to minimize, at the cost of recall.
- Degrade correctly on dark and colored backgrounds.
- Identical detection behaviour on both browsers: same model artifact, same calibration constants, same scoring logic.

**Non-goals**

- Non-English text (backbone is English-only; see §4).
- Sentence-level or sub-paragraph attribution.
- Detecting light AI-assisted editing of human drafts. Heavily AI-rewritten text is in scope.
- Feedback/report UI, telemetry, or post-release extension updates. Fixed feature set.
- Code, forms, navigation, comment widgets, editable regions.
- Safari, mobile browsers.

---

## 2. Known limits (design is built around these)

- Detectors are near chance on short spans (< ~100–150 words). Minimum span length is enforced before scoring.
- False positives concentrate on formal, edited, templated, and non-native-English prose. These are over-sampled as hard negatives in training, and the shipped threshold is the one that satisfies the *worst* genre (§4.5). It is a single global threshold, not a per-genre one: the genre head is dropped at export, so the extension has no genre signal to select on.
- Paraphrase and "humanizer" attacks degrade any classifier. Attack variants are in the training data; robustness is not assumed.
- Most flagged text on the real web will be press releases, SEO listicles, product copy, corporate blogs. Some is human-written. Accepted.

---

## 3. Architecture

```
┌──────────────────────────── page ────────────────────────────┐
│ content script (shared)                                      │
│  extract blocks → language gate → segment → hash →           │
│  viewport-priority request → apply styling                   │
└──────────────────────────────┬───────────────────────────────┘
                               │ runtime.Port
┌──────────────────────────────▼───────────────────────────────┐
│ router                                                       │
│   Chrome:  MV3 service worker (stateless)                    │
│   Firefox: MV3 event page                                    │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ inference host  (shared module `host.js`)                    │
│   Chrome:  offscreen document (persists across SW restarts)  │
│   Firefox: instantiated inside the event page itself         │
│                                                              │
│   score cache (storage.local) → priority queue →             │
│   Worker: ORT Web session (int8 ModernBERT-base) →           │
│   score → cache write → reply                                │
└──────────────────────────────────────────────────────────────┘
        model bundle: downloaded once from own host,
        cached via Cache API, keyed by model version
```

Shared code units: content script, `host.js` (cache, queue, worker management), inference worker, options page, first-run page. Browser-specific: ~50 lines of router glue and one manifest with both background keys.

Options page: per-site allowlist, threshold override, "clear cache / re-download model", display of observed threading mode. Nothing else.

---

## 4. Detector model

### 4.1 Backbone

ModernBERT-base (149M params, 22 layers, alternating local/global attention, GeGLU). English + code pretraining only; non-English text is gated out rather than scored.

Inference sequence length fixed at **512 tokens** (tokenizer `max_length` at export). The 8192 native context is unused; attention cost is the latency budget in WASM.

### 4.2 Training data

Two classes, stratified by web genre:
news, product/marketing copy, press releases, forum/comment, blog/personal, technical documentation, encyclopedia, academic/formal.

**Human side**

- Pre-2022 crawl slices (CommonCrawl, Wikipedia, Reddit, news archives) stratified by genre.
- Hard negatives, over-sampled: press releases, SEO/marketing copy, corporate blogs, templated product descriptions, non-native English forum posts, formal academic prose.

**AI side**

- 12 training generators × temperature/sampling variants × prompt styles: continuation, rewrite-this, "write a blog post about X", instruction with persona, structured/listicle output. Frontier models via the Anthropic, OpenAI and Gemini APIs; open-weight models via vLLM on Modal GPUs.
- **Held-out generators.** A further 4 models appear only in the test split. Generator count is not the knob that determines generalization — the number of generators the model has *never seen* is, and that is what the held-out set measures.
- Attack variants: paraphrase via a second model, humanizer-style rewrites, sentence shuffling, light typo injection. Typo injection is also applied to a matched slice of *human* documents; injecting typos only into AI text would teach "typo ⇒ AI", firing on exactly the non-native prose §2 names as the primary FPR risk.
- Mixed authorship at several edit fractions. Note that "human-draft-then-AI-edit" and "AI-draft-then-human-edit" are the same artifact at the token level — which side was the draft is a fact about process, not about the text. One splicer produces both; the direction is recorded as provenance, not as a second label distribution.
- Every AI document is seeded from one human document and inherits its genre, topic, entities, target length and structural shape, so the two classes share a topic and length distribution by construction rather than by post-hoc balancing.

**Labels**

Regression target = AI fraction ∈ [0, 1]. Binary AI/human label derived at ≥ 0.7 for evaluation. Regression head gives the UI a continuous dial without retraining.

The fraction is **measured, never requested**: a model asked to "edit 30% of this" will not comply, so mixed documents are built by splicing aligned human and AI variants and the label is computed from what was actually emitted.

Documents that are neither pure human nor pure AI carry **character spans** marking the AI-authored regions. Windows recompute their own fraction from those spans (§4.3). The corpus also contains documents at fractions near 0.05, without which §1's non-goal — not flagging lightly AI-edited human prose — is unenforceable rather than merely unmet.

### 4.3 Windowing

Train on the unit scored at inference: random windows cut at sentence boundaries, 40–400 words, weighted toward 80–250 words. Short windows are kept so the model learns low confidence on them rather than confident noise.

Two distinct functions, easily confused:

- **`sample_windows`** — stochastic, training only. Per-document seeded, so windowing is independent of shard count and processing order and the corpus stays append-only.
- **`chunk_block`** — deterministic, what the extension runs (§7.3), pinned by `fixtures/windows.json`.

Both cut on the same sentence splitter; that is what keeps the training distribution aligned with what is actually scored.

**A window's AI fraction is computed from the document's AI spans, not inherited from the document.** A window cut from the human half of a half-AI document is entirely human; giving it the document's 0.5 injects label noise proportional to how localized the AI text is.

At ~1.25–1.35 BPE tokens per word, a 400-word window is ~500–540 tokens, so the 512-token limit truncates the longest windows. That is the common case at the top of the range, not an edge case; `n_tokens` and `truncated` are recorded per window.

### 4.4 Head and loss

- `[CLS]` pooling (mean pooling acceptable) → dropout 0.1 → linear.
- Primary loss: BCE against the soft AI-fraction target, minimized at `σ(z) = y`. "Label smoothing 0.05" means a **target-range clamp** to `[0.025, 0.975]`, which caps `|z|` and stops logit blow-up on the bimodal mass at 0 and 1. It is not smoothing of a derived binary label. Note the tension: the operating point lives in the far right tail of the human score distribution, and symmetric smoothing compresses exactly that region, so an asymmetric variant clamping only the AI end is available and settled by measurement.
- Soft-target BCE has a **nonzero floor** equal to the mean target entropy, so the loss plateaus well above 0 and never approaches it. Excess BCE (`L − H(target)`) is logged alongside it, or training looks broken when it is fine.
- Auxiliary head: genre classification, cross-entropy, weight 0.2, dropped at export.
- **What actually removes the "formal register ⇒ AI" shortcut is per-genre class balance**, not the auxiliary head. A multi-task CE head encourages the backbone to encode genre linearly; it is a reasonable generic regularizer but it is not a shortcut-removal mechanism. The mechanism is `P(AI | genre) ≈ 0.5` for every genre, enforced by the sampler and asserted at dataset load.
- The length penalty (§8) is an **inference-time** correction. Short windows are not down-weighted in the loss: §4.3 keeps them precisely so the model learns low confidence on them, and down-weighting would remove that signal and make short-window predictions more confidently wrong.
- Full fine-tune. lr 3e-5, layerwise lr decay 0.9, 2–3 epochs, warmup 6%.
- Validation metric: FPR at fixed recall, reported **per genre**. Model selection is on **pAUC at 2% FPR**, not AUROC — two checkpoints with equal AUROC can differ by many recall points in the only region we operate in.

### 4.5 Calibration and threshold

Four splits, not three: **train / val / calibration / test**. Calibration selects the threshold; test verifies it. They are disjoint at the group level, and the grouping key is the near-duplicate cluster first and the registered domain second — press releases are syndicated verbatim across dozens of domains, so domain-level grouping alone leaks.

- Temperature-scale on the calibration set.
- Operating threshold: `t_on = max(threshold at 1% FPR overall, max over genres of the threshold at 2% FPR)`. Recall is whatever that yields.
- Temperature and threshold shipped as constants with the model file; selected on the **quantized** artifact (§5).
- Hysteresis pair for the UI (§8): `t_on`, and `t_off` derived from it by a fixed offset **in log-odds space**. The specified `t_off = t_on − 0.1` in probability space is not scale-free: at `t_on = 0.97` that band spans 3.48 → 1.90 in log-odds, and at `t_on = 0.60` it spans 0.41 → 0.00.

Four consequences that the bare thresholds above do not convey:

- **The per-genre bound is the one that binds; the 1% overall bound is nearly decorative.** If every genre is under 2%, the mixture is too, and in practice one hard genre sits near the cap while the rest sit far below it. An overall FPR is also meaningless without a stated genre mixture, so it is reported under both a uniform and a web-like mixture and only `max_genre FPR ≤ 2%` gates the release.
- **A single global threshold can always satisfy the per-genre bound** — per-genre FPR is monotone in the threshold, so the feasible set is never empty. There is no infeasibility here, only cost: the worst genre sets the operating point for every genre. §4.4 drops the genre head at export, so the extension has no genre signal and cannot do better. The eval therefore also computes the per-genre-*oracle* recall, and the gap between the two is the measured price of that decision rather than an opinion about it.
- **Select with margin, verify separately.** A threshold placed at the 98th percentile of a few thousand calibration windows rests on ~140 tail points; selecting *at* the target means missing it about half the time on fresh data. Selection uses a margin below the target, and test verifies with a Clopper–Pearson 95% upper bound.
- **The chunk-level bound is not the user-visible one.** What a reader sees is a highlighted run, produced after the length penalty, run pooling, the document prior and the 150-word minimum (§8). Release therefore also gates on **document-level FPR** with the full §8 pipeline applied to held-out human documents.

### 4.6 External evaluation

RAID (out-of-domain + adversarial splits) for a comparable external number, noting its generator set skews to 2023-era models.

RAID's rows are whole documents, so the eval runs the full pipeline — chunk → score → §8 aggregation → document score. That is the honest comparison and it is also the only exercise of the aggregation code against text we did not generate.

**RAID's more serious limitation is its domains, not its generator vintage.** Its eight domains contain no press releases, no marketing copy, no product descriptions and no corporate blogs — that is, none of the genres §2 identifies as the dominant false-positive risk. RAID will therefore read better than production, and it is a comparability number rather than the number that governs shipping. A small hand-curated in-the-wild set drawn from held-out hard-negative hosts serves that purpose instead.

RAID's human side overlaps common corpora (Reddit via Webis TL;DR-17, arXiv abstracts, Wikipedia, BBC news), so it is a contamination hazard as well as an eval. Those sources are excluded from the harvest outright, and the remaining corpus is filtered against RAID by near-duplicate and n-gram overlap. The removal count is reported, since that is what makes the external number credible.

### 4.7 Model refresh

Regenerate the AI side with current models when warranted; retrain, recalibrate, re-export, bump the model version string. Neither extension changes. Cached scores are keyed on model version, so a new model invalidates old scores automatically on both browsers.

---

## 5. Export and quantization

1. Load checkpoint with `attn_implementation="eager"` (or `sdpa`). The unpadded flash-attention path does not trace.
2. Export via Optimum: opset ≥ 17, dynamic batch and sequence axes, tokenizer `max_length=512`.
3. Verify ONNX logits vs PyTorch on ~200 samples, tolerance ~1e-3, **at several sequence lengths and batch sizes**, not only at 512.

   ModernBERT builds its sliding-window attention mask from `torch.arange(seq_len)`, which is a strong constant-folding candidate under the tracer. If it folds, the local mask is frozen at the export length and the model returns silently wrong logits at every *other* length — a failure a 512-only parity check cannot see. Fallbacks, in order: export with the dynamo path; patch the mask builder inside the export context; accept a static 512 axis and pad in the extension, which costs roughly 2.3× latency since typical chunks are ~215 tokens.
4. Int8 dynamic quantization (`quantize_dynamic`, **MatMul and the embedding `Gather`**). ~600 MB fp32 → ~150 MB int8.

   The fp32 figure is right: 149,014,272 params × 4 B = 596 MB. The int8 figure is only reachable if the embedding table is quantized too. ModernBERT's embeddings are 50368 × 768 = 38.7 M params, and they appear in the graph as a `Gather`, not a `MatMul` — quantizing MatMul alone leaves them in fp32 and lands at **266 MB**, not 150 MB. Per-tensor int8 across a 50k-row embedding table is the riskiest step in this pipeline for a detector whose signal is lexical, so it is gated on a measured accuracy delta, with fp16 embeddings (~189 MB) as the fallback if it fails.

   `Gemm` is not in ONNX Runtime's integer-op registry, so naming it in dynamic mode does nothing. It is harmless here regardless: ModernBERT is bias-free throughout, so every linear exports as a bare `MatMul`.

   Assert the result rather than trusting it. There is a known ONNX Runtime regression in which transformer MatMuls silently stop being quantized and the output file comes back the same size as the input, with no error. The build fails if the `MatMulInteger` node count or the output size is outside the expected range.
5. Re-run calibration and per-genre FPR on the int8 model; set `t_on`/temperature from this artifact.
6. Confirm the exported `tokenizer.json` includes the `[CLS]`/`[SEP]` post-processor so browser tokenization matches training.

**One artifact for both browsers.** WASM int8 is the execution path on Chrome and Firefox alike. Chrome could additionally run a WebGPU execution provider, but that requires a separate fp16 artifact with its own calibration and would break behavioural parity; not adopted. fp16 on WASM is not useful (no native fp16 compute). 4-bit weight-only is not adopted unless verified accuracy-neutral.

Shipped model bundle, exactly six files: `model.onnx` (int8), `tokenizer.json`, `tokenizer_config.json`, `config.json`, `calibration.json`, `SHA256SUMS`. Provenance (run id, git sha, per-genre report, parity report, RAID numbers) is written *beside* the bundle, not inside it — the extension verifies every file in the directory against `SHA256SUMS`.

`calibration.json` carries `{version, temperature, t_on, t_off, min_words, max_length}` plus an `aggregate` block holding the §8 constants: the length-penalty pivot, the run-word minimum, the document-prior fraction and its log-odds bump. Those numbers are calibration outputs, and if they live in TypeScript instead then §4.7's promise that a model refresh needs no extension change is false.

---

## 6. Extension runtime

### 6.1 Manifest (single file, MV3, both browsers)

```json
{
  "manifest_version": 3,
  "background": {
    "service_worker": "router-chrome.js",
    "scripts": ["router-firefox.js"]
  },
  "permissions": ["storage", "unlimitedStorage", "offscreen"],
  "host_permissions": ["<all_urls>", "https://<model-host>/*"],
  "content_scripts": [{ "matches": ["<all_urls>"], "js": ["content.js"], "run_at": "document_idle" }],
  "content_security_policy": {
    "extension_pages": "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'"
  },
  "cross_origin_embedder_policy": { "value": "require-corp" },
  "cross_origin_opener_policy":   { "value": "same-origin" },
  "browser_specific_settings": { "gecko": { "id": "...", "strict_min_version": "140.0" } }
}
```

- Chrome uses `service_worker` and ignores `scripts`; Firefox uses `scripts` (event page) and ignores `service_worker`.
- `offscreen` permission is unknown to Firefox and ignored; COOP/COEP keys are Chrome-only and ignored by Firefox.
- `unlimitedStorage` is required on Chrome for the 150 MB Cache API entry and the score cache; harmless on Firefox.
- All code written against promise-returning `chrome.*` APIs, which Firefox MV3 supports; no polyfill.

### 6.2 Router

**Chrome (service worker).** Stateless. On any content-script connection: ensure the offscreen document exists (`chrome.runtime.getContexts` → `chrome.offscreen.createDocument({ url: "host.html", reasons: ["WORKERS"], justification })`), then forward messages between the content script port and the offscreen document. Holds no cache and no queue; the service worker being killed is a non-event. The offscreen document persists until the extension is reloaded or the browser exits; it is not subject to the 30 s idle rule (that applies only to `AUDIO_PLAYBACK`).

**Firefox (event page).** Router and host in one page. The event page unloads after ~30 s without events, taking the ORT session with it. Mitigations, in layers:
1. Content scripts keep a `runtime.Port` open and send a keepalive every 20 s while the tab has unscored on-screen blocks. Port traffic resets the idle timer.
2. Warm restore: when the page reloads, the host re-creates the session from the Cache API. Session creation cost is measured, not assumed; expected low single-digit seconds. Content scripts tolerate this by re-sending pending requests on port reconnect.

### 6.3 Inference host (`host.js`, shared)

- Owns: score cache read/write, priority queue, a single dedicated Web Worker holding the ORT Web session.
- Instantiated in `host.html` (Chrome offscreen document) or directly by `router-firefox.js`.
- transformers.js configured with `env.allowRemoteModels = false`, `env.localModelPath` → extension URL (tokenizer bundled), `env.backends.onnx.wasm.wasmPaths` → bundled ORT WASM binaries. No CDN fetches.
- Batch size 1 per `run()` in WASM.
- Threading: check `crossOriginIsolated` at startup.
  - Chrome: expected `true` via the manifest COOP/COEP keys; enable `numThreads = min(4, hardwareConcurrency)`.
  - Firefox: not guaranteed; if `false`, single-thread + SIMD, latency ~2–3× worse.
  - Observed mode is written to `storage.local` and shown on the options page.

### 6.4 Model distribution

- Model bundle is **not** packaged in the extension (store size limits; decouples extension and model versions).
- Downloaded on first run from a static host owned by the author; each file checked against the shipped `SHA256SUMS` before use; cached via the Cache API.
- Model version string in `calibration.json` is the cache key for all scores.
- Options page "re-download model" clears Cache API entries and the score cache.

### 6.5 First-run and permissions

- **Firefox MV3 does not grant host permissions at install.** Content scripts do nothing until the user enables site access. First-run page (opened via `runtime.onInstalled`) explains this and calls `chrome.permissions.request({ origins: ["<all_urls>"] })` on a button click; also documents the manual path in the extension's permissions panel.
- **Chrome** grants `<all_urls>` at install with the standard warning. Same first-run page is shown; the request is a no-op.
- The model download starts from the first-run page after permissions are confirmed, with progress shown.

### 6.6 Distribution and install

| | Firefox | Chrome |
|---|---|---|
| Personal use | AMO **unlisted** signing → signed `.xpi`; or Developer Edition with `xpinstall.signatures.required=false` | Load unpacked in `chrome://extensions` developer mode (persistent "disable developer mode extensions" prompt on launch); or self-hosted `.crx` via `ExtensionInstallForcelist` enterprise policy on own machine |
| Public | AMO listed | Chrome Web Store (one-time developer fee, review) |

Both builds come from the same source tree; the only per-browser build step is copying the correct router file and stripping unknown manifest keys if a store validator rejects them.

---

## 7. Content script (shared)

### 7.1 Extraction

- Candidates: `p`, `li`, `blockquote`, `dd`, `td` with substantial text, and text-bearing descendants of `article`/`main`/Readability-identified content root.
- Excluded: `nav`, `header`, `footer`, `aside`, `code`, `pre`, `form`, `textarea`, `[contenteditable]`, `[role=navigation]`, embedded comment widgets, `iframe` content.
- Skip blocks under `min_words` (from `calibration.json`, expected ~40).
- SPAs: `MutationObserver` on the content root, debounced 500 ms, re-runs extraction on added subtrees only.

### 7.2 Language gate

Compact n-gram language detector (franc/cld3-class, bundled) per block. Non-English blocks are never scored. Tuned to prefer "unknown → skip" over misclassifying short blocks as English.

This gate is not one of the coupling points and need not match the training-side detector exactly, but there is an interaction worth watching. franc is weak on short non-native English, and "unknown → skip" means the extension may silently never score exactly the prose §4.2 deliberately over-samples as a hard negative — trained on, never scored. `fixtures/langgate.json` asserts both that the two gates agree on acceptance, and, more importantly, that the cases where they disagree are not systematically non-native.

### 7.3 Segmentation

Score units are paragraph-level blocks. Blocks over ~400 words split at sentence boundaries into ≤ 400-word chunks. Blocks 40–79 words are scored but carry a length penalty in pooling (§8).

### 7.4 Hashing and requests

- Key = `sha256(normalize_for_hash(text))`; that function collapses whitespace, strips zero-width chars, and folds quotes and dashes. Content-keyed, so syndicated text is scored once.
- **`normalize_for_hash` is for the cache key only. It is not what the model is fed.** Curly quotes and em-dash density are among the more reliable surface cues for AI text, and folding them before inference would discard signal the model was trained on. The model receives `collapse_whitespace(text)`, which regularizes whitespace and nothing else. Two functions, and both must match the Python side.
- Neither uses a regex `\s`. Python's `\s` matches U+001C–U+001F and JavaScript's matches U+FEFF instead, so relying on it would guarantee a drift bug that only appears on scraped text. Both implementations enumerate the whitespace set explicitly, and `fixtures/normalize.json` pins every disputed codepoint.
- Requests carry viewport distance (`IntersectionObserver`, rootMargin one screen) for queue priority. Off-screen blocks are queued at low priority.
- Port reconnect logic: on `onDisconnect`, reconnect and re-send any outstanding hashes (covers Chrome SW restarts and Firefox event page reloads identically).

### 7.5 Applying results

- CSS Custom Highlight API (`CSS.highlights`; Chrome ≥ 105, Firefox ≥ 140) so the DOM isn't mutated. Fallback: wrapper `span` with a class.
- Hover tooltip on flagged runs: run score, word count, model version.

---

## 8. Scoring and aggregation (shared)

All constants below come from `calibration.json`, not from the source (§5). `eval/aggregate.py` is the reference implementation and `fixtures/aggregate.json` pins it for both languages; where the prose is ambiguous, **the fixture is the specification**.

- Chunk score `p = sigmoid(logit / temperature)`.
- **Length penalty**: chunks under 80 words have `p` shrunk toward 0.5 by `words / 80`, linearly in probability space, before pooling.
- **Run pooling**: adjacent flagged chunks form a run; run score = unweighted mean of the post-penalty log-odds. A run totaling ≥ 150 words is flagged. "A single isolated chunk under 150 words is not flagged" is not a separate rule — it is this one.
- **Document prior**: if fewer than 20% of scored chunks exceed `t_off` (strict `>`), raise `t_on` for that document — **by a fixed offset in log-odds space**.

  The specified +0.05 in probability space is unsafe. A ≤1% FPR operating point puts `t_on` around 0.95–0.99, where +0.05 yields a threshold above 1.0: every sparse document silently becomes unflaggable, with no error anywhere. In log-odds the bump is scale-free and cannot leave the unit interval.
- **Hysteresis**: a run opens at a chunk ≥ `t_on` and extends over neighbours ≥ `t_off`. This is *spatial*, not temporal — there is no state carried between scoring passes, so re-scoring a page after a DOM mutation always yields the same result.

---

## 9. Rendering (shared)

- Effective background = first non-transparent computed `background-color` walking up ancestors.
- De-emphasis: `color: color-mix(in oklab, currentColor 45%, <effective-bg>)`. Works on dark and tinted themes.
- Secondary cue: 2 px dashed left border in the same mixed color on the run's block ancestors.
- No `opacity` or `filter`.
- Per-site allowlist disables extraction on listed origins.

`color-mix()` and `::highlight()` are supported in both target browsers at the minimum versions above.

---

## 10. Performance budget

Estimates for int8 ModernBERT-base, WASM+SIMD, 2024-era laptop CPU; to be measured on both browsers before `max_length` is finalized:

| Mode | 512-token chunk | 256-token chunk |
|---|---|---|
| single thread | ~600–1200 ms | ~300–600 ms |
| 4 threads | ~200–400 ms | ~100–200 ms |

Chrome is expected to land in the threaded row (offscreen document is cross-origin isolated via manifest); Firefox may land in either.

Memory: ~150 MB weights + ~100 MB WASM heap/activations, resident in the offscreen document (Chrome) or event page (Firefox).

Latency knobs, in order of effect: cap `max_length` at 256, viewport-first queue, score-cache hit rate. Fallback is a re-fine-tuned ModernBERT with 6–8 of 22 layers dropped — a model change, not an extension change.

---

## 11. Storage (shared)

- `storage.local`: score cache `{hash → {p, words, modelVersion, ts}}`, LRU-capped at ~50k entries; site allowlist; user threshold override; observed threading mode.
- Cache API: model bundle.
- Nothing else persisted. No remote logging.

---

## 12. Browser parity checklist

Items that differ by browser and must be handled explicitly; everything else is shared code.

| Concern | Chrome | Firefox |
|---|---|---|
| Background | MV3 service worker (`service_worker`) | MV3 event page (`background.scripts`) |
| Session host | Offscreen document | Event page + keepalive port + warm restore |
| Cross-origin isolation | Manifest COOP/COEP → `crossOriginIsolated === true` | Runtime check; may be single-threaded |
| Host permission at install | Granted | Must be requested by user gesture |
| Storage quota | `unlimitedStorage` required | `unlimitedStorage` accepted, not strictly needed |
| Unknown manifest keys | `browser_specific_settings` ignored | `offscreen`, COOP/COEP ignored |
| Signing | Unpacked / Web Store | AMO signing required |

---

## 13. Deliverables (single release)

- One source tree producing two builds: Chrome (unpacked or `.crx`) and Firefox (signed `.xpi`). Shared: content script, `host.js`, inference worker, bundled ORT WASM binaries, tokenizer files, language detector, options page, first-run page. Per-browser: router file, manifest post-processing.
- Hosted model bundle (§5) at a fixed URL with `SHA256SUMS`.
- Training/export repo: corpus generation, training config, calibration script producing `calibration.json`, export + quantization + verification, RAID eval. Runs on Modal against a Volume; a thin local CLI launches stages.
- Cross-language fixtures (`fixtures/`) emitted by the Python side and read by both test suites.
- Measured numbers in `docs/measurements/`: per-genre FPR with confidence bounds, the global-vs-per-genre-oracle recall gap, the corpus shortcut-probe battery, int8 parity, RAID, and the in-the-wild eval.

### Corpus scale

1.5M training windows, class-balanced at 750k human and 750k AI, from ~300k human documents and ~250k AI rows. **Windows are the binding quota, not documents**: four of the eight genres are intrinsically short-form, so real documents yield ~3 windows each rather than the 5 a naive count assumes.

### Corpus gate

A battery of deliberately weak baselines runs on the finished corpus and **must pass before any training run starts**. It uses pass *bands*, not ceilings: a bag-of-words model at 0.99 AUC means the corpus is trivially separable and the real number will not survive contact with the web, but the same model at 0.55 also means something is wrong, because AI text genuinely does carry lexical signatures and destroying them means the two classes were preprocessed differently.

The sharpest single check is transfer to held-out generators. The generation harness is shared across every model; a model's writing style is not. If a bag-of-words probe transfers to unseen generators as well as it does in-distribution, it has learned our pipeline rather than AI writing.
