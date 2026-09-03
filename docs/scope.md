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
- False positives concentrate on formal, edited, templated, and non-native-English prose. These are hard negatives in training and the threshold is set per genre.
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

- 6–8 current models × temperature/sampling variants × prompt styles: continuation, rewrite-this, "write a blog post about X", instruction with persona, structured/listicle output.
- Attack variants: paraphrase via a second model, humanizer-style rewrites, sentence shuffling, light typo injection.
- Mixed authorship: human-draft-then-AI-edit and AI-draft-then-human-edit at several edit fractions.

**Labels**

Regression target = AI fraction ∈ [0, 1]. Binary AI/human label derived at ≥ 0.7 for evaluation. Regression head gives the UI a continuous dial without retraining.

### 4.3 Windowing

Train on the unit scored at inference: random windows cut at sentence boundaries, 40–400 words, weighted toward 80–250 words. Short windows are kept so the model learns low confidence on them rather than confident noise.

### 4.4 Head and loss

- `[CLS]` pooling (mean pooling acceptable) → dropout 0.1 → linear.
- Primary loss: BCE on AI-fraction target, label smoothing 0.05.
- Auxiliary head: genre classification, cross-entropy, weight 0.2, dropped at export. Purpose: discourage "formal register ⇒ AI" shortcut.
- Full fine-tune. lr 3e-5, layerwise lr decay 0.9, 2–3 epochs, warmup 6%.
- Validation metric: FPR at fixed recall, reported **per genre**.

### 4.5 Calibration and threshold

- Temperature-scale on a held-out set.
- Operating threshold at **≤ 1% FPR** on human web text overall **and ≤ 2% FPR on every genre**. Recall is whatever that yields.
- Temperature and threshold shipped as constants with the model file; selected on the **quantized** artifact (§5).
- Hysteresis pair for the UI (§8): `t_on`, `t_off = t_on − 0.1` in probability space.

### 4.6 External evaluation

RAID (out-of-domain + adversarial splits) for a comparable external number, noting its generator set skews to 2023-era models.

### 4.7 Model refresh

Regenerate the AI side with current models when warranted; retrain, recalibrate, re-export, bump the model version string. Neither extension changes. Cached scores are keyed on model version, so a new model invalidates old scores automatically on both browsers.

---

## 5. Export and quantization

1. Load checkpoint with `attn_implementation="eager"` (or `sdpa`). The unpadded flash-attention path does not trace.
2. Export via Optimum: opset ≥ 17, dynamic batch and sequence axes, tokenizer `max_length=512`.
3. Verify ONNX logits vs PyTorch on ~200 samples, tolerance ~1e-3.
4. Int8 dynamic quantization (`quantize_dynamic`, MatMul/Gemm only). ~600 MB fp32 → ~150 MB int8.
5. Re-run calibration and per-genre FPR on the int8 model; set `t_on`/temperature from this artifact.
6. Confirm the exported `tokenizer.json` includes the `[CLS]`/`[SEP]` post-processor so browser tokenization matches training.

**One artifact for both browsers.** WASM int8 is the execution path on Chrome and Firefox alike. Chrome could additionally run a WebGPU execution provider, but that requires a separate fp16 artifact with its own calibration and would break behavioural parity; not adopted. fp16 on WASM is not useful (no native fp16 compute). 4-bit weight-only is not adopted unless verified accuracy-neutral.

Shipped model bundle: `model.onnx` (int8), `tokenizer.json`, `tokenizer_config.json`, `config.json`, `calibration.json` (`{version, temperature, t_on, t_off, min_words}`), `SHA256SUMS`.

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

### 7.3 Segmentation

Score units are paragraph-level blocks. Blocks over ~400 words split at sentence boundaries into ≤ 400-word chunks. Blocks 40–79 words are scored but carry a length penalty in pooling (§8).

### 7.4 Hashing and requests

- Key = `sha256(normalize(text))`; `normalize` collapses whitespace, strips zero-width chars, normalizes quotes/dashes. Content-keyed, so syndicated text is scored once.
- Requests carry viewport distance (`IntersectionObserver`, rootMargin one screen) for queue priority. Off-screen blocks are queued at low priority.
- Port reconnect logic: on `onDisconnect`, reconnect and re-send any outstanding hashes (covers Chrome SW restarts and Firefox event page reloads identically).

### 7.5 Applying results

- CSS Custom Highlight API (`CSS.highlights`; Chrome ≥ 105, Firefox ≥ 140) so the DOM isn't mutated. Fallback: wrapper `span` with a class.
- Hover tooltip on flagged runs: run score, word count, model version.

---

## 8. Scoring and aggregation (shared)

- Chunk score `p = sigmoid(logit / temperature)`.
- **Length penalty**: chunks under 80 words have `p` shrunk toward 0.5 by `words / 80` before pooling.
- **Run pooling**: adjacent flagged chunks form a run; run score = mean log-odds. A single isolated chunk above `t_on` under 150 words is **not** flagged; a run totaling ≥ 150 words is.
- **Document prior**: if fewer than 20% of scored chunks exceed `t_off`, raise `t_on` for that document by +0.05.
- **Hysteresis**: a run flags at `t_on` and unflags only below `t_off`.

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
- Training/export repo: corpus generation, training config, calibration script producing `calibration.json`, export + quantization + verification, RAID eval.
