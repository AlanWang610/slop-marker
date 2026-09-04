# extension/

TypeScript. One source tree, two builds (scope.md §13). Node 24.

```bash
npm install
npm run build:chrome     # -> dist/chrome   (load unpacked)
npm run build:firefox    # -> dist/firefox  (web-ext / signed .xpi)

npm test                 # unit + cross-language parity
npm run typecheck
npm run lint
npm run e2e              # builds, then drives a real Chromium
```

`npm test` and `npm run e2e` both need a local model bundle:

```bash
uv run --extra modal python ../tools/pull_bundle.py --version mb-base-0.2.0-dev
```

Without it the model-dependent tests **skip** rather than fail — `artifacts/` never enters
git, so a fresh checkout legitimately does not have one.

## What runs where

| | Chrome | Firefox |
|---|---|---|
| Background | MV3 service worker, stateless | MV3 event page |
| Host (session, cache, queue) | Offscreen document | The event page itself |
| Model download | Service worker | The event page |
| Threads | 4 (cross-origin isolated) | Runtime check; may be 1 |

The routers are the only browser-specific runtime code. `router/shared.ts` holds the
protocol, `host/host.ts` the session and queue, and neither knows which browser it is on.

**The download runs in the router, not the host, and that is load-bearing.** The manifest
sets `cross_origin_embedder_policy: require-corp` so extension pages are cross-origin
isolated and ORT can use threads — worth 3.6× (`docs/measurements/extension-latency.md`).
Under `require-corp` a cross-origin fetch needs a CORS-successful response or a CORP header,
and GitHub release assets send neither. So the service worker fetches, verifies and writes
to the Cache API, and the offscreen document only reads. See `host/model-store.ts`.

Only `model.onnx` is downloaded. The tokenizer and `calibration.json` ship inside the
extension (§6.3 bundles the tokenizer so browser tokenization cannot drift), so first run
fetches one 137 MB file rather than a 140 MB directory.

## The coupling points

Five files in `src/shared/` are transcriptions of Python originals, and every one is pinned
by a fixture both test suites read. If you change one, change the Python side and regenerate
— **the fixture is the specification**, not this code.

| TypeScript | Python | Fixture |
|---|---|---|
| `normalize.ts` | `data/normalize.py` | `normalize.json` |
| `sentences.ts` | `data/sentences.py` | `windows.json` |
| `chunking.ts` | `data/chunking.py` | `windows.json` |
| `calibration.ts` | `eval/calibration.py` | `aggregate.json` |
| `aggregate.ts` | `eval/aggregate.py` | `aggregate.json` |
| `tokenize.ts` | (truncation semantics) | `tokenize.json` |
| the whole pipeline | `eval/documents.py` | `documents.json` |

`bundle-config.ts` is **generated** by `tools/sync_extension_assets.py`. No calibration
constant is written by hand anywhere; that is what makes §4.7 true.

## Things that will bite you

**Codepoints, not UTF-16.** Python indexes codepoints. `splitSentences` returns codepoint
offsets and everything works on `Array.from(text)`. A naive `text[i]` shifts every offset
after an astral character, and `normalize.json` has cases that catch it.

**Never use `\s`.** Python's matches U+001C–U+001F, JavaScript's matches U+FEFF. Both sides
enumerate the whitespace set explicitly. U+200B is invisible-and-deleted, not whitespace.

**Truncation differs between the tokenizers.** Python truncates the content then applies the
post-processor, keeping `[SEP]`; transformers.js slices the post-processed sequence and drops
it. Worth 0.42 of logit, and `chunk_block`'s 400-word cap means long chunks truncate
routinely (§4.3). `shared/tokenize.ts` handles it — do not call transformers.js truncation
directly.

**Import `onnxruntime-web/wasm`, not `onnxruntime-web`.** The default entry is the JSEP
build and dynamically imports a 26 MB companion for WebGPU/WebNN, which §5 rules out for
parity. A missing runtime file surfaces as a flat "no available backend found".

**Blocks under `min_words` are skipped.** A page of short paragraphs is scored not at all,
which looks exactly like a page the model declined to flag.

**Nothing shows an offscreen document's console.** Failures are recorded to
`storage.local.lastError` as well as logged, because otherwise a Host that cannot start is
indistinguishable from a page with no AI text on it.

## Firefox packaging

`web-ext lint --source-dir dist/firefox` reports **0 errors**. Three of the four warnings
come from the bundled ONNX Runtime glue and are inherent to running WASM — one
`DANGEROUS_EVAL` (the `Function` constructor) and two `UNSAFE_VAR_ASSIGNMENT` (the dynamic
`import()` of the runtime loader). They need explaining at AMO review; they do not block
unlisted signing. The fourth is `KEY_FIREFOX_ANDROID_UNSUPPORTED_BY_MIN_VERSION`, which is
moot — mobile is a §1 non-goal.

The manifest declares `data_collection_permissions: { required: ["none"] }`, which AMO now
requires and which is simply accurate: all inference is on-device and nothing is persisted
beyond the local caches (§1, §11).

## Known gaps

- **Firefox has never been run.** The build lints clean and its manifest is correct, but no
  Firefox has loaded it. `web-ext run --source-dir dist/firefox` is the next step.
- **The offscreen wrapper is unverified.** Headless Chromium creates offscreen documents but
  never executes their scripts, so `e2e/run.mjs` opens `host.html` as an ordinary tab. Same
  code, same manifest, same relay — only the ~40-line wrapper goes untested.
- **The model host is a placeholder.** `bundle-config.ts` points at
  `https://github.com/OWNER/REPO/releases/download`. Set it with
  `tools/sync_extension_assets.py --base-url`, and publish the bundle there.
- **No `fixtures/langgate.json`.** §7.2 asks for one asserting the two language gates agree,
  and more importantly that their disagreements are not systematically non-native.
