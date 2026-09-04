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
npm run e2e:firefox      # builds, then drives a real Firefox (geckodriver)
npm run e2e:all          # both

# variants
node e2e/run.mjs --headed        # uses the real offscreen document (headless cannot)
node e2e/run.mjs --real-host     # fetches the bundle from the published release
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
| Threads | **4** (cross-origin isolated) | **1** (COOP/COEP stripped, so not isolated) |

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
| `content/langgate.ts` | `corpus/lang.py` (agreement only) | `langgate.json` |

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

**A Chrome offscreen document has only `chrome.runtime`.** Measured over CDP:
`Object.keys(chrome)` is `csi, loadTimes, runtime`, and `chrome.storage` is `undefined`.
Everything else the host needs is a web API and is present — `caches`, `indexedDB`,
`Worker`, `SharedArrayBuffer`, `crossOriginIsolated === true`. Use `shared/storage.ts`
(which proxies through the service worker) rather than `chrome.storage` in anything the
host imports.

**Nothing shows an offscreen document's console.** Failures are recorded to
`storage.local.lastError` as well as logged, because otherwise a Host that cannot start is
indistinguishable from a page with no AI text on it. To see inside one, attach over CDP:
launch with `--remote-debugging-port`, find the `host.html` target in `/json/list` (it
reports as `background_page`), and connect to its `webSocketDebuggerUrl`.

**`npm run e2e` runs headless, which cannot execute offscreen documents at all** — it
substitutes `host.html` as a tab. `npm run e2e -- --headed` uses the real offscreen
document, and is the only thing that would have caught the `chrome.storage` bug.

**A port to a terminated service worker stays writable and swallows what you post.**
Measured: stop the worker over CDP, and the content script's `onDisconnect` never fires, so
every later score request vanishes and the page is never scored again. `onDisconnect` alone
is therefore not a sufficient signal — `content/index.ts` also watches for silence, and
rebuilds the port when nothing has come back for 20 s with chunks still outstanding.

**`closest()` matches the element itself.** Two separate bugs came from this. `td` is a §7.1
candidate, but `table` had been added to the exclusion list, so every cell matched its own
ancestor and no table cell was ever scored. And the nested-candidate guard read
`element.closest(CANDIDATES) !== element`, which can never be true — so `<li><p>…</p></li>`
produced two blocks over the same text, double-counting it in the §8 document prior.

**Block boundaries are word boundaries, and minified HTML has no whitespace node at them.**
`<li><p>Alpha.</p><p>Beta.</p></li>` walks to `Alpha.Beta.` unless extraction inserts the
space itself: one word short, and a token the model never saw. Hand-written test pages are
pretty-printed and hide this; most of the real web is not.

**The content root must be pinned, not re-derived.** `contentRoot()` prefers the first
`<article>`, so recomputing it every pass means an SPA that appends articles as the reader
scrolls narrows the root to the first one it added — and everything after that falls outside
it and is silently never scored. Measured on a three-batch feed: batch one highlighted,
batches two and three invisible. `extract.ts::pinRoot` holds the root and re-derives it only
once the element has left the document.

**A client-rendered page has nothing to score at `document_idle`.** That is precisely the
case §7.1's `MutationObserver` exists for, so extraction must install its observers even
when the first pass finds no blocks at all — otherwise the pages that most need rescanning
are the ones that never get it.

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

### Signing

Release and beta Firefox will not permanently install an unsigned extension, and there is
no preference to override that. `about:debugging` takes the unsigned archive as a temporary
add-on, but it is gone at restart. So a usable Firefox build has to be signed.

Signing is on the **unlisted** channel: signed for self-distribution, not published in the
AMO directory, which is what §13 asks for. Get a JWT issuer and secret from
[the AMO API key page](https://addons.mozilla.org/developers/addon/api/key/), put them in
the environment, and run:

```
$env:WEB_EXT_API_KEY = "user:12345:67"      # PowerShell; export ... on a shell
$env:WEB_EXT_API_SECRET = "..."
npm run package -- --browser=firefox --sign
```

Credentials are read from the environment and never from arguments, so they stay out of
shell history and the process list. Afterwards the returned archive is checked for a
`META-INF/mozilla.*` entry: an unsigned `.xpi` is byte-for-byte plausible and only reveals
itself when Firefox refuses it at install time, which is exactly the failure worth catching
early.

Two things about it are effectively one-way. **The extension id is fixed at first signing**
— it is `slop-marker@alanwang610.github.io`, a namespace tied to the GitHub account rather
than the `@localhost` it started as, because changing it afterwards produces a different
extension and every install has to be removed and re-added by hand. And **AMO refuses a
version it has already seen**, so re-signing means bumping `version` in
`src/assets/manifest.json` first.

## Verified

| | |
|---|---|
| unit, DOM and cross-language parity | 614 tests across 21 files |
| Chrome, headless (host as a tab) | 52/52 |
| Chrome, headed (real offscreen document) | 51/51 |
| Firefox (geckodriver, real UI) | 27/27 |
| Firefox, installed from the **AMO-signed** `.xpi` | 27/27 |
| against the published release host | Chrome 42/42, Firefox 27/27 |
| packaging (`.crx` + `.xpi`, read back) | 8/8 |
| `web-ext lint` | 0 errors |
| Python side | 395 tests |

The counts differ by browser because some checks only apply to one. Headless opens
`host.html` as a tab and can assert its cross-origin isolation directly; headed uses the
real offscreen document instead. `--real-host` skips the corrupted-download pair, which
needs a server of ours to corrupt.

Measured contrast of the de-emphasis (scope.md 9), against a 3.0:1 floor — WCAG AA for
large text, and the point of "de-emphasize, never hide":

| ground | flagged text | ordinary text |
|---|---|---|
| light page | 4.88:1 | 21:1 |
| dark page (`#111`) | 3.49:1 | 16.28:1 |
| tinted callout | 3.49:1 | 13.5:1 |

The dark case is the tight one. `::highlight()` styles cannot be read back with
`getComputedStyle`, so the harness resolves the same `color-mix()` onto a 1×1 canvas and
reads the pixel — Chrome serialises the computed value as `oklab(...)`, which no rgb parse
would have caught.

Two of the vitest files -- `model-parity` and `document-parity` -- skip themselves unless
`artifacts/bundles/<version>/model.onnx` is present, since the bundle never enters git.

The saved corpus in `e2e/real/` is served locally with every external request blocked, so a
run does not depend on the network and does not re-fetch from nasa.gov each time. It reports
how many chunks each page produced as well as whether any was flagged: without that, "no
false positives" on a page whose prose extraction missed entirely would be a vacuous pass.

`--real-host` is not a formality: GitHub release assets carry no
`Access-Control-Allow-Origin`, which is the entire reason the download runs in the service
worker rather than the cross-origin-isolated offscreen document. A local test server sending
`access-control-allow-origin: *` would let that regression through unnoticed.

## Known gaps

Everything below is *not* covered by any automated test. It is listed rather than implied,
because the failures that cost the most here were all invisible ones -- a page that is never
scored looks exactly like a page with nothing to flag.

**The real-page corpus is small, and one-sided.** `e2e/real/` holds four saved pages, and
they only test the false-positive direction, because that is the only direction a licence
allows: every source is a work of the United States Government and so not subject to
copyright (17 U.S.C. 105), and all four are human-written. There is no saved page known to
be AI-written, so the *positive* direction is still only tested on generated markup. Four
pages is also a thin sample -- it says nothing about paywalls, infinite feeds on real
sites, or lazy images that reflow the page under a highlight.

**Re-signing needs a version bump.** `0.1.0` is signed and installable; AMO refuses a
version it has already accepted, so the next signed build needs `version` raised in
`src/assets/manifest.json` first. The harness installs the signed archive in preference to
the unsigned one, so `npm run e2e:firefox:xpi` tests what a user actually installs — but
only as long as someone re-runs `--sign` after changing the extension. An unsigned rebuild
sitting beside a stale signed archive is the failure mode to watch for.

Chrome has no equivalent step, and its `.crx` is not an install route either: Chrome refuses
off-store `.crx` files, so unpacked loading is the path for personal use and the `.crx`
exists to prove the packaging is sound. That asymmetry is worth remembering — the Chrome
build a user runs is `dist/chrome`, a directory nothing signs and nothing pins.

The Chrome signing key is generated on the first `npm run package` and kept at
`artifacts/packages/chrome-key.pem`, which is gitignored. It *is* the extension's identity:
lose it and every existing install sees a different extension id.

**Firefox's site-access grant is bypassed.** The harness sets
`extensions.originControls.grantByDefault` rather than clicking the permission prompt, which
is the one step of the scope.md 6.5 first-run flow no test performs.

**Service-worker idle termination cannot be tested under automation.** Chrome does not retire
an MV3 service worker while a debugger is attached, and Playwright is always attached. The
port-reconnect check therefore stops the worker explicitly over CDP instead of waiting out
the 30 s idle timer.

### Manual pass

Worth walking before calling a build good, since none of it is automated:

- [ ] A known-AI article highlights, and the tooltip reports a sensible score and word count.
- [ ] A hard negative does not: a product listing, and a forum thread written by a
      non-native English speaker (the case scope.md 2 names as the main FPR risk, and the
      one the language gate may quietly decline to score at all). The press-release genre
      is now covered by the saved corpus.
- [ ] A page in a language other than English is left alone.
- [ ] Scrolling a long page fast does not leave stale highlights behind, and the text under
      the cursor is scored before the text far below it.
- [ ] Leave a tab idle for a minute, then scroll: scoring resumes rather than stalling.

Contrast, dark and tinted grounds, the infinite-scroll case and the per-site allowlist used
to be on this list and are now asserted by the browser harnesses. What no assertion can
settle is whether the result *looks* right on a page nobody wrote for the test.


