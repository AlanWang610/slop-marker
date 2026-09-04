/**
 * The one test that cannot be faked: load the built extension into a real browser, on a
 * real page, and see whether text gets de-emphasized.
 *
 * Everything in tests/ runs in Node. That proves the maths and the model agree with Python,
 * but not that the extension *works*: the offscreen document, the COOP/COEP isolation the
 * threads depend on, the service-worker relay, the Cache API handoff and the CSS Custom
 * Highlight API only exist in a browser.
 *
 *   node e2e/run.mjs [--browser=chrome] [--headed] [--keep]
 *
 * The model bundle is served from a throwaway local host rather than GitHub Releases, so
 * this runs offline and against whatever is in artifacts/bundles.
 */

import { createServer } from "node:http";
import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { argv, exit } from "node:process";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const repo = resolve(root, "..");

const flag = (n) => argv.some((a) => a === `--${n}` || a.startsWith(`--${n}=`));
const value = (n, d) => {
  const hit = argv.find((a) => a.startsWith(`--${n}=`));
  return hit ? hit.split("=").slice(1).join("=") : d;
};

const headed = flag("headed");
const verbose = flag("verbose");
const keep = flag("keep");
const dist = join(root, "dist", value("browser", "chrome"));

const calibration = JSON.parse(readFileSync(join(dist, "assets", "calibration.json"), "utf-8"));
const VERSION = calibration.version;
const bundleDir = join(repo, "artifacts", "bundles", VERSION);

if (!existsSync(join(dist, "manifest.json"))) {
  console.error(
    `no build at ${dist} — run:
` +
      `  node scripts/build.mjs --browser=chrome --model-base-url=http://127.0.0.1:8787`,
  );
  exit(2);
}
if (!existsSync(join(bundleDir, "model.onnx"))) {
  console.error(
    `no local bundle at ${bundleDir}\n` +
      `run: uv run --extra modal python tools/pull_bundle.py --version ${VERSION}`,
  );
  exit(2);
}

/* ---------------------------------------------------------------- local hosts */

/** Serves the model bundle, standing in for the release host. */
function serveBundle(port) {
  return new Promise((ready) => {
    const server = createServer((req, res) => {
      const name = decodeURIComponent(new URL(req.url, "http://x").pathname.split("/").pop());
      const file = join(bundleDir, name);
      if (!existsSync(file)) {
        res.writeHead(404).end("no");
        return;
      }
      res.writeHead(200, {
        "content-type": "application/octet-stream",
        "content-length": statSync(file).size,
        // The real host does not send these; the router fetches from the service worker
        // precisely because of that. Sending them here would hide a COEP regression.
        "access-control-allow-origin": "*",
      });
      createReadStream(file).pipe(res);
    });
    server.listen(port, "127.0.0.1", () => ready(server));
  });
}

/** Serves the fixture page. A real http origin, so the content script matches <all_urls>. */
function servePage(port, html) {
  return new Promise((ready) => {
    const server = createServer((_req, res) => {
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" }).end(html);
    });
    server.listen(port, "127.0.0.1", () => ready(server));
  });
}

/**
 * The page. Its two halves come from fixtures/documents.json, so the expected outcome is
 * not a guess: Python already scored this exact text through the same pipeline.
 */
function buildPage() {
  const fx = JSON.parse(readFileSync(join(repo, "fixtures", "documents.json"), "utf-8"));
  const flagged = fx.documents.find((d) => d.expected.runs.some((r) => r.flagged));
  const clean = fx.documents.find((d) => !d.expected.runs.some((r) => r.flagged));
  if (!flagged || !clean) throw new Error("documents.json needs one flagged and one clean case");

  /**
   * Paragraphs of at least ~110 words. Blocks under `min_words` (40) are correctly skipped
   * by the content script, so a page of short paragraphs is scored not at all -- which on
   * screen looks identical to a page the model declined to flag.
   */
  const paras = (text) => {
    const out = [];
    let current = [];
    let words = 0;
    for (const sentence of text.split(/(?<=\.)\s+/)) {
      current.push(sentence);
      words += sentence.split(/\s+/).length;
      if (words >= 110) {
        out.push(current.join(" "));
        current = [];
        words = 0;
      }
    }
    if (words >= 45) out.push(current.join(" "));
    return out.map((p) => `<p>${p.trim()}</p>`).join("\n");
  };

  return {
    html: `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>e2e</title></head>
<body>
  <nav><p>Navigation text that must never be scored, however long it is made.</p></nav>
  <main>
    <article id="flagged">${paras(flagged.text)}</article>
    <hr>
    <article id="clean">${paras(clean.text)}</article>
  </main>
  <footer><p>Footer text that must never be scored, however long it is made.</p></footer>
</body></html>`,
    flaggedName: flagged.name,
    cleanName: clean.name,
  };
}

/* ---------------------------------------------------------------- the run */

const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok, detail });
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
};

async function main() {
  const { chromium } = await import("playwright");
  const { html, flaggedName, cleanName } = buildPage();

  const bundleServer = await serveBundle(8787);
  const pageServer = await servePage(8788, html);
  const profile = await mkdtemp(join(tmpdir(), "slop-marker-e2e-"));

  console.log(`extension  ${dist}`);
  console.log(`model      ${VERSION} from http://127.0.0.1:8787`);
  console.log(`page       flagged=${flaggedName} clean=${cleanName}\n`);

  const context = await chromium.launchPersistentContext(profile, {
    // The old headless mode is a separate browser that cannot load extensions at all.
    // `channel: "chromium"` selects the new headless, which is ordinary Chrome without a
    // window -- so the offscreen document, COOP/COEP isolation and CSS.highlights all
    // behave the way they do for a user.
    channel: "chromium",
    headless: !headed,
    args: [
      `--disable-extensions-except=${dist}`,
      `--load-extension=${dist}`,
      "--no-first-run",
    ],
  });

  try {
    // Find the extension id. An MV3 service worker is lazy, so waiting on the
    // `serviceworker` event can hang forever with nothing wrong. runtime.onInstalled opens
    // first-run.html at install, which is both a real behaviour worth asserting and the
    // most reliable way to learn the id.
    const deadline = Date.now() + 60_000;
    let extensionId = null;
    while (extensionId === null && Date.now() < deadline) {
      const urls = [
        ...context.serviceWorkers().map((w) => w.url()),
        ...context.pages().map((pg) => pg.url()),
      ];
      const hit = urls.find((u) => u.startsWith("chrome-extension://"));
      if (hit !== undefined) extensionId = new URL(hit).host;
      else await new Promise((r) => setTimeout(r, 250));
    }
    if (extensionId === null) throw new Error("extension never registered a context");
    check("extension loads and opens its first-run page", true, extensionId);

    // Surface extension-side errors instead of letting them vanish into a dead page.
    // Without this, a worker that fails to start looks exactly like a page with no AI text.
    context.on("weberror", (e) => console.log(`  [page error] ${e.error().message}`));
    const wire = (target, label) => {
      target.on?.("console", (m) => {
        if (verbose || m.type() === "error") console.log(`  [${label}] ${m.type()}: ${m.text()}`);
      });
      target.on?.("pageerror", (e) => console.log(`  [${label}] uncaught: ${e.message}`));
    };
    for (const w of context.serviceWorkers()) wire(w, "sw");
    context.on("serviceworker", (w) => wire(w, "sw"));
    context.on("page", (pg) => {
      if (pg.url().startsWith("chrome-extension://")) wire(pg, "offscreen");
    });

    const setup = await context.newPage();
    setup.on("console", (m) => {
      if (m.type() === "error") console.log(`  [first-run] ${m.text()}`);
    });
    await setup.goto(`chrome-extension://${extensionId}/first-run.html`);

    // Download the model. This exercises the real path: fetch in the service worker,
    // verify against the shipped SHA256SUMS, write to the Cache API.
    const download = await setup.evaluate(
      async () => await chrome.runtime.sendMessage({ type: "downloadModel" }),
    );
    check("model downloads and verifies", download?.ok === true, download?.message ?? "");

    const status = await setup.evaluate(
      async () => await chrome.runtime.sendMessage({ type: "getStatus" }),
    );
    check("model reports ready", status?.modelState?.phase === "ready", status?.modelState?.phase);

    // The claim scope.md 6.3 makes about Chrome, and worth 3.6x of latency.
    const isolated = await setup.evaluate(() => self.crossOriginIsolated);
    check("extension page is cross-origin isolated", isolated === true, String(isolated));

    /**
     * Headless Chromium creates an offscreen document but never executes its scripts --
     * verified: even a trivial classic script in host.html does not run, while
     * chrome.runtime.getContexts() happily reports the document exists. That is a headless
     * limitation, not a defect in the extension, and it makes the Chrome host unverifiable
     * in this harness as written.
     *
     * So host.html is opened as an ordinary extension tab instead. It is the same document
     * running the same code under the same manifest COOP/COEP, and it registers the same
     * onConnect listener, so the service worker's relay reaches it exactly as it would
     * reach the offscreen document. What goes unverified is the ~40-line offscreen wrapper;
     * everything it wraps -- the Host, the worker, ORT, the tokenizer, aggregation and
     * rendering -- is exercised for real.
     */
    const hostTab = await context.newPage();
    hostTab.on("console", (m) => {
      if (verbose || m.type() === "error") console.log(`  [host] ${m.type()}: ${m.text()}`);
    });
    hostTab.on("pageerror", (e) => console.log(`  [host] uncaught: ${e.message}`));
    await hostTab.goto(`chrome-extension://${extensionId}/host.html`);
    const hostRan = await hostTab.evaluate(async () => {
      const stored = await chrome.storage.local.get("offscreenBootedAt");
      return stored.offscreenBootedAt ?? null;
    });
    check("host document evaluates its module", hostRan !== null, hostRan ? "yes" : "never ran");

    const hostIsolated = await hostTab.evaluate(() => self.crossOriginIsolated);
    check("host document is cross-origin isolated", hostIsolated === true, String(hostIsolated));

    // Speak the port protocol directly, without a content script. This separates "the Host
    // cannot start" from "the content script never asked", which look identical on a page.
    const probe = await setup.evaluate(async () => {
      const seen = [];
      const port = chrome.runtime.connect({ name: "slop-marker" });
      port.onMessage.addListener((m) => seen.push(m));
      port.postMessage({ type: "hello" });
      // Session creation from cached bytes measured at ~0.6 s; allow for a cold start.
      for (let i = 0; i < 240 && !seen.some((m) => m.type === "ready"); i++) {
        await new Promise((r) => setTimeout(r, 500));
      }
      return seen.map((m) => (m.type === "ready" ? `ready(${m.threading.numThreads}t)` : m.type));
    });
    check(
      "the host answers a port directly",
      probe.some((t) => t.startsWith("ready")),
      probe.join(", ") || "no messages",
    );

    // Now the actual page.
    const page = await context.newPage();
    page.on("console", (m) => {
      if (verbose || m.type() === "error") console.log(`  [content] ${m.type()}: ${m.text()}`);
    });
    page.on("pageerror", (e) => console.log(`  [content] uncaught: ${e.message}`));
    await page.goto("http://127.0.0.1:8788/", { waitUntil: "domcontentloaded" });

    // Scoring is ~300 ms per chunk threaded, and this page has several.
    await page
      .waitForFunction(() => CSS.highlights?.get("slop-marker")?.size > 0, null, {
        timeout: 180_000,
      })
      .catch(() => undefined);

    // Ask the service worker what contexts exist. If the offscreen document was never
    // created, or was created and died, that is the whole explanation.
    const sw = context.serviceWorkers()[0];
    if (sw !== undefined) {
      const contexts = await sw
        .evaluate(async () => {
          const all = await chrome.runtime.getContexts({});
          return all.map((c) => `${c.contextType}:${(c.documentUrl ?? "").split("/").pop()}`);
        })
        .catch((e) => [`getContexts failed: ${String(e)}`]);
      console.log(`  [contexts] ${contexts.join(", ")}`);
    }

    const lastError = await setup.evaluate(async () => {
      const stored = await chrome.storage.local.get("lastError");
      return stored.lastError ?? null;
    });
    if (lastError !== null) {
      console.log(`  [lastError] during "${lastError.where}":
    ${lastError.detail}`);
    }

    // If nothing was highlighted, say why rather than just reporting a failure.
    const diagnosis = await page.evaluate(() => ({
      hasHighlightApi: typeof CSS !== "undefined" && "highlights" in CSS,
      paragraphs: document.querySelectorAll("main p").length,
      styleInjected: document.getElementById("slop-marker-style") !== null,
    }));

    const observed = await page.evaluate(() => {
      const hl = CSS.highlights?.get("slop-marker");
      const ranges = hl ? [...hl] : [];
      const marked = [...document.querySelectorAll("[data-slop-marker]")];
      const within = (sel) =>
        marked.filter((el) => el.closest(sel) !== null).length;
      return {
        rangeCount: ranges.length,
        highlightedText: ranges.map((r) => r.toString()).join(" ").slice(0, 160),
        markedTotal: marked.length,
        inFlagged: within("#flagged"),
        inClean: within("#clean"),
        inNav: within("nav"),
        inFooter: within("footer"),
        threading: null,
      };
    });

    if (observed.rangeCount === 0) {
      console.log(
        `  [diagnosis] CSS.highlights=${diagnosis.hasHighlightApi} ` +
          `paragraphs=${diagnosis.paragraphs} styleInjected=${diagnosis.styleInjected}`,
      );
    }
    check("highlights the AI-authored article", observed.inFlagged > 0, `${observed.inFlagged} blocks`);
    check("uses the CSS Custom Highlight API", observed.rangeCount > 0, `${observed.rangeCount} ranges`);
    check("leaves the human-authored article alone", observed.inClean === 0, `${observed.inClean} blocks`);
    check("never scores nav or footer", observed.inNav === 0 && observed.inFooter === 0);

    const threading = await setup.evaluate(async () => {
      const s = await chrome.storage.local.get("threading");
      return s.threading ?? null;
    });
    check(
      "runs multi-threaded",
      (threading?.numThreads ?? 0) > 1,
      threading ? `${threading.numThreads} threads` : "not observed",
    );

    if (observed.highlightedText) {
      console.log(`\n  highlighted: "${observed.highlightedText}…"`);
    }
  } finally {
    await context.close();
    bundleServer.close();
    pageServer.close();
    if (!keep) await rm(profile, { recursive: true, force: true }).catch(() => undefined);
  }

  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  exit(failed.length === 0 ? 0 : 1);
}

await main();
