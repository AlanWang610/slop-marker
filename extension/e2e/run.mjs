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
import { Transform } from "node:stream";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { argv, exit } from "node:process";

import { buildPages } from "./pages.mjs";

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
/**
 * Fetch the bundle from the real release host instead of a local stand-in. This is the
 * only way to test the COEP interaction for real: GitHub release assets carry no
 * Access-Control-Allow-Origin, which is precisely why the download runs in the service
 * worker rather than the cross-origin-isolated offscreen document. A local server sending
 * `access-control-allow-origin: *` would let a regression through unnoticed.
 */
const realHost = flag("real-host");
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

/**
 * Flipped by the corrupt-download scenario. One byte is enough: the checksum is computed
 * over the whole file, so this reproduces a truncated or tampered upload exactly.
 */
const bundleState = { corrupt: false };

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
      if (!bundleState.corrupt) {
        createReadStream(file).pipe(res);
        return;
      }
      let flipped = false;
      const flip = new Transform({
        transform(chunk, _encoding, done) {
          if (!flipped && chunk.length > 0) {
            chunk[0] ^= 0xff;
            flipped = true;
          }
          done(null, chunk);
        },
      });
      createReadStream(file).pipe(flip).pipe(res);
    });
    server.listen(port, "127.0.0.1", () => ready(server));
  });
}

/** Serves the fixture pages. A real http origin, so the content script matches <all_urls>. */
function servePage(port, routes) {
  return new Promise((ready) => {
    const server = createServer((req, res) => {
      const path = new URL(req.url, "http://x").pathname;
      const html = routes[path];
      if (html === undefined) {
        res.writeHead(404).end("no such page");
        return;
      }
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" }).end(html);
    });
    server.listen(port, "127.0.0.1", () => ready(server));
  });
}

/* ---------------------------------------------------------------- the run */

/** Resolve `promise`, or `fallback` if it has not settled in time. */
function withTimeout(promise, ms, fallback) {
  return Promise.race([
    promise.catch(() => fallback),
    new Promise((r) => setTimeout(() => r(fallback), ms)),
  ]);
}

/** Wait for the first highlight, or give up quietly and let the assertion report it. */
async function waitForHighlights(page, timeout = 180_000) {
  await page
    .waitForFunction(() => CSS.highlights?.get("slop-marker")?.size > 0, null, { timeout })
    .catch(() => undefined);
}

/**
 * Measure what a reader actually sees on a flagged block, in the page.
 *
 * scope.md 9 asks for de-emphasis, "never hide", and the two halves of that pull against
 * each other: too little mixing and the cue is invisible, too much and the text is
 * unreadable. Nothing asserted either half until now -- the harness only ever checked that
 * `--slop-marker-bg` resolved to the right colour, which says nothing about the result.
 *
 * `::highlight()` styles cannot be read back with getComputedStyle, so this resolves the
 * same `color-mix()` the stylesheet uses on a probe span inside the block, then computes
 * WCAG relative-luminance contrast against the effective background. Both the mixed colour
 * and the block's own colour are reported, so a caller can assert the text is still legible
 * *and* genuinely quieter than its neighbours.
 */
const CONTRAST_PROBE = `(sel) => {
  const block = document.querySelector(sel);
  if (block === null) return null;

  /**
   * Paint a colour on a 1x1 canvas and read the pixel back.
   *
   * Not getComputedStyle: Chrome serialises a computed \`color-mix(in oklab, ...)\` as
   * \`oklab(L a b)\`, so scraping rgb() out of it silently yields nothing. The canvas gives
   * exact sRGB bytes whatever the serialisation. A rejected colour leaves fillStyle at the
   * sentinel, which is how an unsupported syntax is detected rather than mistaken for
   * magenta.
   */
  const paint = (css) => {
    const canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    const ctx = canvas.getContext("2d");
    const sentinel = "#ff00ff";
    ctx.fillStyle = sentinel;
    ctx.fillStyle = css;
    if (ctx.fillStyle === sentinel && css.replace(/\\s/g, "").toLowerCase() !== sentinel) {
      return null;
    }
    ctx.fillRect(0, 0, 1, 1);
    const data = ctx.getImageData(0, 0, 1, 1).data;
    return [data[0], data[1], data[2]];
  };

  /** Resolve a colour keyword the canvas will not take (Canvas, currentColor) via layout. */
  const resolveBackground = (css) => {
    const probe = document.createElement("span");
    probe.style.backgroundColor = css;
    block.appendChild(probe);
    const value = getComputedStyle(probe).backgroundColor;
    probe.remove();
    return value;
  };

  // WCAG 2.x relative luminance, on sRGB channels.
  const luminance = (channels) => {
    const linear = channels.map((c) => {
      const v = c / 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
  };

  const ratio = (a, b) => {
    if (a === null || b === null) return null;
    const la = luminance(a);
    const lb = luminance(b);
    const [hi, lo] = la > lb ? [la, lb] : [lb, la];
    return Math.round(((hi + 0.05) / (lo + 0.05)) * 100) / 100;
  };

  const declared = block.style.getPropertyValue("--slop-marker-bg") || "Canvas";
  const backgroundCss = resolveBackground(declared);
  const baseCss = getComputedStyle(block).color;

  const background = paint(backgroundCss);
  const base = paint(baseCss);
  // The same mix the stylesheet applies, with currentColor and the ground already resolved
  // so the canvas can take it.
  const mixed = paint(\`color-mix(in oklab, \${baseCss} 45%, \${backgroundCss})\`);

  return {
    background,
    base,
    mixed,
    mixedContrast: ratio(mixed, background),
    baseContrast: ratio(base, background),
  };
}`;

/** Self-or-descendant count of decorated blocks under a selector, evaluated in the page. */
const MARKED_COUNTER = `(sel) => {
  const el = document.querySelector(sel);
  if (el === null) return -1;
  return (el.hasAttribute("data-slop-marker") ? 1 : 0) +
    el.querySelectorAll("[data-slop-marker]").length;
}`;

const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok, detail });
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
};

async function main() {
  const { chromium } = await import("playwright");
  const { routes, flaggedName, cleanName, flaggedParagraphs, realPages } = buildPages();

  const bundleServer = realHost ? null : await serveBundle(8787);
  const pageServer = await servePage(8788, routes);
  const profile = await mkdtemp(join(tmpdir(), "slop-marker-e2e-"));

  console.log(`extension  ${dist}`);
  console.log(`model      ${VERSION} from ${realHost ? "the real release host" : "http://127.0.0.1:8787"}`);
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
    context.on("weberror", (e) => {
      // Saved real pages raise these by design: their scripts are blocked, so inline code
      // referencing jQuery or WordPress globals throws. Only interesting when asked for.
      const message = e.error().message;
      const expected = /(\$|wp|jQuery|dataLayer|gtag) is not defined/.test(message);
      if (verbose || !expected) console.log(`  [page error] ${message}`);
    });
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

    /**
     * Fail closed first. One byte of model.onnx is flipped on the wire, which is what a
     * truncated or tampered upload looks like from here. Nothing may reach the Cache API,
     * because a half-trusted model would then be loaded on every later run with nothing to
     * point at. Skipped against --real-host, where there is no server of ours to corrupt.
     */
    if (!realHost) {
      bundleState.corrupt = true;
      const bad = await setup.evaluate(
        async () => await chrome.runtime.sendMessage({ type: "downloadModel" }),
      );
      check(
        "a corrupted bundle is refused",
        bad?.ok === false && /checksum mismatch/.test(bad?.message ?? ""),
        bad?.message ?? "accepted it",
      );

      const afterBad = await setup.evaluate(
        async () => await chrome.runtime.sendMessage({ type: "getStatus" }),
      );
      check(
        "nothing is cached after a refused download",
        afterBad?.modelState?.phase === "error",
        afterBad?.modelState?.phase ?? "?",
      );
      bundleState.corrupt = false;
    }

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
     * Headless Chromium creates an offscreen document but never executes its scripts.
     * Verified directly: a trivial classic script in host.html does not run, while
     * chrome.runtime.getContexts() happily reports the document exists. That is a headless
     * limitation, not a defect in the extension.
     *
     * Headed, the offscreen document is real and this asserts it. Headless, host.html is
     * opened as an ordinary tab instead -- the same document running the same code under
     * the same manifest COOP/COEP, registering the same onConnect listener, reached by the
     * same service-worker relay. Only the ~40-line offscreen wrapper differs, and `--headed`
     * is what covers that.
     */
    let hostTab = null;
    if (!headed) {
      hostTab = await context.newPage();
      hostTab.on("console", (m) => {
        if (verbose || m.type() === "error") console.log(`  [host] ${m.type()}: ${m.text()}`);
      });
      hostTab.on("pageerror", (e) => console.log(`  [host] uncaught: ${e.message}`));
      await hostTab.goto(`chrome-extension://${extensionId}/host.html`);
    }

    if (hostTab !== null) {
      const hostIsolated = await hostTab.evaluate(() => self.crossOriginIsolated);
      check("host document is cross-origin isolated", hostIsolated === true, String(hostIsolated));
    }

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

    /**
     * Checked *after* the port probe on purpose: the offscreen document is created lazily,
     * on the first content-script connection, so before that there is nothing to ask.
     *
     * The marker is the module's first statement, so its presence means offscreen.js
     * evaluated rather than throwing partway. It threw for a long time -- chrome.storage is
     * undefined in an offscreen document and that was line one -- and the tab stand-in hid
     * it, because an ordinary extension page does have chrome.storage.
     */
    const hostRan = await setup.evaluate(async () => {
      const deadline = Date.now() + 30_000;
      while (Date.now() < deadline) {
        const stored = await chrome.storage.local.get("offscreenBootedAt");
        if (stored.offscreenBootedAt) return stored.offscreenBootedAt;
        await new Promise((r) => setTimeout(r, 500));
      }
      return null;
    });
    check(
      headed ? "offscreen document evaluates its module" : "host document evaluates its module",
      hostRan !== null,
      hostRan ? (headed ? "real offscreen document" : "tab stand-in") : "never ran",
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

    /* ------------------------------------------------ scope.md 7.1, in full */

    console.log("\n  structured page (every candidate, every exclusion):");
    const structured = await context.newPage();
    structured.on("pageerror", (e) => console.log(`  [structured] uncaught: ${e.message}`));
    await structured.goto("http://127.0.0.1:8788/structured", { waitUntil: "domcontentloaded" });
    await waitForHighlights(structured);

    const marks = await structured.evaluate((counter) => {
      const marked = new Function(`return ${counter}`)();
      const ids = [
        "#in-li", "#in-blockquote", "#in-dd", "#in-td", "#nested",
        "#x-pre", "#x-figure", "#disqus_thread", "#x-comments",
        "#x-editable", "#x-aside", "#x-form", "#x-role",
      ];
      return Object.fromEntries(ids.map((id) => [id, marked(id)]));
    }, MARKED_COUNTER);

    for (const [id, label] of [
      ["#in-li", "li"],
      ["#in-blockquote", "blockquote"],
      ["#in-dd", "dd"],
      ["#in-td", "td"],
    ]) {
      // td is the regression: `table` was in the exclusion list, and since exclusion is a
      // closest() test every cell matched its own ancestor. No td had ever been scored.
      check(
        `scores a ${label}, which scope.md 7.1 lists as a candidate`,
        marks[id] >= 1,
        `${marks[id]} marked`,
      );
    }

    // Before the fix the nested-candidate guard was dead code -- closest() matches the
    // element itself -- so the li and its p were both blocks over the same text.
    check(
      "scores nested candidates once, not twice",
      marks["#nested"] === 1,
      `${marks["#nested"]} marked`,
    );

    for (const [id, label] of [
      ["#x-pre", "pre/code"],
      ["#x-figure", "figure/figcaption"],
      ["#disqus_thread", "a Disqus thread"],
      ["#x-comments", "a comment list"],
      ["#x-editable", "editable text"],
      ["#x-aside", "an aside"],
      ["#x-form", "a form"],
      ["#x-role", "[role=complementary]"],
    ]) {
      check(
        `never scores ${label}, even carrying the same text`,
        marks[id] === 0,
        `${marks[id]} marked`,
      );
    }

    /* ------------------------------------------------------ scope.md 9, themes */

    console.log("\n  dark and tinted grounds:");
    const page_ = page; // the article page, aliased so the loop below can shadow `page`
    const dark = await context.newPage();
    await dark.goto("http://127.0.0.1:8788/dark", { waitUntil: "domcontentloaded" });
    await waitForHighlights(dark);

    const grounds = await dark.evaluate(() => {
      const read = (sel) => {
        const el = document.querySelector(sel);
        return el === null ? null : el.style.getPropertyValue("--slop-marker-bg");
      };
      return {
        body: read("#dark-body [data-slop-marker]"),
        callout: read("#callout [data-slop-marker]"),
      };
    });
    check(
      "mixes against the dark page background, not white",
      grounds.body === "rgb(17, 17, 17)",
      grounds.body ?? "nothing marked",
    );
    check(
      "mixes a tinted callout against its own background",
      grounds.callout === "rgb(40, 30, 60)",
      grounds.callout ?? "nothing marked",
    );

    /**
     * The half of scope.md 9 no assertion covered: "de-emphasize, never hide". 3.0 is
     * WCAG AA for large text -- a defensible floor for prose that is meant to recede but
     * still be readable. The upper bound is the block's own text: if the mix were not
     * quieter than its neighbours there would be no cue at all.
     */
    const LEGIBLE = 3.0;
    for (const [page, sel, label] of [
      [page_, "#flagged [data-slop-marker]", "a light page"],
      [dark, "#dark-body [data-slop-marker]", "a dark page"],
      [dark, "#callout [data-slop-marker]", "a tinted callout"],
    ]) {
      const seen = await page.evaluate(
        ([probe, s]) => new Function(`return ${probe}`)()(s),
        [CONTRAST_PROBE, sel],
      );
      if (seen === null) {
        check(`de-emphasized text stays readable on ${label}`, false, "nothing marked");
        continue;
      }
      const measured = seen.mixedContrast !== null && seen.baseContrast !== null;
      check(
        `de-emphasized text stays readable on ${label}`,
        measured && seen.mixedContrast >= LEGIBLE,
        measured ? `contrast ${seen.mixedContrast}:1 (floor ${LEGIBLE})` : "could not measure",
      );
      check(
        `de-emphasized text is quieter than its neighbours on ${label}`,
        measured && seen.mixedContrast < seen.baseContrast,
        measured
          ? `${seen.mixedContrast}:1 against ${seen.baseContrast}:1 for ordinary text`
          : "could not measure",
      );
    }

    /* --------------------------------------------------- scope.md 7.1, SPA rescan */

    console.log("\n  SPA behaviour:");
    const spa = await context.newPage();
    spa.on("pageerror", (e) => console.log(`  [spa] uncaught: ${e.message}`));
    await spa.goto("http://127.0.0.1:8788/spa", { waitUntil: "domcontentloaded" });
    await new Promise((r) => setTimeout(r, 2000));

    const before = await spa.evaluate(
      () => document.querySelectorAll("[data-slop-marker]").length,
    );
    check("an empty shell has nothing to highlight", before === 0, `${before} marked`);

    // The page had no scoreable prose at document_idle, which is exactly the case the
    // MutationObserver exists for -- and the case where the content script used to return
    // before installing it.
    await spa.evaluate((html) => {
      document
        .querySelector("#root")
        .insertAdjacentHTML("beforeend", `<article id="added">${html}</article>`);
    }, flaggedParagraphs);
    await waitForHighlights(spa);

    const afterMutation = await spa.evaluate(
      () => document.querySelectorAll("#added [data-slop-marker]").length,
    );
    check("re-scores a subtree an SPA adds after load", afterMutation > 0, `${afterMutation} marked`);

    /**
     * An infinite feed. Each batch must be scored, and -- the part a single append cannot
     * show -- the batches before it must still be highlighted afterwards. Rendering
     * replaces the whole highlight set on every paint, so a run that stopped being
     * recomputed would silently disappear from the top of the page as the reader scrolled.
     */
    const feed = await context.newPage();
    feed.on("pageerror", (e) => console.log(`  [feed] uncaught: ${e.message}`));
    await feed.goto("http://127.0.0.1:8788/spa", { waitUntil: "domcontentloaded" });
    await new Promise((r) => setTimeout(r, 2000));

    let feedOk = true;
    let feedDetail = "";
    for (let batch = 1; batch <= 3; batch++) {
      await feed.evaluate(
        ([html, id]) => {
          document
            .querySelector("#root")
            .insertAdjacentHTML("beforeend", `<article id="${id}">${html}</article>`);
        },
        [flaggedParagraphs, `batch-${batch}`],
      );
      await feed
        .waitForFunction(
          (id) => document.querySelectorAll(`#${id} [data-slop-marker]`).length > 0,
          `batch-${batch}`,
          { timeout: 120_000 },
        )
        .catch(() => undefined);

      const still = await feed.evaluate(
        (n) => {
          const out = [];
          for (let i = 1; i <= n; i++) {
            out.push(document.querySelectorAll(`#batch-${i} [data-slop-marker]`).length);
          }
          return out;
        },
        batch,
      );
      feedDetail = still.join("/");
      if (still.some((count) => count === 0)) {
        feedOk = false;
        break;
      }
    }
    check(
      "an infinite feed keeps scoring as it grows, without dropping earlier runs",
      feedOk,
      `blocks marked per batch: ${feedDetail}`,
    );
    await feed.close();

    /* -------------------------------------------------- scope.md 11, score cache */

    const cached = await setup.evaluate(
      async () => await chrome.runtime.sendMessage({ type: "getStatus" }),
    );
    check(
      "scored chunks are cached for the next visit",
      (cached?.cachedScores ?? 0) > 0,
      `${cached?.cachedScores ?? 0} entries`,
    );

    /* ------------------------------------- scope.md 9, threshold override, live */

    console.log("\n  options-page controls:");
    const flaggedBefore = await page.evaluate(
      () => document.querySelectorAll("[data-slop-marker]").length,
    );
    await setup.evaluate(async () => {
      await chrome.storage.local.set({ thresholdOverride: 0.999 });
    });
    await page
      .waitForFunction(
        (n) => document.querySelectorAll("[data-slop-marker]").length < n,
        flaggedBefore,
        { timeout: 15_000 },
      )
      .catch(() => undefined);
    const flaggedStrict = await page.evaluate(
      () => document.querySelectorAll("[data-slop-marker]").length,
    );
    check(
      "a stricter threshold repaints open tabs without a reload",
      flaggedStrict < flaggedBefore,
      `${flaggedBefore} -> ${flaggedStrict} blocks`,
    );

    await setup.evaluate(async () => {
      await chrome.storage.local.remove("thresholdOverride");
    });
    await page
      .waitForFunction(
        (n) => document.querySelectorAll("[data-slop-marker]").length === n,
        flaggedBefore,
        { timeout: 15_000 },
      )
      .catch(() => undefined);
    const flaggedRestored = await page.evaluate(
      () => document.querySelectorAll("[data-slop-marker]").length,
    );
    check(
      "clearing the override restores the shipped operating point",
      flaggedRestored === flaggedBefore,
      `${flaggedRestored} of ${flaggedBefore} blocks`,
    );

    /* --------------------------------------------------- scope.md 9, allowlist */

    await setup.evaluate(async () => {
      await chrome.storage.local.set({ allowlist: ["http://127.0.0.1:8788"] });
    });
    const blocked = await context.newPage();
    await blocked.goto("http://127.0.0.1:8788/", { waitUntil: "domcontentloaded" });
    await new Promise((r) => setTimeout(r, 5000));
    const onBlocked = await blocked.evaluate(() => ({
      marked: document.querySelectorAll("[data-slop-marker]").length,
      ranges: CSS.highlights?.get("slop-marker")?.size ?? 0,
    }));
    check(
      "an allowlisted origin is not extracted at all",
      onBlocked.marked === 0 && onBlocked.ranges === 0,
      `${onBlocked.marked} blocks, ${onBlocked.ranges} ranges`,
    );
    await blocked.close();
    await setup.evaluate(async () => {
      await chrome.storage.local.remove("allowlist");
    });

    /* -------------------------------------------- real pages, the false-positive side */

    if (realPages.length === 0) {
      console.log("\n  no saved corpus — run: node e2e/real/fetch.mjs");
    } else {
      console.log(`\n  ${realPages.length} saved real pages (human-written, expect no flags):`);

      /**
       * Wait for scoring to settle. There is no "done" signal, so watch the score cache and
       * stop once it has not grown for a few seconds -- which also reports how many chunks
       * the page produced, and that is the assertion that keeps "no false positives" from
       * being vacuous. A page whose prose extraction missed entirely would trivially have
       * none.
       */
      const settle = async () => {
        let last = -1;
        let stable = 0;
        for (let i = 0; i < 90 && stable < 5; i++) {
          await new Promise((r) => setTimeout(r, 1000));
          const status = await setup.evaluate(
            async () => await chrome.runtime.sendMessage({ type: "getStatus" }),
          );
          const now = status?.cachedScores ?? 0;
          stable = now === last ? stable + 1 : 0;
          last = now;
        }
        return last;
      };

      for (const saved of realPages) {
        const before = await setup.evaluate(
          async () => await chrome.runtime.sendMessage({ type: "getStatus" }),
        );
        const real = await context.newPage();
        // Blocking the page's own scripts guarantees ReferenceErrors from whatever inline
        // code expected them (jQuery, WordPress globals). Expected, and not ours.
        real.on("pageerror", (e) => {
          if (verbose) console.log(`  [${saved.name}] uncaught: ${e.message}`);
        });

        // Nothing but our own server. These pages reference the live site's CSS, fonts,
        // analytics and images; letting them load would make the run depend on the network
        // and quietly re-fetch from nasa.gov on every invocation.
        await real.route("**", (route) => {
          const url = route.request().url();
          if (url.startsWith("http://127.0.0.1:")) return route.continue();
          return route.abort();
        });

        await real
          .goto(`http://127.0.0.1:8788/real/${saved.name}`, { waitUntil: "domcontentloaded" })
          .catch(() => undefined);

        const scored = await settle();
        const chunks = scored - (before?.cachedScores ?? 0);
        const marked = await real.evaluate(() => ({
          blocks: document.querySelectorAll("[data-slop-marker]").length,
          ranges: CSS.highlights?.get("slop-marker")?.size ?? 0,
          text: [...(CSS.highlights?.get("slop-marker") ?? [])]
            .map((r) => r.toString())
            .join(" ")
            .slice(0, 200),
        }));

        check(
          `${saved.name}: extraction finds prose to score`,
          chunks > 0,
          `${chunks} chunks scored`,
        );
        check(
          `${saved.name}: no false positives on human-written text`,
          marked.blocks === 0 && marked.ranges === 0,
          marked.blocks === 0
            ? "clean"
            : `${marked.blocks} blocks flagged — "${marked.text}…"`,
        );
        await real.close();
      }
    }

    /* ------------------------------------------------ scope.md 7.4, port reconnect */

    console.log("\n  port reconnect:");
    const reconnect = await context.newPage();
    reconnect.on("pageerror", (e) => console.log(`  [reconnect] uncaught: ${e.message}`));
    reconnect.on("console", (m) => {
      if (verbose || m.type() === "error" || m.type() === "warning") {
        console.log(`  [reconnect] ${m.type()}: ${m.text()}`);
      }
    });
    await reconnect.goto("http://127.0.0.1:8788/spa", { waitUntil: "domcontentloaded" });
    await new Promise((r) => setTimeout(r, 2000));

    // Stop the service worker under the content script. Its port disconnects, and the
    // content script has to reconnect and re-ask -- the same code path a Firefox event-page
    // unload takes (scope.md 7.4).
    let killed = null;
    try {
      const cdp = await context.newCDPSession(reconnect);
      const { targetInfos } = await cdp.send("Target.getTargets");
      const worker = targetInfos.find((t) => t.type === "service_worker");
      if (worker !== undefined) {
        await cdp.send("Target.closeTarget", { targetId: worker.targetId });
        killed = worker.url.split("/").pop();
      }
      await cdp.detach().catch(() => undefined);
    } catch (err) {
      console.log(`  [reconnect] could not stop the service worker: ${String(err)}`);
    }
    check(
      "the service worker can be stopped mid-session",
      killed !== null,
      killed ?? "no service_worker target",
    );

    if (killed !== null) {
      // Give the content script's onDisconnect its 500 ms reconnect delay, and Chrome a
      // moment to notice the worker is gone.
      await new Promise((r) => setTimeout(r, 3000));

      await reconnect.evaluate((html) => {
        document
          .querySelector("#root")
          .insertAdjacentHTML("beforeend", `<article id="added">${html}</article>`);
      }, flaggedParagraphs);
      await waitForHighlights(reconnect, 120_000);

      const afterKill = await reconnect.evaluate(
        () => document.querySelectorAll("#added [data-slop-marker]").length,
      );
      const ok = afterKill > 0;
      if (!ok) {
        // Say why, rather than reporting a bare zero. The interesting distinction is
        // whether the worker came back at all, and whether the content script is still
        // attached to a live extension context.
        const workers = context.serviceWorkers().map((w) => w.url().split("/").pop());

        // Through `setup`, not through a serviceWorker handle: a handle for a worker that
        // was closed never settles, and this diagnostic then hangs the whole run.
        const contexts = await withTimeout(
          setup.evaluate(async () => {
            const all = await chrome.runtime.getContexts({});
            return all.map((c) => `${c.contextType}:${(c.documentUrl ?? "").split("/").pop()}`);
          }),
          10_000,
          ["getContexts timed out"],
        );
        console.log(`  [reconnect] contexts after restart: ${contexts.join(", ")}`);
        const state = await reconnect.evaluate(() => ({
          added: document.querySelectorAll("#added p").length,
          styled: document.getElementById("slop-marker-style") !== null,
        }));
        const err = await withTimeout(
          setup.evaluate(
            async () => (await chrome.storage.local.get("lastError")).lastError ?? null,
          ),
          10_000,
          null,
        );
        console.log(
          `  [reconnect] workers=[${workers.join(", ")}] addedParagraphs=${state.added} ` +
            `styleInjected=${state.styled}` +
            (err ? ` lastError=${err.where}: ${String(err.detail).slice(0, 120)}` : ""),
        );
      }
      check(
        "the content script reconnects and scoring resumes",
        ok,
        `${afterKill} marked after the worker was stopped`,
      );
    }
  } finally {
    await context.close();
    bundleServer?.close();
    pageServer.close();
    if (!keep) await rm(profile, { recursive: true, force: true }).catch(() => undefined);
  }

  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  exit(failed.length === 0 ? 0 : 1);
}

await main();
