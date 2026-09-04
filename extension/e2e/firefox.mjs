/**
 * The Firefox half of the browser test. Same questions as e2e/run.mjs, different container.
 *
 *   node e2e/firefox.mjs [--headed] [--verbose]
 *
 * Firefox differs from Chrome in the two ways scope.md 6.2 and 6.5 name, and both are what
 * this exercises:
 *
 *   - There is no offscreen document. The event page IS the host, so the ORT session lives
 *     in the same context that routes ports and unloads with the page.
 *   - MV3 grants no host permissions at install, so content scripts do nothing until the
 *     user allows site access. `extensions.originControls.grantByDefault` stands in for
 *     that click, and is the one thing here that is not the real user path.
 *
 * geckodriver rather than Playwright, because Playwright cannot install extensions into
 * Firefox, and geckodriver's temporary-addon install is the mechanism `web-ext run` uses --
 * so it works on release Firefox without disabling signature checks.
 *
 * THE CONSTRAINT THAT SHAPES THIS FILE: geckodriver refuses both `driver.get()` and script
 * execution on privileged pages ("ExecuteScript ... not supported for privileged browsing
 * contexts"). It does allow findElement, getText and click. So extension pages are driven
 * through their real UI -- click the button, read the status line -- and only the ordinary
 * http test page gets executeScript. That turns out to be the better test: it exercises the
 * first-run and options pages a user actually sees, rather than poking internals.
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
const headed = flag("headed");
const verbose = flag("verbose");
/** Fetch the bundle from the real release host instead of a local stand-in. */
const realHost = flag("real-host");

const dist = join(root, "dist", "firefox");
const calibration = JSON.parse(readFileSync(join(dist, "assets", "calibration.json"), "utf-8"));
const VERSION = calibration.version;
const bundleDir = join(repo, "artifacts", "bundles", VERSION);

if (!existsSync(join(dist, "manifest.json"))) {
  console.error(
    "no build at dist/firefox — run:\n" +
      "  node scripts/build.mjs --browser=firefox --model-base-url=http://127.0.0.1:8787",
  );
  exit(2);
}
if (!existsSync(join(bundleDir, "model.onnx"))) {
  console.error(`no local bundle at ${bundleDir}`);
  exit(2);
}

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
        "access-control-allow-origin": "*",
      });
      createReadStream(file).pipe(res);
    });
    server.listen(port, "127.0.0.1", () => ready(server));
  });
}

function servePage(port, html) {
  return new Promise((ready) => {
    const server = createServer((_req, res) => {
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" }).end(html);
    });
    server.listen(port, "127.0.0.1", () => ready(server));
  });
}

/** The same page the Chrome run uses, built from fixtures/documents.json. */
function buildPage() {
  const fx = JSON.parse(readFileSync(join(repo, "fixtures", "documents.json"), "utf-8"));
  const flagged = fx.documents.find((d) => d.expected.runs.some((r) => r.flagged));
  const clean = fx.documents.find((d) => !d.expected.runs.some((r) => r.flagged));

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

const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok, detail });
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
};

/** Poll an element's text until it satisfies `predicate`, without executing any script. */
async function waitForText(driver, By, id, predicate, timeoutMs) {
  const end = Date.now() + timeoutMs;
  let last = "";
  while (Date.now() < end) {
    try {
      last = await driver.findElement(By.id(id)).getText();
      if (predicate(last)) return last;
    } catch {
      /* not rendered yet */
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
  return last;
}

async function main() {
  const { Builder, By } = await import("selenium-webdriver");
  const firefox = (await import("selenium-webdriver/firefox.js")).default;
  const geckodriver = await import("geckodriver");

  const { html, flaggedName, cleanName } = buildPage();
  const bundleServer = realHost ? null : await serveBundle(8787);
  const pageServer = await servePage(8788, html);
  const profile = await mkdtemp(join(tmpdir(), "slop-marker-ff-"));

  console.log(`extension  ${dist}`);
  console.log(`model      ${VERSION} from ${realHost ? "the real release host" : "http://127.0.0.1:8787"}`);
  console.log(`page       flagged=${flaggedName} clean=${cleanName}\n`);

  // start() resolves to the child process; the module exposes no stop().
  const gecko = await geckodriver.start({ port: 4444, log: verbose ? "debug" : "fatal" });
  await new Promise((r) => setTimeout(r, 1500));

  const options = new firefox.Options();
  if (!headed) options.addArguments("-headless");
  // Firefox MV3 withholds host permissions until the user grants site access (scope.md 6.5).
  options.setPreference("extensions.originControls.grantByDefault", true);
  options.setPreference("browser.shell.checkDefaultBrowser", false);

  const driver = await new Builder()
    .forBrowser("firefox")
    .setFirefoxOptions(options)
    .usingServer("http://127.0.0.1:4444")
    .build();

  try {
    await driver.installAddon(dist, true);

    /**
     * Firefox assigns each extension a per-profile moz-extension:// origin and ignores a
     * pre-seeded `extensions.webextensions.uuids` for a temporary add-on. WebDriver exposes
     * no lookup, and reading the pref needs chrome context, which Firefox gates behind a
     * command-line flag that cannot be set through capabilities.
     *
     * So take it from the extension: runtime.onInstalled opens first-run.html, which is a
     * real behaviour worth asserting anyway, and parking on that tab sidesteps the fact
     * that the driver may not navigate to extension URLs at all.
     */
    let extensionTab = null;
    let base = null;
    const deadline = Date.now() + 60_000;
    while (base === null && Date.now() < deadline) {
      for (const handle of await driver.getAllWindowHandles()) {
        await driver.switchTo().window(handle);
        const url = await driver.getCurrentUrl();
        // Not new URL(url).origin: moz-extension is not a "special" scheme, so WHATWG URL
        // returns the string "null" for it.
        const match = /^(moz-extension:\/\/[^/]+)/.exec(url);
        if (match !== null) {
          base = match[1];
          extensionTab = handle;
          break;
        }
      }
      if (base === null) await new Promise((r) => setTimeout(r, 500));
    }
    check("installs and opens its first-run page", base !== null, base ?? "no extension tab");
    if (base === null) throw new Error("the extension never opened a page");

    check("first-run page renders", (await driver.getTitle()).includes("slop-marker"));

    // Firefox is the browser where this matters: without site access nothing is ever
    // scored, and the page has to say so.
    const permission = await driver.findElement(By.id("permission-status")).getText();
    check("reports site access as granted", permission.includes("Granted"), permission || "(blank)");

    // Drive the real button. On Firefox this is the whole download path: the event page
    // fetches, verifies against the shipped SHA256SUMS and writes to the Cache API.
    await driver.findElement(By.id("download")).click();
    const status = await waitForText(
      driver,
      By,
      "model-status",
      (t) => t.startsWith("Ready") || t.startsWith("Failed"),
      300_000,
    );
    check("model downloads and verifies", status.startsWith("Ready"), status);

    // The page. In a new tab, because the driver cannot navigate the privileged one.
    await driver.switchTo().newWindow("tab");
    await driver.get("http://127.0.0.1:8788/");
    await driver
      .wait(
        async () =>
          await driver.executeScript(
            "return (CSS.highlights && CSS.highlights.get('slop-marker')?.size) > 0;",
          ),
        300_000,
      )
      .catch(() => undefined);

    const observed = await driver.executeScript(function () {
      const hl = CSS.highlights && CSS.highlights.get("slop-marker");
      const ranges = hl ? Array.from(hl) : [];
      const marked = Array.from(document.querySelectorAll("[data-slop-marker]"));
      const within = (sel) => marked.filter((el) => el.closest(sel) !== null).length;
      return {
        rangeCount: ranges.length,
        highlightedText: ranges
          .map((r) => r.toString())
          .join(" ")
          .slice(0, 140),
        inFlagged: within("#flagged"),
        inClean: within("#clean"),
        inNav: within("nav"),
        inFooter: within("footer"),
      };
    });

    check("highlights the AI-authored article", observed.inFlagged > 0, `${observed.inFlagged} blocks`);
    check("uses the CSS Custom Highlight API", observed.rangeCount > 0, `${observed.rangeCount} ranges`);
    check("leaves the human-authored article alone", observed.inClean === 0, `${observed.inClean} blocks`);
    check("never scores nav or footer", observed.inNav === 0 && observed.inFooter === 0);

    // Options page, reached by clicking the link rather than navigating, for the same
    // privileged-context reason. scope.md 6.3 asks that the observed mode be shown.
    await driver.switchTo().window(extensionTab);
    await driver.findElement(By.id("open-options")).click();
    const threading = await waitForText(
      driver,
      By,
      "threading",
      (t) => t.length > 0 && t !== "—" && t !== "not yet observed",
      60_000,
    );
    check("options page shows the observed threading mode", threading.includes("thread"), threading);

    const modelState = await driver.findElement(By.id("model-state")).getText();
    check("options page shows the model as ready", modelState === "ready", modelState);

    if (observed.highlightedText) {
      console.log(`\n  highlighted: "${observed.highlightedText}…"`);
    }
  } catch (err) {
    console.log(`\n  ERROR: ${err?.message ?? String(err)}`);
    results.push({ name: "run completed", ok: false, detail: String(err?.message ?? err) });
  } finally {
    await driver.quit().catch(() => undefined);
    gecko?.kill?.();
    bundleServer?.close();
    pageServer.close();
    await rm(profile, { recursive: true, force: true }).catch(() => undefined);
  }

  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  exit(failed.length === 0 ? 0 : 1);
}

await main();
