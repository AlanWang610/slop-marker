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

import { buildPages } from "./pages.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const repo = resolve(root, "..");

const flag = (n) => argv.some((a) => a === `--${n}` || a.startsWith(`--${n}=`));
const headed = flag("headed");
const verbose = flag("verbose");
/** Fetch the bundle from the real release host instead of a local stand-in. */
const realHost = flag("real-host");
/**
 * Install the packaged archive rather than the unpacked directory. The two are meant to be
 * the same extension, and this is the only thing that checks that they are -- `web-ext
 * build` re-reads the tree and could omit a file the directory has.
 */
const useXpi = flag("xpi");

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

const manifestVersion = JSON.parse(readFileSync(join(dist, "manifest.json"), "utf-8")).version;
const packages = join(repo, "artifacts", "packages");
/**
 * Prefer the signed archive. It is the only one release Firefox will install, so it is the
 * artifact worth testing; the unsigned build stands in when signing has not been run.
 */
const signedXpi = join(packages, `slop-marker-${manifestVersion}-signed.xpi`);
const unsignedXpi = join(packages, `slop-marker-${manifestVersion}.xpi`);
const xpi = existsSync(signedXpi) ? signedXpi : unsignedXpi;
if (useXpi && !existsSync(xpi)) {
  console.error(`no packaged archive at ${xpi} — run: node scripts/package.mjs --browser=firefox`);
  exit(2);
}
const addon = useXpi ? xpi : dist;
const addonLabel = useXpi ? (xpi === signedXpi ? " (signed)" : " (packaged, unsigned)") : "";

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

  const { routes, flaggedName, cleanName, flaggedParagraphs } = buildPages();
  const bundleServer = realHost ? null : await serveBundle(8787);
  const pageServer = await servePage(8788, routes);
  const profile = await mkdtemp(join(tmpdir(), "slop-marker-ff-"));

  console.log(`extension  ${addon}${addonLabel}`);
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
    await driver.installAddon(addon, true);

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
        inNav: within("nav"),
        inFooter: within("footer"),
      };
    });

    check("highlights the AI-authored article", observed.inFlagged > 0, `${observed.inFlagged} blocks`);
    check("uses the CSS Custom Highlight API", observed.rangeCount > 0, `${observed.rangeCount} ranges`);
    // On its own page: contentRoot() returns the first <article>, so a second one on the
    // same page is never extracted and the assertion would pass for want of being scored.
    await driver.get("http://127.0.0.1:8788/clean");
    await new Promise((r) => setTimeout(r, 20_000));
    const cleanSeen = await driver.executeScript(function () {
      return {
        marked: document.querySelectorAll("#clean [data-slop-marker]").length,
        blocks: document.querySelectorAll("#clean p").length,
      };
    });
    check(
      "leaves the human-authored article alone",
      cleanSeen.marked === 0 && cleanSeen.blocks > 0,
      `${cleanSeen.marked} of ${cleanSeen.blocks} blocks flagged`,
    );
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

    /* ------------------------------------------------ scope.md 7.1, in full */

    console.log("\n  structured page (every candidate, every exclusion):");
    await driver.switchTo().newWindow("tab");
    await driver.get("http://127.0.0.1:8788/structured");
    await driver
      .wait(
        async () =>
          await driver.executeScript(
            "return document.querySelectorAll('[data-slop-marker]').length > 0;",
          ),
        300_000,
      )
      .catch(() => undefined);

    const marks = await driver.executeScript(function () {
      const marked = (sel) => {
        const el = document.querySelector(sel);
        if (el === null) return -1;
        return (
          (el.hasAttribute("data-slop-marker") ? 1 : 0) +
          el.querySelectorAll("[data-slop-marker]").length
        );
      };
      const ids = [
        "#in-li", "#in-blockquote", "#in-dd", "#in-td", "#nested",
        "#x-pre", "#x-figure", "#disqus_thread", "#x-comments",
        "#x-editable", "#x-aside", "#x-form", "#x-role",
      ];
      const out = {};
      for (const id of ids) out[id] = marked(id);
      return out;
    });

    for (const [id, label] of [
      ["#in-li", "li"],
      ["#in-blockquote", "blockquote"],
      ["#in-dd", "dd"],
      ["#in-td", "td"],
    ]) {
      check(
        `scores a ${label}, which scope.md 7.1 lists as a candidate`,
        marks[id] >= 1,
        `${marks[id]} marked`,
      );
    }
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

    /* --------------------------------------------------- scope.md 7.1, SPA rescan */

    console.log("\n  SPA behaviour:");
    await driver.switchTo().newWindow("tab");
    await driver.get("http://127.0.0.1:8788/spa");
    await new Promise((r) => setTimeout(r, 3000));

    const emptyShell = await driver.executeScript(
      "return document.querySelectorAll('[data-slop-marker]').length;",
    );
    check("an empty shell has nothing to highlight", emptyShell === 0, `${emptyShell} marked`);

    await driver.executeScript(
      "document.querySelector('#root').insertAdjacentHTML('beforeend', arguments[0]);",
      `<article id="added">${flaggedParagraphs}</article>`,
    );
    await driver
      .wait(
        async () =>
          await driver.executeScript(
            "return document.querySelectorAll('#added [data-slop-marker]').length > 0;",
          ),
        300_000,
      )
      .catch(() => undefined);
    const afterMutation = await driver.executeScript(
      "return document.querySelectorAll('#added [data-slop-marker]').length;",
    );
    check("re-scores a subtree an SPA adds after load", afterMutation > 0, `${afterMutation} marked`);

    /* ------------------------------------------- scope.md 9, allowlist, via the UI */

    console.log("\n  allowlist, driven through the options page:");
    await driver.switchTo().window(extensionTab);
    const allowlistBox = await driver.findElement(By.id("allowlist"));
    await allowlistBox.clear();
    await allowlistBox.sendKeys("http://127.0.0.1:8788");
    await driver.findElement(By.id("save-allowlist")).click();
    const saved = await waitForText(driver, By, "allowlist-status", (t) => t.length > 0, 15_000);
    check("the options page saves an excluded origin", saved.startsWith("Saved"), saved || "(blank)");

    await driver.switchTo().newWindow("tab");
    await driver.get("http://127.0.0.1:8788/");
    await new Promise((r) => setTimeout(r, 8000));
    const onBlocked = await driver.executeScript(function () {
      return {
        marked: document.querySelectorAll("[data-slop-marker]").length,
        ranges: (CSS.highlights && CSS.highlights.get("slop-marker")?.size) || 0,
      };
    });
    check(
      "an allowlisted origin is not extracted at all",
      onBlocked.marked === 0 && onBlocked.ranges === 0,
      `${onBlocked.marked} blocks, ${onBlocked.ranges} ranges`,
    );

    // Put it back, so a --keep profile is not left excluding the test origin.
    await driver.switchTo().window(extensionTab);
    await (await driver.findElement(By.id("allowlist"))).clear();
    await driver.findElement(By.id("save-allowlist")).click();
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
