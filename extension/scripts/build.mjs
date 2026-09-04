/**
 * One source tree, two builds (scope.md 13).
 *
 *   node scripts/build.mjs --browser=chrome   [--watch] [--dev]
 *   node scripts/build.mjs --browser=firefox
 *
 * The per-browser work is deliberately small: pick the router entry point, and strip the
 * manifest keys the other browser's validator would reject. Everything else is shared.
 *
 * Chrome ignores `background.scripts` and Firefox ignores `background.service_worker`, so
 * in principle one manifest serves both. In practice the store validators are stricter than
 * the runtimes, so each build gets only the keys its target understands.
 */

import { build, context } from "esbuild";
import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { argv, exit } from "node:process";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const src = join(root, "src");

const flag = (name) => argv.some((a) => a === `--${name}` || a.startsWith(`--${name}=`));
const value = (name, fallback) => {
  const hit = argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.split("=").slice(1).join("=") : fallback;
};

const browser = value("browser", "chrome");
// Empty means "use the host baked into bundle-config.ts by tools/sync_extension_assets.py".
const modelBaseUrl = value("model-base-url", "");
const dev = flag("dev");
const watch = flag("watch");

if (!["chrome", "firefox"].includes(browser)) {
  console.error(`--browser must be chrome or firefox, got ${browser}`);
  exit(1);
}

const outdir = join(root, "dist", browser);

/**
 * Entry points. Each is bundled independently: extension contexts do not share a module
 * graph, and a content script cannot be an ES module in either browser.
 */
const ENTRIES = {
  content: { in: join(src, "content", "index.ts"), format: "iife" },
  worker: { in: join(src, "worker", "worker.ts"), format: "esm" },
  offscreen: { in: join(src, "host", "offscreen.ts"), format: "esm" },
  "router-chrome": { in: join(src, "router", "chrome.ts"), format: "esm" },
  "router-firefox": { in: join(src, "router", "firefox.ts"), format: "iife" },
  options: { in: join(src, "pages", "options.ts"), format: "esm" },
  "first-run": { in: join(src, "pages", "first-run.ts"), format: "esm" },
};

// Chrome runs the router as a module service worker; Firefox's event page loads
// background.scripts as classic scripts, so its router must be an IIFE.
const SKIP = {
  chrome: ["router-firefox"],
  firefox: ["router-chrome", "offscreen"],
};

async function buildManifest() {
  const manifest = JSON.parse(await readFile(join(src, "assets", "manifest.json"), "utf-8"));

  if (browser === "chrome") {
    delete manifest.background.scripts;
    delete manifest.browser_specific_settings;
  } else {
    delete manifest.background.service_worker;
    delete manifest.background.type;
    // Unknown to Firefox and rejected by the AMO validator rather than ignored.
    delete manifest.cross_origin_embedder_policy;
    delete manifest.cross_origin_opener_policy;
    manifest.permissions = manifest.permissions.filter((p) => p !== "offscreen");
  }

  await writeFile(join(outdir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  return manifest;
}

/**
 * ORT's WASM binaries, served from the extension so nothing is fetched from a CDN. The
 * manifest CSP allows 'wasm-unsafe-eval' for exactly this.
 *
 * Only the plain SIMD+threads build is copied. The other three are 66 MB between them and
 * we cannot reach any of them: `jsep` is the WebGPU/WebNN build, `asyncify` and `jspi` are
 * for execution providers that suspend. We create sessions with executionProviders:
 * ["wasm"], which scope.md 5 fixes as the single execution path on both browsers so that
 * detection behaviour is identical -- a WebGPU path would need its own fp16 artifact and
 * its own calibration.
 *
 * Both files are needed: the runtime dynamically imports the .mjs at session creation and
 * that module fetches the .wasm. The worker imports "onnxruntime-web/wasm" rather than
 * "onnxruntime-web" for exactly this reason -- the default entry asks for the .jsep pair
 * instead, and a missing file surfaces as a flat "no available backend found" at the first
 * score, with the real cause one line further down. e2e/run.mjs is what caught it.
 */
const ORT_RUNTIME = ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"];

async function copyOrtRuntime() {
  const dist = join(root, "node_modules", "onnxruntime-web", "dist");
  const target = join(outdir, "ort");
  await mkdir(target, { recursive: true });
  for (const name of ORT_RUNTIME) await cp(join(dist, name), join(target, name));
  return ORT_RUNTIME.length;
}

async function copyStatic() {
  await mkdir(join(outdir, "assets"), { recursive: true });
  for (const name of ["tokenizer.json", "tokenizer_config.json", "calibration.json", "SHA256SUMS"]) {
    await cp(join(src, "assets", name), join(outdir, "assets", name));
  }
  for (const name of ["options.html", "first-run.html", "shared.css"]) {
    await cp(join(src, "pages", name), join(outdir, name));
  }
  if (browser === "chrome") {
    await cp(join(src, "host", "host.html"), join(outdir, "host.html"));
  }
  const icons = join(src, "assets", "icons");
  if (existsSync(icons)) await cp(icons, join(outdir, "icons"), { recursive: true });
}

const shared = {
  bundle: true,
  outdir,
  target: ["chrome120", "firefox140"],
  minify: !dev,
  sourcemap: dev ? "inline" : false,
  logLevel: "warning",
  // Compiled away, so a dev build can keep assertions the shipped one drops.
  define: {
    "process.env.NODE_ENV": JSON.stringify(dev ? "development" : "production"),
    MODEL_HOST_OVERRIDE: JSON.stringify(modelBaseUrl),
  },
};

async function run() {
  await rm(outdir, { recursive: true, force: true });
  await mkdir(outdir, { recursive: true });

  const jobs = Object.entries(ENTRIES)
    .filter(([name]) => !SKIP[browser].includes(name))
    .map(([name, spec]) => ({
      ...shared,
      entryPoints: { [name]: spec.in },
      format: spec.format,
    }));

  if (watch) {
    const contexts = await Promise.all(jobs.map((job) => context(job)));
    await Promise.all(contexts.map((c) => c.watch()));
    console.log(`watching ${browser} -> dist/${browser}`);
  } else {
    await Promise.all(jobs.map((job) => build(job)));
  }

  await copyStatic();
  const wasm = await copyOrtRuntime();
  const manifest = await buildManifest();

  console.log(
    `built ${browser} v${manifest.version} -> dist/${browser} ` +
      `(${jobs.length} bundles, ${wasm} ORT runtime files${dev ? ", dev" : ""})` +
      (modelBaseUrl ? `
  model host overridden -> ${modelBaseUrl}` : ""),
  );
}

await run();
