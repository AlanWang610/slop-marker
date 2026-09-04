/**
 * Fetch the real-page corpus (extension/README.md, "Known gaps").
 *
 *   node e2e/real/fetch.mjs [--force]
 *
 * Every automated test until now ran on markup this repo generated, so the harness had
 * never met a CMS template, a cookie banner, a sticky header, an SVG sprite sheet or a
 * lazily-hydrated sidebar. Those are what break extraction, and none of them appear in a
 * page written to be tested.
 *
 * Every source is a work of the United States Government, which is not subject to copyright
 * under 17 U.S.C. 105. That is the reason for the narrow selection: the pages are checked
 * into a public repository, so anything carrying a licence -- Wikipedia's share-alike
 * included -- was ruled out rather than attributed.
 *
 * All four are human-written, which is what makes them useful: they test the direction that
 * matters for a high-precision detector, and the one synthetic pages cannot test at all.
 * A false positive here is a real finding about the model, not a broken test.
 */

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { argv, exit } from "node:process";

const here = dirname(fileURLToPath(import.meta.url));
const force = argv.includes("--force");

/**
 * Chosen for structural variety rather than subject matter: a modern JavaScript-heavy CMS,
 * and an older server-rendered template, are the two shapes most of the web still is.
 */
const SOURCES = [
  {
    name: "nasa-press-release",
    url: "https://www.nasa.gov/news-release/nasa-sets-coverage-for-agencys-spacex-crew-11-launch-docking/",
    note: "modern CMS, heavy nav and script, press-release genre (the scope.md 2 hard negative)",
  },
  {
    name: "weather-heat-safety",
    url: "https://www.weather.gov/safety/heat",
    note: "older government template, table-based layout, short paragraphs",
  },
  {
    name: "weather-lightning",
    url: "https://www.weather.gov/safety/lightning-science-overview",
    note: "same template, longer expository prose",
  },
  {
    name: "weather-flood",
    url: "https://www.weather.gov/safety/flood",
    note: "same template, list-heavy",
  },
];

const HEADERS = {
  "user-agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) " +
    "Chrome/131.0.0.0 Safari/537.36",
  accept: "text/html,application/xhtml+xml",
  "accept-language": "en-US,en;q=0.9",
};

const sha256 = (text) => createHash("sha256").update(text, "utf-8").digest("hex");

async function main() {
  mkdirSync(here, { recursive: true });
  const manifestPath = join(here, "manifest.json");
  const previous = existsSync(manifestPath)
    ? JSON.parse(readFileSync(manifestPath, "utf-8"))
    : { pages: [] };

  const pages = [];
  for (const source of SOURCES) {
    const file = join(here, `${source.name}.html`);
    if (existsSync(file) && !force) {
      const html = readFileSync(file, "utf-8");
      const known = previous.pages.find((p) => p.name === source.name);
      console.log(`  keep   ${source.name} (${(html.length / 1024).toFixed(0)} KB)`);
      pages.push({ ...source, bytes: html.length, sha256: sha256(html), fetched: known?.fetched });
      continue;
    }

    const response = await fetch(source.url, { headers: HEADERS });
    if (!response.ok) {
      console.error(`  FAIL   ${source.name}: HTTP ${response.status} from ${source.url}`);
      exit(1);
    }
    const html = await response.text();
    writeFileSync(file, html, "utf-8");
    console.log(`  fetch  ${source.name} (${(html.length / 1024).toFixed(0)} KB)`);
    pages.push({
      ...source,
      bytes: html.length,
      sha256: sha256(html),
      fetched: new Date().toISOString().slice(0, 10),
    });
  }

  writeFileSync(
    manifestPath,
    `${JSON.stringify(
      {
        note:
          "Saved verbatim. Every source is a work of the United States Government and is " +
          "not subject to copyright (17 U.S.C. 105). Re-fetch with: node e2e/real/fetch.mjs --force",
        pages,
      },
      null,
      2,
    )}\n`,
    "utf-8",
  );
  console.log(`\n${pages.length} pages, manifest written`);
}

await main();
