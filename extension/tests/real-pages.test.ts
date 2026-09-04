/**
 * @vitest-environment jsdom
 *
 * Extraction against real markup, offline.
 *
 * Every other test in this suite runs on markup this repo generated, which means it has
 * never met a CMS wrapper, a cookie banner, an SVG sprite sheet or a nav rendered as a
 * table. `e2e/real/` holds four saved pages -- all works of the United States Government,
 * all human-written -- and this asserts what can be asserted without a model: that
 * extraction finds their prose, and that it does not drag in their furniture.
 *
 * The browser harness takes it from here and asserts the part that needs the model: that
 * none of it is flagged. That assertion would be vacuous if extraction found nothing, which
 * is what the counts below exist to rule out.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { extractBlocks, pinRoot } from "../src/content/extract.js";
import { buildChunks } from "../src/content/pipeline.js";
import { SHIPPED_CALIBRATION_JSON } from "../src/shared/bundle-config.js";
import { parseCalibration } from "../src/shared/calibration.js";
import { countWords } from "../src/shared/normalize.js";

const here = dirname(fileURLToPath(import.meta.url));
const corpus = join(here, "..", "e2e", "real");
const manifestPath = join(corpus, "manifest.json");

interface Manifest {
  pages: Array<{ name: string; url: string; note: string; bytes: number }>;
}

const have = existsSync(manifestPath);
const manifest: Manifest = have
  ? (JSON.parse(readFileSync(manifestPath, "utf-8")) as Manifest)
  : { pages: [] };
const cal = parseCalibration(SHIPPED_CALIBRATION_JSON);

/** Load a saved page into the jsdom document, exactly as the content script would meet it. */
function load(name: string): void {
  const html = readFileSync(join(corpus, `${name}.html`), "utf-8");
  // Replacing documentElement rather than body: the pages carry <head> content and
  // attributes on <html> that the real extraction walk would also see.
  document.documentElement.innerHTML = html
    .replace(/<!doctype[^>]*>/i, "")
    .replace(/<\/?html[^>]*>/gi, "");
}

describe.skipIf(!have)("saved real pages", () => {
  for (const page of manifest.pages) {
    describe(page.name, () => {
      it("extracts scoreable prose", () => {
        load(page.name);
        const chunks = buildChunks(cal, extractBlocks(pinRoot(null, document)));
        expect(chunks.length).toBeGreaterThan(0);
      });

      it("never emits a chunk under min_words", () => {
        load(page.name);
        for (const chunk of buildChunks(cal, extractBlocks(pinRoot(null, document)))) {
          // A chunk may fall under the floor only as a block's trailing remainder, which
          // chunking.ts documents; what must never happen is a whole block under it.
          expect(countWords(chunk.block.text)).toBeGreaterThanOrEqual(cal.min_words);
        }
      });

      it("drags in none of the page's furniture", () => {
        load(page.name);
        const excluded = "nav,header,footer,aside,form,figure,code,pre,button";
        for (const chunk of buildChunks(cal, extractBlocks(pinRoot(null, document)))) {
          expect(chunk.block.element.closest(excluded)).toBeNull();
          expect(chunk.block.element.closest("[contenteditable]")).toBeNull();
        }
      });

      it("produces chunks that index their own block", () => {
        load(page.name);
        for (const chunk of buildChunks(cal, extractBlocks(pinRoot(null, document)))) {
          expect(chunk.text.length).toBeGreaterThan(0);
          expect(chunk.words).toBeGreaterThan(0);
          expect(chunk.end).toBeGreaterThan(chunk.start);
        }
      });
    });
  }

  it("covers more than one page template", () => {
    expect(manifest.pages.length).toBeGreaterThanOrEqual(2);
  });

  it("reports what each page yields", () => {
    const rows = manifest.pages.map((page) => {
      load(page.name);
      const blocks = extractBlocks(pinRoot(null, document));
      const chunks = buildChunks(cal, blocks);
      return {
        page: page.name,
        kb: Math.round(page.bytes / 1024),
        blocks: blocks.length,
        chunks: chunks.length,
        words: chunks.reduce((sum, c) => sum + c.words, 0),
      };
    });
    console.table(rows);
    // Across the corpus there has to be real prose, or the browser-side "no false
    // positives" assertion is measuring nothing at all.
    expect(rows.reduce((sum, r) => sum + r.words, 0)).toBeGreaterThan(500);
  });
});
