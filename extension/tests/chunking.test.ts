import { describe, expect, it } from "vitest";

import { chunkBlock, chunkSpans } from "../src/shared/chunking.js";
import { collapseWhitespace, toCodePoints } from "../src/shared/normalize.js";
import { loadFixture, type WindowsFixture } from "./fixtures.js";

const fixture = loadFixture<WindowsFixture>("windows.json");

/**
 * chunkSpans exists so the content script can map a chunk back to a DOM range. It is only
 * safe if it is the same chunker as chunkBlock, which is what this asserts -- on the same
 * fixture cases the Python side pins, so neither can drift from the other or from Python.
 */
describe("chunkSpans agrees with chunkBlock", () => {
  for (const c of fixture.chunk_cases) {
    it(c.name, () => {
      const collapsed = collapseWhitespace(c.input);
      const cp = toCodePoints(collapsed);
      const fromSpans = chunkSpans(collapsed, c.min_words, c.max_words).map(([a, b]) =>
        cp.slice(a, b).join(""),
      );
      expect(fromSpans).toEqual(chunkBlock(c.input, c.min_words, c.max_words));
    });
  }

  it("returns spans that are ordered, disjoint and inside the text", () => {
    for (const c of fixture.chunk_cases) {
      const collapsed = collapseWhitespace(c.input);
      const n = toCodePoints(collapsed).length;
      const spans = chunkSpans(collapsed, c.min_words, c.max_words);
      let previousEnd = -1;
      for (const [a, b] of spans) {
        expect(a).toBeGreaterThan(previousEnd - 1);
        expect(a).toBeLessThan(b);
        expect(b).toBeLessThanOrEqual(n);
        previousEnd = b;
      }
    }
  });
});
