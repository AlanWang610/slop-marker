import { describe, expect, it } from "vitest";

import {
  collapseWhitespace,
  contentHash,
  countWords,
  normalizeForHash,
} from "../src/shared/normalize.js";
import { loadFixture, type NormalizeFixture } from "./fixtures.js";

const fixture = loadFixture<NormalizeFixture>("normalize.json");

describe("normalize parity with fixtures/normalize.json", () => {
  it("has cases to check", () => {
    expect(fixture.cases.length).toBeGreaterThan(0);
  });

  for (const c of fixture.cases) {
    describe(c.name, () => {
      it("collapseWhitespace", () => {
        expect(collapseWhitespace(c.input)).toBe(c.collapsed);
      });
      it("normalizeForHash", () => {
        expect(normalizeForHash(c.input)).toBe(c.normalized);
      });
      it("countWords", () => {
        expect(countWords(c.input)).toBe(c.words);
      });
      it("contentHash", async () => {
        expect(await contentHash(c.input)).toBe(c.sha256);
      });
    });
  }
});

describe("invariants the fixtures encode", () => {
  it("folds typography for the hash but not for the model input", () => {
    const curly = `He said \u201Cno\u201D \u2014 twice\u2026`;
    expect(normalizeForHash(curly)).toBe('He said "no" - twice...');
    expect(collapseWhitespace(curly)).toBe(curly);
  });

  it("keeps ZWNJ and ZWJ, which are load-bearing", () => {
    for (const cp of [0x200c, 0x200d]) {
      const ch = String.fromCodePoint(cp);
      expect(normalizeForHash(`a${ch}b`)).toBe(`a${ch}b`);
    }
  });

  it("treats U+200B as invisible, not as whitespace", () => {
    expect(normalizeForHash("a\u200Bb")).toBe("ab");
  });
});
