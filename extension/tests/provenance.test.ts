/**
 * @vitest-environment jsdom
 *
 * The collapsed-text <-> DOM mapping. This is the code that decides whether a highlight
 * lands on the words the model actually scored, and it is the trickiest in the extension:
 * `collapseWithProvenance` duplicates `collapseWhitespace` (the shared one cannot report
 * provenance) and `rangeFor` walks the result back through UTF-16 offsets.
 *
 * The duplication is pinned two ways. Within a single block the two must agree character
 * for character, and the expected value comes from `fixtures/normalize.json` -- so this is
 * a cross-language assertion against Python, not self-consistency. The text is scattered
 * across randomly-placed inline elements first, because real markup is never one text node
 * and a single-node test would prove nothing about the walk.
 */

import { describe, expect, it } from "vitest";

import { collapseWithProvenance, extractBlocks, rangeFor } from "../src/content/extract.js";
import { chunkSpans } from "../src/shared/chunking.js";
import { collapseWhitespace, toCodePoints } from "../src/shared/normalize.js";
import { loadFixture, type NormalizeFixture } from "./fixtures.js";

interface DocumentsFixture {
  documents: Array<{ name: string; text: string; chunks: Array<{ text: string; words: number }> }>;
}

const normalizeFx = loadFixture<NormalizeFixture>("normalize.json");
const documentsFx = loadFixture<DocumentsFixture>("documents.json");

/** mulberry32: a seeded PRNG, so a failure is reproducible from its case name alone. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const INLINE = ["em", "a", "b", "span", "i", "strong"];

/**
 * A <p> holding `text`, cut at random points into bare text nodes and inline wrappers.
 * Inline only: a block-level wrapper would legitimately insert a word boundary, which the
 * separate suite at the bottom covers.
 */
function scatter(text: string, seed: number): HTMLParagraphElement {
  const random = rng(seed);
  const p = document.createElement("p");
  const cp = toCodePoints(text);
  let i = 0;
  while (i < cp.length) {
    const take = 1 + Math.floor(random() * 12);
    const piece = cp.slice(i, i + take).join("");
    i += take;
    if (random() < 0.45) {
      const el = document.createElement(INLINE[Math.floor(random() * INLINE.length)]!);
      el.append(document.createTextNode(piece));
      p.append(el);
    } else {
      p.append(document.createTextNode(piece));
    }
  }
  document.body.replaceChildren(p);
  return p;
}

/** A block whose text is a single text node, for the offset edge cases. */
function plainBlock(text: string) {
  document.body.replaceChildren();
  const p = document.createElement("p");
  p.append(document.createTextNode(text));
  document.body.append(p);
  return extractBlocks(document.body)[0]!;
}

describe("collapseWithProvenance agrees with the Python-pinned collapse", () => {
  for (const [index, testCase] of normalizeFx.cases.entries()) {
    it(`normalize.json: ${testCase.name}`, () => {
      expect(collapseWithProvenance(scatter(testCase.input, index + 1)).text).toBe(
        testCase.collapsed,
      );
    });
  }

  for (const [index, doc] of documentsFx.documents.entries()) {
    it(`documents.json: ${doc.name}`, () => {
      expect(collapseWithProvenance(scatter(doc.text, 1000 + index)).text).toBe(
        collapseWhitespace(doc.text),
      );
    });
  }

  it("holds under 25 different scatterings of the same text", () => {
    const text = documentsFx.documents[0]!.text;
    const want = collapseWhitespace(text);
    for (let seed = 1; seed <= 25; seed++) {
      expect(collapseWithProvenance(scatter(text, seed)).text).toBe(want);
    }
  });

  it("records exactly one provenance entry per codepoint", () => {
    for (const [index, testCase] of normalizeFx.cases.entries()) {
      const { text, provenance } = collapseWithProvenance(scatter(testCase.input, index + 500));
      expect(provenance).toHaveLength(toCodePoints(text).length);
    }
  });

  it("keeps every provenance offset inside its own node", () => {
    const { provenance } = collapseWithProvenance(scatter(documentsFx.documents[0]!.text, 7));
    for (const entry of provenance) {
      expect(entry.offset).toBeGreaterThanOrEqual(0);
      expect(entry.offset).toBeLessThan(entry.node.length);
    }
  });
});

describe("rangeFor round-trips every chunk of every fixture document", () => {
  for (const [index, doc] of documentsFx.documents.entries()) {
    it(doc.name, () => {
      scatter(doc.text, 2000 + index);
      const block = extractBlocks(document.body)[0]!;
      const cp = toCodePoints(block.text);
      const spans = chunkSpans(block.text, 40, 400);
      expect(spans.length).toBeGreaterThan(0);

      for (const [start, end] of spans) {
        const range = rangeFor(block, start, end);
        expect(range).not.toBeNull();
        expect(collapseWhitespace(range!.toString())).toBe(
          collapseWhitespace(cp.slice(start, end).join("")),
        );
      }
    });
  }

  it("produces the fixture's own chunk texts from the DOM", () => {
    for (const [index, doc] of documentsFx.documents.entries()) {
      scatter(doc.text, 3000 + index);
      const block = extractBlocks(document.body)[0]!;
      const cp = toCodePoints(block.text);
      const got = chunkSpans(block.text, 40, 400).map(([a, b]) => cp.slice(a, b).join(""));
      expect(got).toEqual(doc.chunks.map((c) => c.text));
    }
  });
});

describe("rangeFor edge cases", () => {
  it("returns null for an empty span", () => {
    expect(rangeFor(plainBlock("alpha beta gamma"), 3, 3)).toBeNull();
  });

  it("returns null for a reversed span", () => {
    expect(rangeFor(plainBlock("alpha beta gamma"), 8, 2)).toBeNull();
  });

  it("returns null past the end of the text", () => {
    expect(rangeFor(plainBlock("alpha beta gamma"), 0, 999)).toBeNull();
  });

  it("returns null once a page mutation has detached the nodes", () => {
    const block = plainBlock("alpha beta gamma");
    document.body.replaceChildren();
    expect(rangeFor(block, 0, 5)).toBeNull();
  });

  it("selects a single codepoint", () => {
    expect(rangeFor(plainBlock("alpha beta gamma"), 0, 1)!.toString()).toBe("a");
  });

  it("spans several text nodes and elements", () => {
    document.body.replaceChildren();
    const p = document.createElement("p");
    p.append(document.createTextNode("alpha "));
    const em = document.createElement("em");
    em.append(document.createTextNode("beta"));
    p.append(em, document.createTextNode(" gamma"));
    document.body.append(p);
    const block = extractBlocks(document.body)[0]!;
    expect(block.text).toBe("alpha beta gamma");
    expect(rangeFor(block, 0, 16)!.toString()).toBe("alpha beta gamma");
    expect(rangeFor(block, 6, 10)!.toString()).toBe("beta");
  });
});

describe("astral characters and combining marks", () => {
  const cases: Array<[string, string]> = [
    ["emoji", "alpha \u{1F600} beta gamma delta"],
    ["emoji ZWJ family", "alpha \u{1F468}‍\u{1F469}‍\u{1F467}‍\u{1F466} beta"],
    ["regional indicator flag", "alpha \u{1F1EF}\u{1F1F5} beta gamma"],
    ["combining acute", "alpha éclair beta gamma"],
    ["CJK", "alpha 日本語 beta gamma"],
    ["math bold plane 1", "alpha \u{1D400}\u{1D401} beta gamma"],
  ];

  for (const [name, text] of cases) {
    it(`${name}: one provenance entry per codepoint`, () => {
      const { text: collapsed, provenance } = collapseWithProvenance(scatter(text, 99));
      expect(collapsed).toBe(collapseWhitespace(text));
      expect(provenance).toHaveLength(toCodePoints(collapsed).length);
    });

    it(`${name}: every prefix and suffix range round-trips`, () => {
      const block = plainBlock(text);
      const cp = toCodePoints(block.text);
      for (let end = 1; end <= cp.length; end++) {
        expect(rangeFor(block, 0, end)!.toString()).toBe(cp.slice(0, end).join(""));
      }
      for (let start = 0; start < cp.length; start++) {
        expect(rangeFor(block, start, cp.length)!.toString()).toBe(cp.slice(start).join(""));
      }
    });
  }
});

describe("multi-block content", () => {
  it("inserts a separator the DOM has no node for, and still covers the right glyphs", () => {
    document.body.replaceChildren();
    const li = document.createElement("li");
    li.innerHTML = "<p>Alpha alpha.</p><p>Beta beta.</p>";
    document.body.append(li);
    const block = extractBlocks(document.body)[0]!;
    expect(block.text).toBe("Alpha alpha. Beta beta.");

    // The synthetic boundary space has no DOM counterpart, so toString() lacks it. What
    // has to hold is that the range covers exactly the same glyphs.
    const strip = (s: string): string => s.replace(/\s+/gu, "");
    const range = rangeFor(block, 0, toCodePoints(block.text).length)!;
    expect(strip(range.toString())).toBe(strip(block.text));
  });
});
