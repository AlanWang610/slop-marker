/**
 * @vitest-environment jsdom
 *
 * Document parity *through the DOM*.
 *
 * document-parity.test.ts already pins chunk -> score -> aggregate on raw strings. What it
 * cannot see is everything the content script does either side of that: pulling text out of
 * a page, deciding which blocks are scoreable, and mapping runs back onto ranges. A build
 * could pass every string-level test and still highlight the wrong paragraph.
 *
 * So this renders each fixture document as real markup, runs the real extraction, feeds the
 * fixture's own logits, and asserts the runs Python computed come back out.
 *
 * Each document goes in as a single <p>, because `fixtures/documents.json` was produced by
 * chunking the whole document text as one block. Splitting it across paragraphs would chunk
 * differently and compare nothing.
 */

import { beforeEach, describe, expect, it } from "vitest";

import { SHIPPED_CALIBRATION_JSON } from "../src/shared/bundle-config.js";
import { parseCalibration } from "../src/shared/calibration.js";
import { collapseWhitespace, toCodePoints } from "../src/shared/normalize.js";
import { extractBlocks } from "../src/content/extract.js";
import { buildChunks, computeRuns } from "../src/content/pipeline.js";
import { loadFixture } from "./fixtures.js";

interface DocumentsFixture {
  model_version: string;
  tolerance: number;
  documents: Array<{
    name: string;
    text: string;
    chunks: Array<{ text: string; words: number; logit: number }>;
    expected: {
      runs: Array<{ start: number; end: number; words: number; score: number; flagged: boolean }>;
    };
  }>;
}

const fx = loadFixture<DocumentsFixture>("documents.json");
const cal = parseCalibration(SHIPPED_CALIBRATION_JSON);

/** Whole document in one <p>, matching how the fixture was chunked. */
function renderDocument(text: string): void {
  document.body.replaceChildren();
  const article = document.createElement("article");
  const p = document.createElement("p");
  p.append(document.createTextNode(text));
  article.append(p);
  document.body.append(article);
}

beforeEach(() => {
  document.body.replaceChildren();
});

it("the shipped calibration is the one the fixture was built against", () => {
  expect(cal.version).toBe(fx.model_version);
});

describe("extraction and chunking reproduce the fixture", () => {
  for (const doc of fx.documents) {
    it(doc.name, () => {
      renderDocument(doc.text);
      const chunks = buildChunks(cal, extractBlocks());
      expect(chunks.map((c) => c.text)).toEqual(doc.chunks.map((c) => c.text));
      expect(chunks.map((c) => c.words)).toEqual(doc.chunks.map((c) => c.words));
    });
  }
});

describe("aggregation through the DOM reproduces Python's runs", () => {
  for (const doc of fx.documents) {
    it(doc.name, () => {
      renderDocument(doc.text);
      const chunks = buildChunks(cal, extractBlocks());
      for (const [i, chunk] of chunks.entries()) chunk.logit = doc.chunks[i]!.logit;

      const runs = computeRuns(chunks, cal);
      const wanted = doc.expected.runs.filter((r) => r.flagged);
      expect(runs).toHaveLength(wanted.length);

      for (const [i, run] of runs.entries()) {
        expect(run.score).toBeCloseTo(wanted[i]!.score, 9);
        expect(run.words).toBe(wanted[i]!.words);
        expect(run.modelVersion).toBe(cal.version);
        expect(run.ranges.length).toBeGreaterThan(0);
      }
    });
  }

  it("covers at least one flagged and one clean document", () => {
    const flagged = fx.documents.filter((d) => d.expected.runs.some((r) => r.flagged));
    expect(flagged.length).toBeGreaterThan(0);
    expect(fx.documents.length - flagged.length).toBeGreaterThan(0);
  });

  it("selects exactly the chunks the flagged run covers", () => {
    const doc = fx.documents.find((d) => d.expected.runs.some((r) => r.flagged))!;
    renderDocument(doc.text);
    const chunks = buildChunks(cal, extractBlocks());
    for (const [i, chunk] of chunks.entries()) chunk.logit = doc.chunks[i]!.logit;

    const run = computeRuns(chunks, cal)[0]!;
    const want = doc.expected.runs.filter((r) => r.flagged)[0]!;
    const covered = chunks.slice(want.start, want.end + 1).map((c) => c.text).join(" ");
    const highlighted = run.ranges.map((r) => r.toString()).join(" ");
    expect(collapseWhitespace(highlighted)).toBe(collapseWhitespace(covered));
  });
});

describe("computeRuns holds back a partially scored page", () => {
  it("returns nothing while any chunk is unscored, so highlights never flash", () => {
    const doc = fx.documents.find((d) => d.chunks.length > 1 && d.expected.runs.some((r) => r.flagged));
    if (doc === undefined) return; // no multi-chunk flagged document in the fixture
    renderDocument(doc.text);
    const chunks = buildChunks(cal, extractBlocks());
    for (const [i, chunk] of chunks.entries()) chunk.logit = doc.chunks[i]!.logit;

    chunks[chunks.length - 1]!.logit = null;
    expect(computeRuns(chunks, cal)).toEqual([]);
  });

  it("returns nothing for an empty page", () => {
    expect(computeRuns([], cal)).toEqual([]);
  });
});

describe("buildChunks drops what must never be scored", () => {
  it("skips blocks under min_words (scope.md 7.1)", () => {
    document.body.replaceChildren();
    const p = document.createElement("p");
    p.append(document.createTextNode("Only a handful of words here."));
    document.body.append(p);
    expect(buildChunks(cal, extractBlocks())).toEqual([]);
  });

  it("skips non-English blocks (scope.md 7.2)", () => {
    const french =
      "Le développement de ce modèle repose sur une analyse détaillée des textes " +
      "rédigés par des auteurs humains et par des systèmes automatiques, afin de " +
      "distinguer les deux catégories avec une précision suffisante pour être utile " +
      "dans la pratique quotidienne des lecteurs attentifs et exigeants du web moderne.";
    document.body.replaceChildren();
    const p = document.createElement("p");
    p.append(document.createTextNode(french));
    document.body.append(p);
    expect(buildChunks(cal, extractBlocks())).toEqual([]);
  });

  it("chunks each block independently, so a run may span blocks", () => {
    const doc = fx.documents[0]!;
    document.body.replaceChildren();
    const article = document.createElement("article");
    for (let i = 0; i < 3; i++) {
      const p = document.createElement("p");
      p.append(document.createTextNode(doc.text));
      article.append(p);
    }
    document.body.append(article);

    const chunks = buildChunks(cal, extractBlocks());
    expect(chunks).toHaveLength(doc.chunks.length * 3);
    expect(new Set(chunks.map((c) => c.block.element)).size).toBe(3);
  });

  it("carries a codepoint span that indexes its own block", () => {
    renderDocument(fx.documents[0]!.text);
    for (const chunk of buildChunks(cal, extractBlocks())) {
      const cp = toCodePoints(chunk.block.text);
      expect(chunk.start).toBeGreaterThanOrEqual(0);
      expect(chunk.end).toBeLessThanOrEqual(cp.length);
      expect(cp.slice(chunk.start, chunk.end).join("")).toBe(chunk.text);
    }
  });
});
