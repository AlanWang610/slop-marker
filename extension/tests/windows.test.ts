import { describe, expect, it } from "vitest";

import { chunkBlock } from "../src/shared/chunking.js";
import { countWords } from "../src/shared/normalize.js";
import { sentences } from "../src/shared/sentences.js";
import { loadFixture, type WindowsFixture } from "./fixtures.js";

const fixture = loadFixture<WindowsFixture>("windows.json");

describe("splitSentences parity with fixtures/windows.json", () => {
  it("has cases to check", () => {
    expect(fixture.sentence_cases.length).toBeGreaterThan(0);
  });

  for (const c of fixture.sentence_cases) {
    it(c.name, () => {
      expect(sentences(c.input)).toEqual(c.sentences);
    });
  }

  it("covers the input exactly, as the Python invariant asserts", () => {
    for (const c of fixture.sentence_cases) {
      expect(sentences(c.input).join(" ")).toBe(c.input.trim());
    }
  });
});

describe("chunkBlock parity with fixtures/windows.json", () => {
  it("has cases to check", () => {
    expect(fixture.chunk_cases.length).toBeGreaterThan(0);
  });

  for (const c of fixture.chunk_cases) {
    it(c.name, () => {
      const chunks = chunkBlock(c.input, c.min_words, c.max_words);
      expect(chunks).toEqual(c.chunks.map((x) => x.text));
      expect(chunks.map(countWords)).toEqual(c.chunks.map((x) => x.words));
    });
  }
});
