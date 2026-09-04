/**
 * The whole pipeline, end to end, against Python: chunkBlock -> tokenize -> ORT Web ->
 * aggregate -> runs.
 *
 * The other parity tests each pin one stage. This one pins the composition, which is what
 * scope.md 1 actually promises: "identical detection behaviour on both browsers: same model
 * artifact, same calibration constants, same scoring logic." A build could chunk correctly,
 * score correctly, and still aggregate over the wrong sequence -- every other test in this
 * suite would stay green.
 *
 * It runs against the shipped calibration, not the synthetic fixture constants, because
 * here the real operating point is the thing under test.
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";

import { aggregate, type Chunk } from "../src/shared/aggregate.js";
import { parseCalibration } from "../src/shared/calibration.js";
import { chunkBlock } from "../src/shared/chunking.js";
import { collapseWhitespace, countWords } from "../src/shared/normalize.js";
import { truncateLikeTokenizers } from "../src/shared/tokenize.js";
import { bundleDirFor, loadFixture } from "./fixtures.js";

interface DocumentsFixture {
  model_version: string;
  tolerance: number;
  documents: Array<{
    name: string;
    text: string;
    chunks: Array<{ text: string; words: number; logit: number }>;
    expected: {
      t_on_effective: number;
      chunk_p: number[];
      chunk_p_penalized: number[];
      runs: Array<{ start: number; end: number; words: number; score: number; flagged: boolean }>;
    };
  }>;
}

const fx = loadFixture<DocumentsFixture>("documents.json");
const bundle = bundleDirFor(fx.model_version);
const have = bundle !== null && existsSync(join(bundle, "model.onnx"));

describe.skipIf(!have)(`document parity (${fx.model_version})`, () => {
  const cal = parseCalibration(readFileSync(join(bundle!, "calibration.json"), "utf-8"));
  let score: (text: string) => Promise<number>;

  beforeAll(async () => {
    const ort = await import("onnxruntime-web");
    const { PreTrainedTokenizer } = await import("@huggingface/transformers");
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = "error";

    // Constructed exactly as src/worker/worker.ts does.
    const read = (name: string): unknown =>
      JSON.parse(readFileSync(join(bundle!, name), "utf-8")) as unknown;
    const tokenizer = new PreTrainedTokenizer(read("tokenizer.json"), read("tokenizer_config.json"));
    const session = await ort.InferenceSession.create(
      new Uint8Array(readFileSync(join(bundle!, "model.onnx"))),
      { executionProviders: ["wasm"] },
    );

    // The worker's exact encode path, truncation fix included.
    score = async (text) => {
      const encoded = tokenizer(collapseWhitespace(text), { truncation: false });
      const { ids, mask } = truncateLikeTokenizers(
        Array.from(encoded.input_ids.data as BigInt64Array, Number),
        Array.from(encoded.attention_mask.data as BigInt64Array, Number),
        cal.max_length,
        tokenizer.sep_token_id,
      );
      const out = await session.run({
        input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]),
        attention_mask: new ort.Tensor("int64", BigInt64Array.from(mask, BigInt), [1, mask.length]),
      });
      return (out[session.outputNames[0]!]!.data as Float32Array)[0]!;
    };
  }, 180_000);

  it("uses the calibration the fixture was generated against", () => {
    expect(cal.version).toBe(fx.model_version);
  });

  it("chunks every document exactly as Python does", () => {
    for (const doc of fx.documents) {
      const chunks = chunkBlock(collapseWhitespace(doc.text), cal.min_words);
      expect(chunks, doc.name).toEqual(doc.chunks.map((c) => c.text));
      expect(chunks.map(countWords), doc.name).toEqual(doc.chunks.map((c) => c.words));
    }
  });

  it("reproduces every run, flag and threshold end to end", async () => {
    for (const doc of fx.documents) {
      const chunkTexts = chunkBlock(collapseWhitespace(doc.text), cal.min_words);
      const scored: Chunk[] = [];
      for (const text of chunkTexts) {
        scored.push({ logit: await score(text), words: countWords(text) });
      }
      const got = aggregate(scored, cal);
      const want = doc.expected;

      expect(got.t_on_effective, `${doc.name}: t_on_effective`).toBeCloseTo(want.t_on_effective, 9);
      expect(got.runs.length, `${doc.name}: run count`).toBe(want.runs.length);

      got.runs.forEach((run, i) => {
        const expected = want.runs[i]!;
        expect(
          { start: run.start, end: run.end, words: run.words, flagged: run.flagged },
          `${doc.name}: run ${i}`,
        ).toEqual({
          start: expected.start,
          end: expected.end,
          words: expected.words,
          flagged: expected.flagged,
        });
        // Scores come through the model, so they carry the int8 runtime's own tolerance
        // rather than being exact the way the pure-arithmetic fixtures are.
        expect(Math.abs(run.score - expected.score), `${doc.name}: run ${i} score`).toBeLessThan(
          fx.tolerance,
        );
      });
    }
  }, 900_000);

  it("covers both outcomes, so passing means something", () => {
    const flagged = fx.documents.filter((d) => d.expected.runs.some((r) => r.flagged));
    expect(flagged.length).toBeGreaterThan(0);
    expect(flagged.length).toBeLessThan(fx.documents.length);
  });

  it("covers at least one truncating chunk, where the tokenizers disagree", () => {
    // A 400-word chunk is ~500-540 tokens (scope.md 4.3), so this path is routine, not rare.
    const longest = Math.max(
      ...fx.documents.flatMap((d) => d.chunks.map((c) => c.words)),
    );
    expect(longest).toBeGreaterThan(330);
  });

  it("exercises the document prior", () => {
    // Sparse documents get t_on raised in log-odds space; if none did, the branch is untested.
    const bumped = fx.documents.filter((d) => d.expected.t_on_effective > cal.t_on + 1e-9);
    expect(bumped.length).toBeGreaterThan(0);
  });
});
