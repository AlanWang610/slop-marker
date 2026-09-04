/**
 * The gate scope.md 5 has no equivalent of: does the browser runtime reproduce the scores?
 *
 * fixtures/logits.json is the oracle, produced by Python onnxruntime on the shipped int8
 * artifact. This feeds ONNX Runtime Web the *same token ids* -- so a failure here is the
 * model runtime, never the tokenizer, which tokenizer-parity.test.ts covers separately.
 *
 * The r1 export shipped a bundle that loaded, ran, passed every structural check and had
 * lost twenty points of AUROC. This is the check that would have caught it in the browser.
 *
 * Skipped when the local bundle is absent or is a different version; run
 * `uv run --extra modal python tools/pull_bundle.py --version <v>` to enable it.
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";

import { bundleDirFor, loadFixture } from "./fixtures.js";

interface LogitsFixture {
  version: number;
  model_version: string;
  tolerance: number;
  cases: Array<{ text: string; n_tokens: number; logit: number }>;
}
interface TokenizeFixture {
  model_version: string;
  cases: Array<{ text: string; input_ids: number[]; attention_mask: number[] }>;
}

const logitsFx = loadFixture<LogitsFixture>("logits.json");
const tokenizeFx = loadFixture<TokenizeFixture>("tokenize.json");
const bundle = bundleDirFor(logitsFx.model_version);
const haveModel = bundle !== null && existsSync(join(bundle, "model.onnx"));

describe.skipIf(!haveModel)(`ORT Web logit parity (${logitsFx.model_version})`, () => {
  let run: (ids: number[], mask: number[]) => Promise<number>;

  beforeAll(async () => {
    const ort = await import("onnxruntime-web");
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = "error";
    const bytes = new Uint8Array(readFileSync(join(bundle!, "model.onnx")));
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    run = async (ids, mask) => {
      const out = await session.run({
        input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]),
        attention_mask: new ort.Tensor("int64", BigInt64Array.from(mask, BigInt), [1, mask.length]),
      });
      const logits = out[session.outputNames[0]!]!.data as Float32Array;
      return logits[0]!;
    };
  }, 120_000);

  it("fixtures agree on which texts they describe", () => {
    expect(tokenizeFx.model_version).toBe(logitsFx.model_version);
    expect(tokenizeFx.cases.map((c) => c.text)).toEqual(logitsFx.cases.map((c) => c.text));
  });

  it("reproduces every logit within tolerance", async () => {
    const deltas: number[] = [];
    for (const [i, want] of logitsFx.cases.entries()) {
      const tokens = tokenizeFx.cases[i]!;
      const got = await run(tokens.input_ids, tokens.attention_mask);
      deltas.push(Math.abs(got - want.logit));
    }
    const max = Math.max(...deltas);
    const mean = deltas.reduce((a, b) => a + b, 0) / deltas.length;
    // Surfaced because a silent near-miss is the interesting failure, not a hard one.
    console.log(`  logit parity: max |delta| ${max.toExponential(2)}, mean ${mean.toExponential(2)}`);
    expect(max).toBeLessThan(logitsFx.tolerance);
  }, 600_000);

  it("puts the two classes on opposite sides, so the oracle exercises real range", () => {
    const values = logitsFx.cases.map((c) => c.logit);
    expect(Math.min(...values)).toBeLessThan(-1);
    expect(Math.max(...values)).toBeGreaterThan(1);
  });
});
