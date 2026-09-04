/**
 * scope.md 5.6 asks that the browser tokenizer match training, including the [CLS]/[SEP]
 * post-processor. Nothing asserted it: verify_graph only ever checked model_max_length.
 * fixtures/tokenize.json is that assertion, emitted by the Python `tokenizers` bindings.
 *
 * This is why transformers.js is a dependency at all. We drive the ONNX session ourselves,
 * but re-implementing HF's BPE for a detector whose signal is lexical is not a risk worth
 * taking, and this test is what makes using their implementation safe.
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { bundleDirFor, loadFixture } from "./fixtures.js";

interface TokenizeFixture {
  model_version: string;
  max_length: number;
  cls_token_id: number;
  sep_token_id: number;
  cases: Array<{ text: string; input_ids: number[]; attention_mask: number[] }>;
}

const fx = loadFixture<TokenizeFixture>("tokenize.json");
const bundle = bundleDirFor(fx.model_version);
const have = bundle !== null && existsSync(`${bundle}/tokenizer.json`);

describe.skipIf(!have)(`tokenizer parity (${fx.model_version})`, async () => {
  const { PreTrainedTokenizer } = await import("@huggingface/transformers");
  // Constructed exactly the way src/worker/worker.ts constructs it, from the two bundled
  // JSON blobs. Testing a different load path would prove something we do not ship.
  const read = (name: string): unknown =>
    JSON.parse(readFileSync(join(bundle!, name), "utf-8")) as unknown;
  const tokenizer = new PreTrainedTokenizer(
    read("tokenizer.json"),
    read("tokenizer_config.json"),
  );

  it("adds [CLS] and [SEP] via the post-processor", () => {
    const out = tokenizer("hello world", { add_special_tokens: true });
    const ids = Array.from(out.input_ids.data as BigInt64Array, Number);
    expect(ids[0]).toBe(fx.cls_token_id);
    expect(ids[ids.length - 1]).toBe(fx.sep_token_id);
  });

  it("matches the Python token ids on every case", () => {
    const mismatches: string[] = [];
    for (const c of fx.cases) {
      const out = tokenizer(c.text, { truncation: true, max_length: fx.max_length });
      const ids = Array.from(out.input_ids.data as BigInt64Array, Number);
      const mask = Array.from(out.attention_mask.data as BigInt64Array, Number);
      if (JSON.stringify(ids) !== JSON.stringify(c.input_ids)) {
        mismatches.push(`ids differ for ${JSON.stringify(c.text.slice(0, 60))}`);
      } else if (JSON.stringify(mask) !== JSON.stringify(c.attention_mask)) {
        mismatches.push(`mask differs for ${JSON.stringify(c.text.slice(0, 60))}`);
      }
    }
    expect(mismatches).toEqual([]);
  });

  it("truncates at max_length", () => {
    for (const c of fx.cases) expect(c.input_ids.length).toBeLessThanOrEqual(fx.max_length);
  });
});
