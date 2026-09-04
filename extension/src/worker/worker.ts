/**
 * The inference worker: one ORT Web session and one tokenizer, on a dedicated thread.
 *
 * Owns nothing else. No cache, no queue, no fetching -- the host hands it bytes that have
 * already been verified against SHA256SUMS, and it answers one text at a time.
 *
 * Batch is 1 per run() (scope.md 6.3). The graph's sequence axis is genuinely dynamic
 * (docs/measurements/export-verification.md), so a 215-token chunk costs what a 215-token
 * chunk should rather than being padded to 512.
 */

/// <reference lib="webworker" />

import { PreTrainedTokenizer } from "@huggingface/transformers";
import * as ort from "onnxruntime-web";

import { collapseWhitespace } from "../shared/normalize.js";
import type { WorkerRequest, WorkerResponse } from "../shared/protocol.js";

let session: ort.InferenceSession | null = null;
let tokenizer: PreTrainedTokenizer | null = null;
let maxLength = 512;

function post(message: WorkerResponse): void {
  (self as DedicatedWorkerGlobalScope).postMessage(message);
}

async function init(req: Extract<WorkerRequest, { type: "init" }>): Promise<void> {
  ort.env.wasm.wasmPaths = req.wasmPaths;
  ort.env.wasm.numThreads = req.numThreads;
  ort.env.wasm.simd = true;
  ort.env.logLevel = "error";
  maxLength = req.maxLength;

  // Built straight from the two bundled JSON blobs. Not AutoTokenizer.from_pretrained:
  // that path resolves files through transformers.js's own loader and could, if misconfig-
  // ured, reach a CDN. There is nothing to fetch and nothing to get wrong this way, and
  // tests/tokenizer-parity.test.ts pins this exact construction against Python.
  tokenizer = new PreTrainedTokenizer(
    JSON.parse(req.tokenizerJson),
    JSON.parse(req.tokenizerConfigJson),
  );

  session = await ort.InferenceSession.create(new Uint8Array(req.model), {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  post({
    type: "ready",
    threading: {
      crossOriginIsolated: self.crossOriginIsolated,
      numThreads: req.numThreads,
      hardwareConcurrency: navigator.hardwareConcurrency,
    },
  });
}

async function run(id: number, text: string): Promise<void> {
  if (session === null || tokenizer === null) {
    post({ type: "error", id, message: "worker is not initialised" });
    return;
  }
  // collapseWhitespace is what the model was trained on (scope.md 7.4). Not
  // normalizeForHash -- folding curly quotes and em dashes would discard real signal.
  const encoded = tokenizer(collapseWhitespace(text), {
    truncation: true,
    max_length: maxLength,
  });

  const ids = encoded.input_ids.data as BigInt64Array;
  const mask = encoded.attention_mask.data as BigInt64Array;
  const nTokens = ids.length;

  const outputs = await session.run({
    input_ids: new ort.Tensor("int64", ids, [1, nTokens]),
    attention_mask: new ort.Tensor("int64", mask, [1, nTokens]),
  });
  const head = outputs[session.outputNames[0]!];
  if (head === undefined) {
    post({ type: "error", id, message: "session produced no output" });
    return;
  }
  post({ type: "result", id, logit: (head.data as Float32Array)[0]!, nTokens });
}

self.onmessage = (event: MessageEvent<WorkerRequest>): void => {
  const req = event.data;
  void (async () => {
    try {
      switch (req.type) {
        case "init":
          await init(req);
          break;
        case "run":
          await run(req.id, req.text);
          break;
        case "close":
          await session?.release();
          session = null;
          tokenizer = null;
          break;
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      post(req.type === "run" ? { type: "error", id: req.id, message } : { type: "error", message });
    }
  })();
};
