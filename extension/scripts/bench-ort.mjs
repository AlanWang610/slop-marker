/**
 * Phase 0 gate: can the shipped bundle actually run on ONNX Runtime Web's WASM backend,
 * and how fast?
 *
 * The shipping artifact is weight-only MatMulNBits (int8 encoder + int4 embedding table).
 * ORT's MLAS assigns QNBitGemmDispatch only under MLAS_TARGET_{AMD64,ARM64,RISCV64,LARCH64}
 * -- there is no WASM branch -- so MlasIsQNBitGemmAvailable() returns false in a WASM build
 * and every MatMulNBits node falls back to ComputeBUnpacked, which dequantizes the whole
 * weight matrix and transposes it on every forward pass. This measures what that costs.
 *
 *   node scripts/bench-ort.mjs [--bundle <dir>] [--threads N] [--reps N]
 */

import { readFileSync } from "node:fs";
import { argv } from "node:process";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..", "..");

function arg(name, fallback) {
  const hit = argv.find((a) => a.startsWith(`--${name}=`));
  if (hit) return hit.split("=").slice(1).join("=");
  const i = argv.indexOf(`--${name}`);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback;
}

const bundleDir = resolve(arg("bundle", join(repo, "artifacts", "bundles", "mb-base-0.2.0-dev")));
const threads = Number(arg("threads", "1"));
const reps = Number(arg("reps", "3"));

const ort = await import("onnxruntime-web");
ort.env.wasm.numThreads = threads;
ort.env.wasm.simd = true;
ort.env.logLevel = "error";

const calibration = JSON.parse(readFileSync(join(bundleDir, "calibration.json"), "utf-8"));
console.log(`bundle      ${bundleDir}`);
console.log(`version     ${calibration.version}`);
console.log(`threads     ${threads}`);
console.log(`ort         ${ort.env.versions?.common ?? "unknown"}`);

const modelBytes = new Uint8Array(readFileSync(join(bundleDir, "model.onnx")));
console.log(`model       ${(modelBytes.byteLength / 1e6).toFixed(1)} MB`);

const t0 = performance.now();
let session;
try {
  session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
} catch (err) {
  console.error(`\nSESSION CREATE FAILED: ${err?.message ?? err}`);
  console.error("This is the Phase 0 gate failing closed. See docs/measurements/.");
  process.exit(2);
}
const createMs = performance.now() - t0;
console.log(`session     created in ${(createMs / 1000).toFixed(1)} s`);
console.log(`inputs      ${session.inputNames.join(", ")}`);
console.log(`outputs     ${session.outputNames.join(", ")}`);

// Synthetic ids are fine for latency: cost depends on sequence length, not token identity.
function makeFeeds(seqLen) {
  const ids = BigInt64Array.from({ length: seqLen }, (_, i) =>
    BigInt(i === 0 ? 50281 : i === seqLen - 1 ? 50282 : 1000 + (i % 40000)),
  );
  const mask = BigInt64Array.from({ length: seqLen }, () => 1n);
  return {
    input_ids: new ort.Tensor("int64", ids, [1, seqLen]),
    attention_mask: new ort.Tensor("int64", mask, [1, seqLen]),
  };
}

console.log(`\n${"seq".padStart(6)} ${"median ms".padStart(12)} ${"min ms".padStart(10)}`);
const results = {};
for (const seqLen of [128, 215, 256, 384, 512]) {
  const feeds = makeFeeds(seqLen);
  await session.run(feeds); // warm up
  const times = [];
  for (let r = 0; r < reps; r++) {
    const t = performance.now();
    await session.run(feeds);
    times.push(performance.now() - t);
  }
  times.sort((a, b) => a - b);
  const median = times[Math.floor(times.length / 2)];
  results[seqLen] = median;
  console.log(
    `${String(seqLen).padStart(6)} ${median.toFixed(0).padStart(12)} ${times[0].toFixed(0).padStart(10)}`,
  );
}

const heap = process.memoryUsage();
console.log(`\nrss         ${(heap.rss / 1e6).toFixed(0)} MB`);
console.log(`budget      scope.md 10 wants ~300-600 ms at 256 tokens single-threaded`);
const at256 = results[256];
console.log(
  `verdict     256-token chunk at ${at256.toFixed(0)} ms -> ${at256 <= 600 ? "WITHIN" : "OVER"} budget` +
    (at256 > 600 ? ` by ${(at256 / 600).toFixed(1)}x` : ""),
);
