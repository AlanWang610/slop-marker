/**
 * Fetching, verifying and caching the bundle (scope.md 6.4).
 *
 * The assertions that matter are the negative ones. This is the only code in the extension
 * that decides whether to trust 137 MB of downloaded weights, and every failure mode has to
 * fail *closed*: a corrupted file must throw and must not reach the cache, because a
 * half-written model would then be loaded on every subsequent run with nothing to point at.
 *
 * `caches` and `fetch` are stubbed rather than mocked at the module level, so the real
 * control flow -- including the streaming, the state callbacks and the Cache API writes --
 * is what runs.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  clearCache,
  download,
  isCached,
  parseSha256Sums,
  readCached,
  readCachedText,
  type BundleSpec,
} from "../src/host/model-store.js";
import type { ModelState } from "../src/shared/protocol.js";

/* ------------------------------------------------------------ an in-memory Cache API */

class FakeCache {
  readonly entries = new Map<string, ArrayBuffer>();
  async match(url: string): Promise<Response | undefined> {
    const bytes = this.entries.get(url);
    return bytes === undefined ? undefined : new Response(bytes);
  }
  async put(url: string, response: Response): Promise<void> {
    this.entries.set(url, await response.arrayBuffer());
  }
}

let cache: FakeCache;

function installCaches(): void {
  cache = new FakeCache();
  vi.stubGlobal("caches", {
    open: async () => cache,
    delete: async () => {
      cache.entries.clear();
      return true;
    },
  });
}

const bytesOf = (text: string): ArrayBuffer => new TextEncoder().encode(text).buffer;

async function sha256(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytesOf(text));
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/** A two-file bundle whose contents we control end to end. */
async function spec(contents: Record<string, string>): Promise<BundleSpec> {
  const expected = new Map<string, string>();
  for (const [name, text] of Object.entries(contents)) expected.set(name, await sha256(text));
  return { files: Object.keys(contents), expected };
}

function serve(contents: Record<string, string | null>): void {
  vi.stubGlobal("fetch", (url: string) => {
    const name = decodeURIComponent(url.split("/").pop()!);
    const body = contents[name];
    if (body === undefined || body === null) {
      return Promise.resolve(new Response("nope", { status: 404 }));
    }
    return Promise.resolve(
      new Response(body, { status: 200, headers: { "content-length": String(body.length) } }),
    );
  });
}

beforeEach(installCaches);
afterEach(() => vi.unstubAllGlobals());

/* ------------------------------------------------------------------- parseSha256Sums */

describe("parseSha256Sums", () => {
  const hash = "a".repeat(64);
  const other = "b".repeat(64);

  it("reads coreutils output with two spaces", () => {
    expect(parseSha256Sums(`${hash}  model.onnx`).get("model.onnx")).toBe(hash);
  });

  it("reads a single space", () => {
    expect(parseSha256Sums(`${hash} model.onnx`).get("model.onnx")).toBe(hash);
  });

  it("reads the binary-mode asterisk", () => {
    expect(parseSha256Sums(`${hash} *model.onnx`).get("model.onnx")).toBe(hash);
  });

  it("reads several lines", () => {
    const map = parseSha256Sums(`${hash}  model.onnx\n${other}  calibration.json\n`);
    expect(map.size).toBe(2);
    expect(map.get("calibration.json")).toBe(other);
  });

  it("tolerates CRLF and blank lines", () => {
    const map = parseSha256Sums(`${hash}  model.onnx\r\n\r\n${other}  config.json\r\n`);
    expect(map.get("model.onnx")).toBe(hash);
    expect(map.get("config.json")).toBe(other);
  });

  it("keeps names containing spaces", () => {
    expect(parseSha256Sums(`${hash}  a file.json`).get("a file.json")).toBe(hash);
  });

  it("rejects a malformed line rather than silently skipping it", () => {
    expect(() => parseSha256Sums("not a checksum line")).toThrow(/malformed/);
  });

  it("rejects a truncated digest", () => {
    expect(() => parseSha256Sums(`${"a".repeat(63)}  model.onnx`)).toThrow(/malformed/);
  });

  it("rejects a digest with non-hex characters", () => {
    expect(() => parseSha256Sums(`${"z".repeat(64)}  model.onnx`)).toThrow(/malformed/);
  });

  it("returns an empty map for empty input", () => {
    expect(parseSha256Sums("\n\n").size).toBe(0);
  });
});

/* -------------------------------------------------------------------------- download */

describe("download", () => {
  const contents = { "model.onnx": "the weights", "calibration.json": "{}" };

  it("fetches, verifies and caches every file", async () => {
    serve(contents);
    const states: ModelState[] = [];
    await download("v1", (s) => states.push(s), await spec(contents));

    expect(cache.entries.size).toBe(2);
    expect(states.map((s) => s.phase)).toContain("downloading");
    expect(states.map((s) => s.phase)).toContain("verifying");
    expect(states.at(-1)).toEqual({ phase: "ready", version: "v1" });
  });

  it("reports progress against bytes actually received", async () => {
    serve(contents);
    const states: ModelState[] = [];
    await download("v1", (s) => states.push(s), await spec(contents));
    const progress = states.filter((s) => s.phase === "downloading");
    expect(progress.at(-1)).toMatchObject({ received: 13 });
  });

  it("puts each file under a version-scoped URL", async () => {
    serve(contents);
    await download("v1", () => undefined, await spec(contents));
    for (const url of cache.entries.keys()) expect(url).toContain("/v1/");
  });

  it("fails closed on a checksum mismatch, caching nothing", async () => {
    const good = await spec(contents);
    serve({ ...contents, "model.onnx": "tampered weights" });

    await expect(download("v1", () => undefined, good)).rejects.toThrow(/checksum mismatch/);
    expect(cache.entries.size).toBe(0);
  });

  it("names the offending file in the checksum error", async () => {
    const good = await spec(contents);
    serve({ ...contents, "model.onnx": "tampered weights" });
    await expect(download("v1", () => undefined, good)).rejects.toThrow(/model\.onnx/);
  });

  it("fails closed on a truncated file", async () => {
    const good = await spec(contents);
    serve({ ...contents, "model.onnx": "the weight" });
    await expect(download("v1", () => undefined, good)).rejects.toThrow(/checksum mismatch/);
    expect(cache.entries.size).toBe(0);
  });

  it("does not cache the good files that preceded a bad one", async () => {
    const good = await spec(contents);
    serve({ "model.onnx": "the weights", "calibration.json": "tampered" });
    await expect(download("v1", () => undefined, good)).rejects.toThrow();
    // model.onnx verified and was cached; calibration.json failed. Nothing is marked ready,
    // and isCached is false, so the next run re-downloads rather than loading a partial set.
    expect(await isCached("v1", good.files)).toBe(false);
  });

  it("throws on an HTTP error, naming the file and the URL", async () => {
    serve({ "model.onnx": null, "calibration.json": "{}" });
    await expect(download("v1", () => undefined, await spec(contents))).rejects.toThrow(
      /model\.onnx: HTTP 404 from https:/,
    );
  });

  it("refuses a file that the shipped SHA256SUMS does not list", async () => {
    serve({ ...contents, extra: "surprise" });
    const partial = await spec({ "model.onnx": "the weights" });
    const rogue: BundleSpec = { files: ["model.onnx", "extra"], expected: partial.expected };
    await expect(download("v1", () => undefined, rogue)).rejects.toThrow(/not listed/);
  });

  it("is idempotent once everything is cached, without fetching again", async () => {
    const s = await spec(contents);
    serve(contents);
    await download("v1", () => undefined, s);

    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    const states: ModelState[] = [];
    await download("v1", (st) => states.push(st), s);

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(states).toEqual([{ phase: "ready", version: "v1" }]);
  });

  it("keeps versions apart, so a new model is a fresh download", async () => {
    const s = await spec(contents);
    serve(contents);
    await download("v1", () => undefined, s);
    expect(await isCached("v2", s.files)).toBe(false);
  });
});

/* ------------------------------------------------------------------- cache accessors */

describe("isCached", () => {
  const contents = { "model.onnx": "w", "calibration.json": "{}" };

  it("is false on an empty cache", async () => {
    expect(await isCached("v1", ["model.onnx"])).toBe(false);
  });

  it("is false when only some files are present", async () => {
    const s = await spec(contents);
    serve({ "model.onnx": "w", "calibration.json": "bad" });
    await expect(download("v1", () => undefined, s)).rejects.toThrow();
    expect(await isCached("v1", s.files)).toBe(false);
  });

  it("is true once every file is there", async () => {
    const s = await spec(contents);
    serve(contents);
    await download("v1", () => undefined, s);
    expect(await isCached("v1", s.files)).toBe(true);
  });
});

describe("reading back", () => {
  const contents = { "model.onnx": "the weights" };

  it("returns the exact bytes that were verified", async () => {
    const s = await spec(contents);
    serve(contents);
    await download("v1", () => undefined, s);
    expect(new TextDecoder().decode(await readCached("v1", "model.onnx"))).toBe("the weights");
  });

  it("decodes text files", async () => {
    const s = await spec(contents);
    serve(contents);
    await download("v1", () => undefined, s);
    expect(await readCachedText("v1", "model.onnx")).toBe("the weights");
  });

  it("throws rather than returning empty bytes when nothing is cached", async () => {
    await expect(readCached("v1", "model.onnx")).rejects.toThrow(/not cached/);
  });

  it("clearCache empties everything", async () => {
    const s = await spec(contents);
    serve(contents);
    await download("v1", () => undefined, s);
    await clearCache();
    expect(await isCached("v1", s.files)).toBe(false);
  });
});
