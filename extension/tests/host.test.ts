/**
 * The inference host's queue (scope.md 6.3, 7.4).
 *
 * One chunk in flight at a time, always the most urgent one *currently* queued. That
 * "currently" is the whole point: a reader can scroll while a chunk is being scored, and the
 * queue has to be re-read between runs or viewport-first scheduling is decorative. It is
 * also the one part of the extension where a mistake is invisible -- the page still ends up
 * fully scored, just in the wrong order, which no screenshot would show.
 *
 * Driven with a fake worker so ordering is deterministic. The real one is covered by
 * model-parity.test.ts and the browser harnesses.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

// The Host writes the observed threading mode through the storage shim, which in Node has
// no chrome global to reach. Nothing here asserts on it.
vi.mock("../src/shared/storage.js", () => ({
  isProxied: false,
  get: async () => ({}),
  set: async () => undefined,
  remove: async () => undefined,
  handleProxyMessage: () => false,
}));

import type { HostEnv } from "../src/host/host.js";
import type { CachedScore } from "../src/host/score-cache.js";
import type { HostMessage, ScoreRequest } from "../src/shared/protocol.js";

// Imported after the mock above is registered, so the Host picks up the stubbed storage.
const { Host } = await import("../src/host/host.js");
const { MODEL_VERSION } = await import("../src/shared/bundle-config.js");

/* ------------------------------------------------------------------- fake worker */

interface Run {
  id: number;
  text: string;
}

class FakeWorker {
  readonly runs: Run[] = [];
  readonly listeners = new Set<(event: { data: unknown }) => void>();
  terminated = false;
  /** Set to make init report failure instead of readiness. */
  failInit: string | null = null;

  addEventListener(_type: string, fn: (event: { data: unknown }) => void): void {
    this.listeners.add(fn);
  }
  removeEventListener(_type: string, fn: (event: { data: unknown }) => void): void {
    this.listeners.delete(fn);
  }
  terminate(): void {
    this.terminated = true;
  }

  postMessage(message: { type: string; id?: number; text?: string }): void {
    if (message.type === "init") {
      queueMicrotask(() =>
        this.emit(
          this.failInit === null
            ? { type: "ready", threading: { crossOriginIsolated: true, numThreads: 4, hardwareConcurrency: 8 } }
            : { type: "error", message: this.failInit },
        ),
      );
      return;
    }
    if (message.type === "run") this.runs.push({ id: message.id!, text: message.text! });
  }

  emit(data: unknown): void {
    for (const fn of [...this.listeners]) fn({ data });
  }
  finish(id: number, logit: number): void {
    this.emit({ type: "result", id, logit, nTokens: 215 });
  }
  fail(id: number, message: string): void {
    this.emit({ type: "error", id, message });
  }
  as(): Worker {
    return this as unknown as Worker;
  }
}

/* -------------------------------------------------------------------- fake cache */

function fakeCache() {
  const store = new Map<string, CachedScore>();
  const calls = { gets: 0, puts: 0 };
  return {
    store,
    calls,
    cache: {
      get: async (hash: string, version: string) => {
        calls.gets++;
        const hit = store.get(hash);
        return hit !== undefined && hit.modelVersion === version ? hit : null;
      },
      put: async (hash: string, entry: CachedScore) => {
        calls.puts++;
        store.set(hash, entry);
      },
      clear: async () => store.clear(),
      evictOtherVersions: async () => 0,
      size: async () => store.size,
    },
  };
}

/* --------------------------------------------------------------------------- setup */

let worker: FakeWorker;
let cache: ReturnType<typeof fakeCache>;
let created: number;

function makeHost(over: Partial<HostEnv> = {}) {
  worker = new FakeWorker();
  cache = fakeCache();
  created = 0;
  const env: HostEnv = {
    createWorker: () => {
      created++;
      return worker.as();
    },
    readModel: async () => new ArrayBuffer(8),
    readAsset: async () => "{}",
    cache: cache.cache,
    numThreads: () => 4,
    wasmPaths: "ort/",
    ...over,
  };
  return new Host(env);
}

const request = (hash: string, priority = 0): ScoreRequest => ({
  hash,
  text: `text for ${hash}`,
  words: 200,
  priority,
});

/** Wait until the worker has been asked to run `n` chunks. */
const awaitRuns = (n: number) => vi.waitFor(() => expect(worker.runs.length).toBe(n));

function collector() {
  const messages: HostMessage[] = [];
  const reply = (m: HostMessage): void => void messages.push(m);
  return { messages, reply };
}

beforeEach(() => {
  vi.restoreAllMocks();
});

/* --------------------------------------------------------------------------- tests */

describe("startup", () => {
  it("reports the threading mode the worker observed", async () => {
    const host = makeHost();
    expect(await host.start()).toMatchObject({ numThreads: 4, crossOriginIsolated: true });
    expect(host.threading).toMatchObject({ numThreads: 4 });
  });

  it("boots at most once however many callers ask", async () => {
    const host = makeHost();
    await Promise.all([host.start(), host.start(), host.start()]);
    expect(created).toBe(1);
  });

  it("hands the worker the model bytes and both tokenizer assets", async () => {
    const assets: string[] = [];
    const host = makeHost({
      readAsset: async (name) => {
        assets.push(name);
        return "{}";
      },
    });
    await host.start();
    expect(assets.sort()).toEqual(["tokenizer.json", "tokenizer_config.json"]);
  });

  it("rejects when the worker cannot build a session", async () => {
    const host = makeHost();
    worker.failInit = "no available backend";
    await expect(host.start()).rejects.toThrow("no available backend");
  });
});

describe("the score cache short-circuit", () => {
  it("answers from cache without ever starting the worker", async () => {
    const host = makeHost();
    cache.store.set("h1", { logit: 2.5, words: 300, modelVersion: MODEL_VERSION, ts: 1 });
    const { messages, reply } = collector();

    await host.score(request("h1"), reply);
    expect(messages).toEqual([
      { type: "score", hash: "h1", logit: 2.5, words: 300, modelVersion: MODEL_VERSION },
    ]);
    expect(created).toBe(0);
  });

  it("ignores an entry written by another model version", async () => {
    const host = makeHost();
    cache.store.set("h1", { logit: 2.5, words: 300, modelVersion: "older", ts: 1 });
    const { reply } = collector();

    await host.score(request("h1"), reply);
    await awaitRuns(1);
    expect(worker.runs[0]!.text).toBe("text for h1");
  });

  it("writes a fresh score through to the cache", async () => {
    const host = makeHost();
    const { reply } = collector();
    await host.score(request("h1"), reply);
    await awaitRuns(1);
    worker.finish(worker.runs[0]!.id, 3.25);

    await vi.waitFor(() => expect(cache.store.get("h1")).toBeDefined());
    expect(cache.store.get("h1")).toMatchObject({
      logit: 3.25,
      words: 200,
      modelVersion: MODEL_VERSION,
    });
  });
});

describe("priority", () => {
  /**
   * Park the drain loop on one chunk so the queue can be filled deterministically. Without
   * a blocker the loop may pick before every request has landed, and the test would be
   * asserting on a race rather than on the ordering.
   */
  async function blocked(host: InstanceType<typeof Host>, reply: (m: HostMessage) => void) {
    await host.start();
    await host.score(request("blocker", 0), reply);
    await awaitRuns(1);
  }

  it("scores the most urgent chunk first", async () => {
    const host = makeHost();
    const { reply } = collector();
    await blocked(host, reply);

    await host.score(request("far", 5), reply);
    await host.score(request("near", 0), reply);
    await host.score(request("mid", 2), reply);
    worker.finish(worker.runs[0]!.id, 1);

    await awaitRuns(2);
    expect(worker.runs[1]!.text).toBe("text for near");
  });

  it("works down the queue in priority order", async () => {
    const host = makeHost();
    const { reply } = collector();
    await blocked(host, reply);

    await host.score(request("far", 5), reply);
    await host.score(request("near", 0), reply);
    await host.score(request("mid", 2), reply);

    const seen: string[] = [];
    for (let i = 0; i < 4; i++) {
      await awaitRuns(i + 1);
      seen.push(worker.runs[i]!.text);
      worker.finish(worker.runs[i]!.id, 1);
    }
    expect(seen).toEqual([
      "text for blocker",
      "text for near",
      "text for mid",
      "text for far",
    ]);
  });

  it("re-reads the queue between runs, so a scroll changes what comes next", async () => {
    const host = makeHost();
    const { reply } = collector();
    await host.start();

    await host.score(request("a", 1), reply);
    await host.score(request("b", 2), reply);
    await awaitRuns(1);
    expect(worker.runs[0]!.text).toBe("text for a");

    // The reader scrolls: something previously far away is now on screen.
    await host.score(request("c", 9), reply);
    host.reprioritize({ c: 0 });
    worker.finish(worker.runs[0]!.id, 1);

    await awaitRuns(2);
    expect(worker.runs[1]!.text).toBe("text for c");
  });

  it("reprioritize on an unknown hash is harmless", async () => {
    const host = makeHost();
    await host.start();
    expect(() => host.reprioritize({ nothing: 0 })).not.toThrow();
  });

  it("takes the most urgent priority when a chunk is asked for twice", async () => {
    const host = makeHost();
    const { reply } = collector();
    await blocked(host, reply);

    await host.score(request("x", 9), reply);
    await host.score(request("y", 4), reply);
    await host.score(request("x", 0), reply); // same hash, now on screen
    worker.finish(worker.runs[0]!.id, 1);

    await awaitRuns(2);
    expect(worker.runs[1]!.text).toBe("text for x");
  });
});

describe("de-duplication", () => {
  it("scores a hash once and answers every waiter", async () => {
    const host = makeHost();
    const first = collector();
    const second = collector();
    await host.start();

    await host.score(request("h1"), first.reply);
    await host.score(request("h1"), second.reply);
    await awaitRuns(1);
    worker.finish(worker.runs[0]!.id, 1.5);

    await vi.waitFor(() => expect(first.messages).toHaveLength(1));
    await vi.waitFor(() => expect(second.messages).toHaveLength(1));
    expect(worker.runs).toHaveLength(1);
  });

  it("joins a request that arrives while the chunk is already in flight", async () => {
    const host = makeHost();
    const first = collector();
    const late = collector();
    await host.start();

    await host.score(request("h1"), first.reply);
    await awaitRuns(1);
    await host.score(request("h1"), late.reply); // now in flight, not queued
    worker.finish(worker.runs[0]!.id, 2);

    await vi.waitFor(() => expect(late.messages).toHaveLength(1));
    expect(worker.runs).toHaveLength(1);
  });
});

describe("cancel and detach", () => {
  it("drops queued work once its last waiter cancels", async () => {
    const host = makeHost();
    const { reply } = collector();
    await host.start();

    await host.score(request("a", 0), reply);
    await host.score(request("b", 1), reply);
    await awaitRuns(1);

    host.cancel(["b"], reply);
    worker.finish(worker.runs[0]!.id, 1);
    await new Promise((r) => setTimeout(r, 10));
    expect(worker.runs).toHaveLength(1);
  });

  it("keeps queued work another client still wants", async () => {
    const host = makeHost();
    const one = collector();
    const two = collector();
    await host.start();

    await host.score(request("a", 0), one.reply);
    await host.score(request("b", 1), one.reply);
    await host.score(request("b", 1), two.reply);
    await awaitRuns(1);

    host.cancel(["b"], one.reply);
    worker.finish(worker.runs[0]!.id, 1);
    await awaitRuns(2);
    expect(worker.runs[1]!.text).toBe("text for b");
  });

  it("detach drops what only the departing client wanted", async () => {
    const host = makeHost();
    const gone = collector();
    const staying = collector();
    await host.start();

    await host.score(request("a", 0), staying.reply);
    await host.score(request("solo", 1), gone.reply);
    await host.score(request("shared", 1), gone.reply);
    await host.score(request("shared", 1), staying.reply);
    await awaitRuns(1);

    host.detach(gone.reply);
    worker.finish(worker.runs[0]!.id, 1);
    await awaitRuns(2);
    expect(worker.runs[1]!.text).toBe("text for shared");
  });

  it("a detached client is never called again", async () => {
    const host = makeHost();
    const gone = collector();
    await host.start();

    await host.score(request("h1"), gone.reply);
    await awaitRuns(1);
    host.detach(gone.reply);
    worker.finish(worker.runs[0]!.id, 1);

    await new Promise((r) => setTimeout(r, 10));
    expect(gone.messages).toEqual([]);
  });
});

describe("failure", () => {
  it("fails only the chunk the worker could not score", async () => {
    const host = makeHost();
    const { messages, reply } = collector();
    await host.start();

    await host.score(request("a", 0), reply);
    await host.score(request("b", 1), reply);
    await awaitRuns(1);
    worker.fail(worker.runs[0]!.id, "tokenizer blew up");

    await awaitRuns(2);
    worker.finish(worker.runs[1]!.id, 1.5);

    await vi.waitFor(() => expect(messages).toHaveLength(2));
    expect(messages[0]).toMatchObject({ type: "failed", hash: "a", message: "tokenizer blew up" });
    expect(messages[1]).toMatchObject({ type: "score", hash: "b" });
  });

  it("does not cache a failed chunk", async () => {
    const host = makeHost();
    const { reply } = collector();
    await host.start();
    await host.score(request("a"), reply);
    await awaitRuns(1);
    worker.fail(worker.runs[0]!.id, "nope");

    await new Promise((r) => setTimeout(r, 10));
    expect(cache.store.size).toBe(0);
  });

  it("fails every queued chunk when the session cannot start, rather than hanging", async () => {
    const host = makeHost();
    worker.failInit = "no available backend found";
    const { messages, reply } = collector();

    await host.score(request("a", 0), reply);
    await host.score(request("b", 1), reply);

    await vi.waitFor(() => expect(messages.length).toBeGreaterThan(0));
    expect(messages.every((m) => m.type === "failed")).toBe(true);
    expect(messages.map((m) => (m as { hash: string }).hash).sort()).toEqual(["a", "b"]);
  });

  it("ignores an unattributable worker error instead of crashing the queue", async () => {
    const host = makeHost();
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const { reply } = collector();
    await host.start();
    await host.score(request("a"), reply);
    await awaitRuns(1);

    expect(() => worker.emit({ type: "error", message: "detached failure" })).not.toThrow();
    worker.finish(worker.runs[0]!.id, 1);
    await vi.waitFor(() => expect(cache.store.get("a")).toBeDefined());
  });
});

describe("dispose", () => {
  it("terminates the worker and lets a later start build a new one", async () => {
    const host = makeHost();
    await host.start();
    await host.dispose();
    expect(worker.terminated).toBe(true);

    await host.start();
    expect(created).toBe(2);
  });

  it("is safe before anything started", async () => {
    await expect(makeHost().dispose()).resolves.toBeUndefined();
  });
});
