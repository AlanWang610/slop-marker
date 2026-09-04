/**
 * The port protocol and the page commands (scope.md 6.2).
 *
 * Both browsers run this same handler; only *who is allowed to fetch* differs, and that is
 * the `RouterEnv` injected here. So driving it with fakes covers the Chrome offscreen
 * arrangement and the Firefox event page at once.
 *
 * The error-recording case at the bottom is not incidental. This code runs in a Chrome
 * offscreen document, whose console no devtools window shows by default -- a host that
 * fails to start looks exactly like a page with no AI text on it. The options page can only
 * report what got written to storage.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Host } from "../src/host/host.js";
import {
  handleCommand,
  Router,
  type CommandDeps,
  type RouterEnv,
} from "../src/router/shared.js";
import { MODEL_VERSION } from "../src/shared/bundle-config.js";
import type { ContentMessage, HostMessage, ModelState } from "../src/shared/protocol.js";

/* ----------------------------------------------------------------------- fake port */

class FakePort {
  readonly posted: HostMessage[] = [];
  readonly name = "slop-marker";
  #messageListeners = new Set<(message: unknown) => void>();
  #disconnectListeners = new Set<() => void>();

  readonly onMessage = { addListener: (fn: (m: unknown) => void) => this.#messageListeners.add(fn) };
  readonly onDisconnect = { addListener: (fn: () => void) => this.#disconnectListeners.add(fn) };

  postMessage(message: HostMessage): void {
    this.posted.push(message);
  }
  disconnect(): void {
    for (const fn of this.#disconnectListeners) fn();
  }
  /** Simulate the content script sending something. */
  send(message: ContentMessage): void {
    for (const fn of this.#messageListeners) fn(message);
  }
  types(): string[] {
    return this.posted.map((m) => m.type);
  }
  as(): chrome.runtime.Port {
    return this as unknown as chrome.runtime.Port;
  }
}

/* ----------------------------------------------------------------------- fake host */

function fakeHost() {
  const calls = {
    start: 0,
    scored: [] as string[],
    reprioritized: [] as Record<string, number>[],
    cancelled: [] as string[][],
    detached: 0,
    disposed: 0,
  };
  const host = {
    calibration: { version: MODEL_VERSION, min_words: 40 },
    start: async () => {
      calls.start++;
      return { crossOriginIsolated: true, numThreads: 4, hardwareConcurrency: 8 };
    },
    score: async (request: { hash: string }, reply: (m: HostMessage) => void) => {
      calls.scored.push(request.hash);
      reply({ type: "score", hash: request.hash, logit: 1.5, words: 200, modelVersion: MODEL_VERSION });
    },
    reprioritize: (p: Record<string, number>) => calls.reprioritized.push(p),
    cancel: (hashes: readonly string[]) => calls.cancelled.push([...hashes]),
    detach: () => calls.detached++,
    dispose: async () => void calls.disposed++,
  };
  return { calls, host: host as unknown as Host };
}

/* ---------------------------------------------------------------------- fake env */

function env(over: Partial<RouterEnv> & { cached?: boolean } = {}) {
  const stored: Record<string, unknown> = {};
  const created = { count: 0 };
  const { host, calls } = fakeHost();
  const base: RouterEnv = {
    ensureModel: async () => undefined,
    createHost: () => {
      created.count++;
      return host;
    },
    isCached: async () => over.cached ?? false,
    setStorage: async (items) => void Object.assign(stored, items),
    ...over,
  };
  return { env: base, stored, created, hostCalls: calls };
}

const flush = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

/* --------------------------------------------------------------------------- state */

describe("model state", () => {
  it("starts absent", () => {
    expect(new Router(env().env).modelState).toEqual({ phase: "absent" });
  });

  it("refresh reports ready when the bytes are cached", async () => {
    const router = new Router(env({ cached: true }).env);
    await router.refresh();
    expect(router.modelState).toEqual({ phase: "ready", version: MODEL_VERSION });
  });

  it("refresh reports absent when they are not", async () => {
    const router = new Router(env({ cached: false }).env);
    await router.refresh();
    expect(router.modelState).toEqual({ phase: "absent" });
  });

  it("mirrors state into storage for the pages to read", async () => {
    const e = env({ cached: true });
    await new Router(e.env).refresh();
    expect(e.stored["modelState"]).toEqual({ phase: "ready", version: MODEL_VERSION });
  });

  it("does not overwrite an error with absent, so a failure survives to be shown", async () => {
    const e = env({ cached: false, ensureModel: () => Promise.reject(new Error("boom")) });
    const router = new Router(e.env);
    await expect(router.ensureModel()).rejects.toThrow("boom");
    expect(router.modelState).toMatchObject({ phase: "error", message: "boom" });

    await router.refresh();
    expect(router.modelState).toMatchObject({ phase: "error" });
  });
});

describe("ensureModel", () => {
  it("does nothing when the model is already there", async () => {
    const ensureModel = vi.fn(async () => undefined);
    const router = new Router(env({ cached: true, ensureModel }).env);
    await router.ensureModel();
    expect(ensureModel).not.toHaveBeenCalled();
  });

  it("de-duplicates concurrent callers into one download", async () => {
    let cached = false;
    const ensureModel = vi.fn(async () => {
      await flush();
      cached = true;
    });
    const router = new Router({ ...env({ ensureModel }).env, isCached: async () => cached });

    await Promise.all([router.ensureModel(), router.ensureModel(), router.ensureModel()]);
    expect(ensureModel).toHaveBeenCalledTimes(1);
    expect(router.modelState).toMatchObject({ phase: "ready" });
  });

  it("records the failure and rethrows, rather than swallowing it", async () => {
    const e = env({ ensureModel: () => Promise.reject(new Error("no network")) });
    const router = new Router(e.env);
    await expect(router.ensureModel()).rejects.toThrow("no network");
    expect(e.stored["modelState"]).toMatchObject({ phase: "error", message: "no network" });
  });

  it("can be retried after a failure", async () => {
    let fail = true;
    let cached = false;
    const router = new Router({
      ...env().env,
      isCached: async () => cached,
      ensureModel: async () => {
        if (fail) throw new Error("first attempt");
        cached = true;
      },
    });
    await expect(router.ensureModel()).rejects.toThrow();
    fail = false;
    await router.ensureModel();
    expect(router.modelState).toMatchObject({ phase: "ready" });
  });
});

/* ---------------------------------------------------------------------- attachPort */

describe("attachPort", () => {
  let port: FakePort;
  beforeEach(() => {
    port = new FakePort();
  });

  it("does not repeat an unchanged state", async () => {
    const router = new Router(env({ cached: true }).env);
    router.attachPort(port.as());
    await router.refresh();
    await router.refresh();
    await router.refresh();
    expect(port.types()).toEqual(["model"]);
  });

  it("answers hello with the model state and nothing else while absent", async () => {
    const router = new Router(env({ cached: false }).env);
    router.attachPort(port.as());
    port.send({ type: "hello" });
    await flush();

    expect(port.types()).toEqual(["model"]);
    expect(port.posted[0]).toMatchObject({ state: { phase: "absent" } });
  });

  it("answers hello with the calibration once the model is ready", async () => {
    const router = new Router(env({ cached: true }).env);
    router.attachPort(port.as());
    port.send({ type: "hello" });
    await flush();

    expect(port.types()).toEqual(["model", "ready"]);
    expect(port.posted[1]).toMatchObject({
      calibration: { version: MODEL_VERSION },
      threading: { numThreads: 4 },
    });
  });

  it("creates the host at most once across several ports", async () => {
    const e = env({ cached: true });
    const router = new Router(e.env);
    for (const p of [new FakePort(), new FakePort(), new FakePort()]) {
      router.attachPort(p.as());
      p.send({ type: "hello" });
    }
    await flush();
    expect(e.created.count).toBe(1);
  });

  it("drops score requests that arrive before the model is ready", async () => {
    const e = env({ cached: false });
    const router = new Router(e.env);
    router.attachPort(port.as());
    port.send({ type: "score", items: [{ hash: "h1", text: "t", words: 200, priority: 0 }] });
    await flush();

    expect(e.hostCalls.scored).toEqual([]);
    expect(port.types()).toEqual([]);
  });

  it("scores every item in a batch and replies per chunk", async () => {
    const e = env({ cached: true });
    const router = new Router(e.env);
    router.attachPort(port.as());
    port.send({ type: "hello" });
    await flush();
    port.send({
      type: "score",
      items: [
        { hash: "h1", text: "a", words: 200, priority: 0 },
        { hash: "h2", text: "b", words: 200, priority: 1 },
      ],
    });
    await flush();

    expect(e.hostCalls.scored).toEqual(["h1", "h2"]);
    expect(port.posted.filter((m) => m.type === "score")).toHaveLength(2);
  });

  it("forwards reprioritize to the host", async () => {
    const e = env({ cached: true });
    const router = new Router(e.env);
    router.attachPort(port.as());
    port.send({ type: "hello" });
    await flush();
    port.send({ type: "reprioritize", priorities: { h1: 0 } });
    await flush();

    expect(e.hostCalls.reprioritized).toEqual([{ h1: 0 }]);
  });

  it("forwards cancel to the host", async () => {
    const e = env({ cached: true });
    const router = new Router(e.env);
    router.attachPort(port.as());
    port.send({ type: "hello" });
    await flush();
    port.send({ type: "cancel", hashes: ["h1", "h2"] });
    await flush();

    expect(e.hostCalls.cancelled).toEqual([["h1", "h2"]]);
  });

  it("treats keepalive as traffic and nothing more", async () => {
    const e = env({ cached: true });
    const router = new Router(e.env);
    router.attachPort(port.as());
    port.send({ type: "hello" });
    await flush();
    const before = port.posted.length;
    port.send({ type: "keepalive" });
    await flush();

    expect(port.posted).toHaveLength(before);
    expect(e.hostCalls.scored).toEqual([]);
  });

  it("pushes model-state changes to every attached port", async () => {
    let cached = false;
    const router = new Router({ ...env().env, isCached: async () => cached });
    const a = new FakePort();
    const b = new FakePort();
    router.attachPort(a.as());
    router.attachPort(b.as());

    cached = true;
    await router.refresh();
    expect(a.types()).toContain("model");
    expect(b.types()).toContain("model");
  });

  it("detaches the client's queued work on disconnect", async () => {
    const e = env({ cached: true });
    const router = new Router(e.env);
    router.attachPort(port.as());
    port.send({ type: "hello" });
    await flush();
    port.disconnect();

    expect(e.hostCalls.detached).toBe(1);
  });

  it("stops pushing state to a disconnected port", async () => {
    let cached = false;
    const router = new Router({ ...env().env, isCached: async () => cached });
    router.attachPort(port.as());
    port.disconnect();

    cached = true;
    await router.refresh();
    expect(port.types()).toEqual([]);
  });

  it("survives a port that throws on postMessage", async () => {
    const router = new Router(env({ cached: true }).env);
    const broken = new FakePort();
    broken.postMessage = () => {
      throw new Error("port closed");
    };
    router.attachPort(broken.as());
    expect(() => broken.send({ type: "hello" })).not.toThrow();
    await flush();
  });

  it("records a handler failure where the options page can find it", async () => {
    const e = env({
      cached: true,
      createHost: () =>
        ({
          calibration: {},
          start: () => Promise.reject(new Error("session would not start")),
          detach: () => undefined,
          dispose: async () => undefined,
        }) as unknown as Host,
    });
    const router = new Router(e.env);
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    router.attachPort(port.as());
    port.send({ type: "hello" });
    await flush();

    expect(e.stored["lastError"]).toMatchObject({ where: "hello" });
    expect(String((e.stored["lastError"] as { detail: string }).detail)).toContain(
      "session would not start",
    );
    vi.restoreAllMocks();
  });
});

describe("disposeHost", () => {
  it("disposes and forgets the host, so the next port builds a fresh one", async () => {
    const e = env({ cached: true });
    const router = new Router(e.env);
    const port = new FakePort();
    router.attachPort(port.as());
    port.send({ type: "hello" });
    await flush();

    await router.disposeHost();
    const second = new FakePort();
    router.attachPort(second.as());
    second.send({ type: "hello" });
    await flush();

    expect(e.created.count).toBe(2);
  });

  it("is safe when no host was ever created", async () => {
    await expect(new Router(env().env).disposeHost()).resolves.toBeUndefined();
  });
});

/* ------------------------------------------------------------------- handleCommand */

describe("handleCommand", () => {
  function deps(over: Partial<CommandDeps> = {}) {
    const stored: Record<string, unknown> = {};
    const calls = { downloads: 0, clearedModel: 0, clearedScores: 0 };
    const base: CommandDeps = {
      isCached: async () => false,
      download: async () => void calls.downloads++,
      clearModelCache: async () => void calls.clearedModel++,
      clearScores: async () => void calls.clearedScores++,
      countScores: async () => 1234,
      storageGet: async (keys) => Object.fromEntries(keys.map((k) => [k, stored[k]])),
      storageSet: async (items) => void Object.assign(stored, items),
      ...over,
    };
    return { deps: base, stored, calls };
  }

  const answer = (message: { type: string }, d: CommandDeps) =>
    new Promise<Record<string, unknown>>((resolve) => {
      handleCommand(message, (r) => resolve(r as Record<string, unknown>), undefined, d);
    });

  it("returns false for a message it does not own", () => {
    expect(handleCommand({ type: "storage" }, vi.fn(), undefined, deps().deps)).toBe(false);
  });

  it("getStatus trusts the Cache API over the stored state", async () => {
    const d = deps({ isCached: async () => true });
    d.stored["modelState"] = { phase: "error", message: "stale" };
    const status = await answer({ type: "getStatus" }, d.deps);
    expect(status["modelState"]).toEqual({ phase: "ready", version: MODEL_VERSION });
  });

  it("getStatus keeps a stored error when nothing is cached", async () => {
    const d = deps();
    d.stored["modelState"] = { phase: "error", message: "checksum mismatch" };
    const status = await answer({ type: "getStatus" }, d.deps);
    expect(status["modelState"]).toMatchObject({ phase: "error" });
  });

  it("getStatus reports absent when there is no history", async () => {
    const status = await answer({ type: "getStatus" }, deps().deps);
    expect(status["modelState"]).toEqual({ phase: "absent" });
  });

  it("getStatus reports the score count and threading", async () => {
    const d = deps();
    d.stored["threading"] = { numThreads: 4 };
    const status = await answer({ type: "getStatus" }, d.deps);
    expect(status["cachedScores"]).toBe(1234);
    expect(status["threading"]).toEqual({ numThreads: 4 });
    expect(status["modelVersion"]).toBe(MODEL_VERSION);
  });

  it("downloadModel reports success", async () => {
    const d = deps();
    expect(await answer({ type: "downloadModel" }, d.deps)).toEqual({ ok: true });
    expect(d.calls.downloads).toBe(1);
  });

  it("downloadModel mirrors progress into storage", async () => {
    const states: ModelState[] = [
      { phase: "downloading", received: 10, total: 100 },
      { phase: "ready", version: MODEL_VERSION },
    ];
    const d = deps({
      download: async (_v, onState) => {
        for (const s of states) onState(s);
      },
    });
    await answer({ type: "downloadModel" }, d.deps);
    expect(d.stored["modelState"]).toEqual({ phase: "ready", version: MODEL_VERSION });
  });

  it("downloadModel reports a checksum failure back to the page", async () => {
    const d = deps({
      download: () => Promise.reject(new Error("model.onnx: checksum mismatch")),
    });
    const response = await answer({ type: "downloadModel" }, d.deps);
    expect(response).toEqual({ ok: false, message: "model.onnx: checksum mismatch" });
    expect(d.stored["modelState"]).toMatchObject({ phase: "error" });
  });

  it("clearModel drops the bytes, the scores and the state", async () => {
    const d = deps();
    const cleared = vi.fn();
    await new Promise((resolve) => {
      handleCommand({ type: "clearModel" }, resolve, cleared, d.deps);
    });
    expect(d.calls.clearedModel).toBe(1);
    expect(d.calls.clearedScores).toBe(1);
    expect(d.stored["modelState"]).toEqual({ phase: "absent" });
    expect(cleared).toHaveBeenCalled();
  });

  it("clearScores leaves the model alone", async () => {
    const d = deps();
    expect(await answer({ type: "clearScores" }, d.deps)).toEqual({ ok: true });
    expect(d.calls.clearedScores).toBe(1);
    expect(d.calls.clearedModel).toBe(0);
  });
});
