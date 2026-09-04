/**
 * The inference host (scope.md 6.3): score cache, priority queue, one dedicated worker.
 *
 * Shared between browsers. Chrome instantiates it inside an offscreen document so it
 * survives service-worker restarts; Firefox instantiates it directly in the event page.
 * Neither router knows what is in here, and this file knows nothing about either.
 *
 * It never downloads. The router does that -- see model-store.ts on why COEP forces it --
 * and the host reads the verified bytes back out of the Cache API.
 */

import { MODEL_VERSION, SHIPPED_CALIBRATION_JSON } from "../shared/bundle-config.js";
import { type Calibration, parseCalibration } from "../shared/calibration.js";
import type {
  HostMessage,
  ScoreRequest,
  ThreadingInfo,
  WorkerRequest,
  WorkerResponse,
} from "../shared/protocol.js";
import * as storage from "../shared/storage.js";
import { readCached } from "./model-store.js";
import * as scoreCache from "./score-cache.js";
import type { ScoreCache } from "./score-cache.js";

/**
 * ORT threading (scope.md 6.3). Threads need cross-origin isolation, which Chrome should
 * get from the manifest COOP/COEP keys and Firefox may not. Measured, not assumed: it is
 * worth 3.6x on this model (docs/measurements/extension-latency.md).
 */
function chooseThreads(): number {
  if (!self.crossOriginIsolated) return 1;
  return Math.min(4, navigator.hardwareConcurrency || 1);
}

type Reply = (message: HostMessage) => void;

/**
 * Everything the Host reaches outside itself. Injected rather than imported so the queue --
 * the part with real ordering logic, and the part scope.md 7.4's viewport-first promise
 * rests on -- can be driven deterministically by a test with a fake worker and an in-memory
 * cache, instead of only through a browser.
 */
export interface HostEnv {
  createWorker(): Worker;
  /** The verified model bytes, from wherever the router put them. */
  readModel(): Promise<ArrayBuffer>;
  /** A packaged text asset, by name under assets/. */
  readAsset(name: string): Promise<string>;
  readonly cache: ScoreCache;
  numThreads(): number;
  readonly wasmPaths: string;
}

/** What both routers use in a real browser. */
export function defaultHostEnv(workerUrl: string, wasmPaths: string): HostEnv {
  return {
    createWorker: () => new Worker(workerUrl, { type: "module" }),
    readModel: () => readCached(MODEL_VERSION, "model.onnx"),
    readAsset: (name) => fetchText(chrome.runtime.getURL(`assets/${name}`)),
    cache: scoreCache,
    numThreads: chooseThreads,
    wasmPaths,
  };
}

interface Pending {
  readonly request: ScoreRequest;
  priority: number;
  /** Every client that asked for this hash while it was outstanding. */
  readonly waiters: Set<Reply>;
}

export class Host {
  readonly calibration: Calibration = parseCalibration(SHIPPED_CALIBRATION_JSON);

  #worker: Worker | null = null;
  #ready: Promise<ThreadingInfo> | null = null;
  #threading: ThreadingInfo | null = null;

  /** Queued but not yet sent to the worker, keyed by hash so duplicates collapse. */
  readonly #queue = new Map<string, Pending>();
  /** Sent to the worker and awaiting a result. */
  readonly #inFlight = new Map<number, Pending>();
  /** Resolvers that let the drain loop wait for exactly one run to finish. */
  readonly #settle = new Map<number, () => void>();
  #nextId = 1;
  #draining = false;

  constructor(private readonly env: HostEnv) {}

  get threading(): ThreadingInfo | null {
    return this.#threading;
  }

  /** Idempotent. Creates the session from cached, already-verified bytes. */
  async start(): Promise<ThreadingInfo> {
    this.#ready ??= this.#boot();
    return this.#ready;
  }

  async #boot(): Promise<ThreadingInfo> {
    const removed = await this.env.cache.evictOtherVersions(MODEL_VERSION);
    if (removed > 0) {
      console.info(`slop-marker: dropped ${removed} scores from an older model`);
    }

    const [model, tokenizerJson, tokenizerConfigJson] = await Promise.all([
      this.env.readModel(),
      this.env.readAsset("tokenizer.json"),
      this.env.readAsset("tokenizer_config.json"),
    ]);

    const worker = this.env.createWorker();
    this.#worker = worker;

    const threading = await new Promise<ThreadingInfo>((resolve, reject) => {
      const onFirst = (event: MessageEvent<WorkerResponse>): void => {
        const message = event.data;
        if (message.type === "ready") {
          worker.removeEventListener("message", onFirst);
          worker.addEventListener("message", this.#onWorkerMessage);
          resolve(message.threading);
        } else if (message.type === "error") {
          worker.removeEventListener("message", onFirst);
          reject(new Error(message.message));
        }
      };
      worker.addEventListener("message", onFirst);

      const init: WorkerRequest = {
        type: "init",
        model,
        tokenizerJson,
        tokenizerConfigJson,
        maxLength: this.calibration.max_length,
        numThreads: this.env.numThreads(),
        wasmPaths: this.env.wasmPaths,
      };
      // Transfer the 137 MB buffer rather than structured-cloning it.
      worker.postMessage(init, [model]);
    });

    this.#threading = threading;
    // Via the shim: in a Chrome offscreen document chrome.storage does not exist.
    await storage.set({ threading });
    return threading;
  }

  /**
   * Score one chunk. Resolves from cache without waking the worker when it can, which is
   * the difference between a revisited page being instant and being re-scored.
   */
  async score(request: ScoreRequest, reply: Reply): Promise<void> {
    const hit = await this.env.cache.get(request.hash, MODEL_VERSION);
    if (hit !== null) {
      reply({
        type: "score",
        hash: request.hash,
        logit: hit.logit,
        words: hit.words,
        modelVersion: MODEL_VERSION,
      });
      return;
    }

    const queued = this.#queue.get(request.hash);
    if (queued !== undefined) {
      queued.waiters.add(reply);
      queued.priority = Math.min(queued.priority, request.priority);
      return;
    }
    for (const pending of this.#inFlight.values()) {
      if (pending.request.hash === request.hash) {
        pending.waiters.add(reply);
        return;
      }
    }

    this.#queue.set(request.hash, {
      request,
      priority: request.priority,
      waiters: new Set([reply]),
    });
    void this.#drain();
  }

  reprioritize(priorities: Readonly<Record<string, number>>): void {
    for (const [hash, priority] of Object.entries(priorities)) {
      const pending = this.#queue.get(hash);
      if (pending !== undefined) pending.priority = priority;
    }
  }

  /** Drop queued work for a client that went away. In-flight work is left to finish. */
  cancel(hashes: readonly string[], reply: Reply): void {
    for (const hash of hashes) {
      const pending = this.#queue.get(hash);
      if (pending === undefined) continue;
      pending.waiters.delete(reply);
      if (pending.waiters.size === 0) this.#queue.delete(hash);
    }
  }

  /** Forget a disconnected client entirely, dropping anything only it wanted. */
  detach(reply: Reply): void {
    for (const [hash, pending] of this.#queue) {
      pending.waiters.delete(reply);
      if (pending.waiters.size === 0) this.#queue.delete(hash);
    }
    for (const pending of this.#inFlight.values()) pending.waiters.delete(reply);
  }

  /**
   * One chunk at a time (batch 1 per run(), scope.md 6.3), always the most urgent queued.
   * Re-reading the queue between runs is what makes viewport-first real: a scroll can
   * change which chunk matters most while the previous one is still being scored.
   */
  async #drain(): Promise<void> {
    if (this.#draining) return;
    this.#draining = true;
    try {
      await this.start();
      while (this.#queue.size > 0) {
        let best: Pending | null = null;
        for (const pending of this.#queue.values()) {
          if (best === null || pending.priority < best.priority) best = pending;
        }
        if (best === null) break;
        this.#queue.delete(best.request.hash);

        const id = this.#nextId++;
        this.#inFlight.set(id, best);
        const run: WorkerRequest = { type: "run", id, text: best.request.text };
        this.#worker?.postMessage(run);
        await new Promise<void>((resolve) => this.#settle.set(id, resolve));
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      for (const pending of this.#queue.values()) {
        for (const waiter of pending.waiters) {
          waiter({ type: "failed", hash: pending.request.hash, message });
        }
      }
      this.#queue.clear();
    } finally {
      this.#draining = false;
    }
  }

  readonly #onWorkerMessage = (event: MessageEvent<WorkerResponse>): void => {
    const message = event.data;
    if (message.type === "ready") return;

    if (message.type === "error" && message.id === undefined) {
      // An init-time or otherwise unattributable failure; no waiter to tell.
      console.error(`slop-marker worker: ${message.message}`);
      return;
    }
    const id = message.id!;
    const pending = this.#inFlight.get(id);
    this.#inFlight.delete(id);
    this.#settle.get(id)?.();
    this.#settle.delete(id);
    if (pending === undefined) return;

    if (message.type === "result") {
      void this.env.cache.put(pending.request.hash, {
        logit: message.logit,
        words: pending.request.words,
        modelVersion: MODEL_VERSION,
        ts: Date.now(),
      });
      for (const waiter of pending.waiters) {
        waiter({
          type: "score",
          hash: pending.request.hash,
          logit: message.logit,
          words: pending.request.words,
          modelVersion: MODEL_VERSION,
        });
      }
    } else {
      for (const waiter of pending.waiters) {
        waiter({ type: "failed", hash: pending.request.hash, message: message.message });
      }
    }
  };

  async dispose(): Promise<void> {
    this.#worker?.postMessage({ type: "close" } satisfies WorkerRequest);
    this.#worker?.terminate();
    this.#worker = null;
    this.#ready = null;
    await Promise.resolve();
  }
}

async function fetchText(url: string): Promise<string> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.text();
}
