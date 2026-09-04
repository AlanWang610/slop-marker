/**
 * The part of the router that is not browser-specific.
 *
 * Both browsers run this same protocol handler in whichever context owns the Host: the
 * offscreen document on Chrome, the event page on Firefox (scope.md 6.2). What differs is
 * only *who is allowed to fetch*, which is the `ensureModel` injected below.
 *
 * On Chrome the Host lives in an offscreen document, and extension pages are cross-origin
 * isolated by the manifest COOP/COEP keys so ORT can use threads. Under require-corp a
 * cross-origin fetch needs a CORS-successful response or a CORP header, and GitHub release
 * assets send neither -- so the offscreen document delegates the download to the service
 * worker and reads the bytes back out of the Cache API, which is per-origin and shared.
 * On Firefox COOP/COEP are ignored and the event page just downloads.
 */

import type { Host } from "../host/host.js";
import * as modelStore from "../host/model-store.js";
import * as scoreCache from "../host/score-cache.js";
import { MODEL_VERSION } from "../shared/bundle-config.js";
import type { ContentMessage, HostMessage, ModelState } from "../shared/protocol.js";
import * as storage from "../shared/storage.js";

export interface RouterEnv {
  /** Download and verify the bundle. Must run somewhere COEP does not apply. */
  ensureModel(): Promise<void>;
  /** Lazily construct the Host. Called at most once. */
  createHost(): Host;
  /** Ground truth for "is the model here". Defaults to the Cache API. */
  isCached?: (version: string) => Promise<boolean>;
  /** Where the model state is mirrored for the pages to read. Defaults to storage.local. */
  setStorage?: (items: Record<string, unknown>) => Promise<void>;
}

export class Router {
  #state: ModelState = { phase: "absent" };
  #host: Host | null = null;
  #downloading: Promise<void> | null = null;
  readonly #listeners = new Set<(state: ModelState) => void>();

  constructor(private readonly env: RouterEnv) {}

  get modelState(): ModelState {
    return this.#state;
  }

  #setState(state: ModelState): void {
    // Identical state is not an event. Without this, every `hello` answers `model` twice --
    // refresh() notifies the listener that attachPort just added, and then attachPort
    // replies again -- and every no-op refresh wakes every open tab.
    if (JSON.stringify(state) === JSON.stringify(this.#state)) return;
    this.#state = state;
    const write = this.env.setStorage ?? ((items) => storage.set(items));
    void write({ modelState: state }).catch(() => undefined);
    for (const listener of this.#listeners) listener(state);
  }

  /**
   * Reflect what is already in the Cache API, without fetching.
   *
   * `force` is required after a download completes. Otherwise the guard below -- which is
   * there so a concurrent refresh cannot wipe out live progress -- sees the `downloading`
   * state that `ensureModel` itself set, returns early, and the Router stays `downloading`
   * forever with the verified model sitting on disk. Every content script that connects
   * then gets `phase: "downloading"` and is never sent a calibration, so the page is silently
   * never scored.
   */
  async refresh(force = false): Promise<void> {
    if (!force && (this.#state.phase === "downloading" || this.#state.phase === "verifying")) {
      return;
    }
    const isCached = this.env.isCached ?? modelStore.isCached;
    if (await isCached(MODEL_VERSION)) {
      this.#setState({ phase: "ready", version: MODEL_VERSION });
    } else if (this.#state.phase !== "error") {
      this.#setState({ phase: "absent" });
    }
  }

  async ensureModel(): Promise<void> {
    await this.refresh();
    if (this.#state.phase === "ready") return;
    this.#downloading ??= (async () => {
      try {
        this.#setState({ phase: "downloading", received: 0, total: 0 });
        await this.env.ensureModel();
        await this.refresh(true);
      } catch (err) {
        this.#setState({
          phase: "error",
          message: err instanceof Error ? err.message : String(err),
        });
        throw err;
      } finally {
        this.#downloading = null;
      }
    })();
    return this.#downloading;
  }

  /** Drop the session, e.g. after the model cache is cleared under us. */
  async disposeHost(): Promise<void> {
    await this.#host?.dispose();
    this.#host = null;
  }

  #getHost(): Host {
    this.#host ??= this.env.createHost();
    return this.#host;
  }

  /**
   * Wire one content-script port. Identical on both browsers: the content script reconnects
   * on disconnect and re-sends outstanding hashes, which covers a Chrome service worker
   * restart and a Firefox event-page unload with the same code (scope.md 7.4).
   */
  attachPort(port: chrome.runtime.Port): void {
    const reply = (message: HostMessage): void => {
      try {
        port.postMessage(message);
      } catch {
        // Port closed under us; the content script reconnects and re-asks.
      }
    };

    // One `model` message per actual change, per port. `hello` refreshes and then reports
    // the state, and the refresh may itself have pushed the same state through the listener
    // a moment earlier -- so without this every connect answers `model` twice.
    let lastSent: string | null = null;
    const sendState = (state: ModelState): void => {
      const encoded = JSON.stringify(state);
      if (encoded === lastSent) return;
      lastSent = encoded;
      reply({ type: "model", state });
    };

    const listener = (state: ModelState): void => sendState(state);
    this.#listeners.add(listener);

    port.onMessage.addListener((raw: unknown) => {
      const message = raw as ContentMessage;
      void (async () => {
        try {
          switch (message.type) {
            case "hello": {
              await this.refresh();
              sendState(this.#state);
              if (this.#state.phase !== "ready") return;
              const host = this.#getHost();
              const threading = await host.start();
              reply({ type: "ready", calibration: host.calibration, threading });
              return;
            }
            case "score": {
              if (this.#state.phase !== "ready") return;
              const host = this.#getHost();
              for (const item of message.items) await host.score(item, reply);
              return;
            }
            case "reprioritize":
              this.#host?.reprioritize(message.priorities);
              return;
            case "cancel":
              this.#host?.cancel(message.hashes, reply);
              return;
            case "keepalive":
              // Port traffic alone resets the Firefox event-page idle timer (scope.md 6.2).
              return;
          }
        } catch (err) {
          // Also recorded, not just logged. This runs in the Chrome offscreen document,
          // whose console no devtools window shows by default -- so a Host that fails to
          // start looks exactly like a page with no AI text on it. The options page reads
          // this back.
          const detail = err instanceof Error ? (err.stack ?? err.message) : String(err);
          console.error("slop-marker router:", err);
          const write = this.env.setStorage ?? ((items) => storage.set(items));
          void write({ lastError: { at: Date.now(), where: message.type, detail } }).catch(
            () => undefined,
          );
        }
      })();
    });

    port.onDisconnect.addListener(() => {
      this.#listeners.delete(listener);
      this.#host?.detach(reply);
    });
  }

}

/** Download directly. Used where COEP does not constrain fetching. */
export async function downloadHere(onState?: (state: ModelState) => void): Promise<void> {
  await modelStore.download(MODEL_VERSION, onState ?? (() => undefined));
}

/**
 * Options- and first-run-page commands, answered by whichever context may fetch: the
 * service worker on Chrome, the event page on Firefox.
 *
 * Deliberately independent of the Host. On Chrome the Host lives in an offscreen document
 * that is created lazily, so routing status through it meant the first page to ask got no
 * answer at all -- sendMessage reaches every extension context, and if none of them handles
 * the message the promise resolves to undefined. Everything here reads storage.local and
 * the Cache API, which any context can do.
 */
export interface CommandDeps {
  isCached(version: string): Promise<boolean>;
  download(version: string, onState: (state: ModelState) => void): Promise<void>;
  clearModelCache(): Promise<void>;
  clearScores(): Promise<void>;
  countScores(): Promise<number>;
  storageGet(keys: string[]): Promise<Record<string, unknown>>;
  storageSet(items: Record<string, unknown>): Promise<void>;
}

/** The real browser wiring. Substituted only by tests. */
export const defaultCommandDeps: CommandDeps = {
  isCached: (version) => modelStore.isCached(version),
  download: (version, onState) => modelStore.download(version, onState),
  clearModelCache: () => modelStore.clearCache(),
  clearScores: () => scoreCache.clear(),
  countScores: () => scoreCache.size(),
  storageGet: (keys) => chrome.storage.local.get(keys),
  storageSet: (items) => chrome.storage.local.set(items),
};

export function handleCommand(
  message: { type: string },
  sendResponse: (response: unknown) => void,
  onModelCleared?: () => void,
  deps: CommandDeps = defaultCommandDeps,
): boolean {
  switch (message.type) {
    case "getStatus":
      void (async () => {
        const cached = await deps.isCached(MODEL_VERSION);
        const stored = await deps.storageGet(["modelState", "threading"]);
        const previous = stored["modelState"] as ModelState | undefined;
        // The Cache API is the ground truth; a stored error survives until a retry clears it.
        const state: ModelState = cached
          ? { phase: "ready", version: MODEL_VERSION }
          : (previous?.phase === "downloading" ||
              previous?.phase === "verifying" ||
              previous?.phase === "error"
              ? previous
              : { phase: "absent" });
        sendResponse({
          modelState: state,
          modelVersion: MODEL_VERSION,
          cachedScores: await deps.countScores(),
          threading: stored["threading"] ?? null,
        });
      })();
      return true;

    case "downloadModel":
      void (async () => {
        try {
          await deps.download(MODEL_VERSION, (state) => {
            void deps.storageSet({ modelState: state });
          });
          sendResponse({ ok: true });
        } catch (err) {
          const detail = err instanceof Error ? err.message : String(err);
          await deps.storageSet({ modelState: { phase: "error", message: detail } });
          sendResponse({ ok: false, message: detail });
        }
      })();
      return true;

    case "clearModel":
      void (async () => {
        await deps.clearModelCache();
        await deps.clearScores();
        await deps.storageSet({ modelState: { phase: "absent" } });
        onModelCleared?.();
        sendResponse({ ok: true });
      })();
      return true;

    case "clearScores":
      void deps.clearScores().then(() => sendResponse({ ok: true }));
      return true;

    default:
      return false;
  }
}
