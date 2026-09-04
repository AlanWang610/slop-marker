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

export interface RouterEnv {
  /** Download and verify the bundle. Must run somewhere COEP does not apply. */
  ensureModel(): Promise<void>;
  /** Lazily construct the Host. Called at most once. */
  createHost(): Host;
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
    this.#state = state;
    void chrome.storage.local.set({ modelState: state });
    for (const listener of this.#listeners) listener(state);
  }

  /** Reflect what is already in the Cache API, without fetching. */
  async refresh(): Promise<void> {
    if (this.#state.phase === "downloading" || this.#state.phase === "verifying") return;
    if (await modelStore.isCached(MODEL_VERSION)) {
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
        await this.refresh();
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

    const listener = (state: ModelState): void => reply({ type: "model", state });
    this.#listeners.add(listener);

    port.onMessage.addListener((raw: unknown) => {
      const message = raw as ContentMessage;
      void (async () => {
        try {
          switch (message.type) {
            case "hello": {
              await this.refresh();
              reply({ type: "model", state: this.#state });
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
          console.error("slop-marker router:", err);
        }
      })();
    });

    port.onDisconnect.addListener(() => {
      this.#listeners.delete(listener);
      this.#host?.detach(reply);
    });
  }

  /** Options-page and first-run-page commands. Same on both browsers. */
  handleCommand(message: { type: string }, sendResponse: (response: unknown) => void): boolean {
    switch (message.type) {
      case "getStatus":
        void (async () => {
          await this.refresh();
          const stored = await chrome.storage.local.get("threading");
          sendResponse({
            modelState: this.#state,
            modelVersion: MODEL_VERSION,
            cachedScores: await scoreCache.size(),
            threading: stored["threading"] ?? null,
          });
        })();
        return true;

      case "downloadModel":
        void this.ensureModel().then(
          () => sendResponse({ ok: true, state: this.#state }),
          (err: unknown) => sendResponse({ ok: false, message: String(err) }),
        );
        return true;

      case "clearModel":
        void (async () => {
          await this.#host?.dispose();
          this.#host = null;
          await modelStore.clearCache();
          await scoreCache.clear();
          this.#setState({ phase: "absent" });
          sendResponse({ ok: true });
        })();
        return true;

      case "clearScores":
        void scoreCache.clear().then(() => sendResponse({ ok: true }));
        return true;

      default:
        return false;
    }
  }
}

/** Download directly. Used where COEP does not constrain fetching. */
export async function downloadHere(onState?: (state: ModelState) => void): Promise<void> {
  await modelStore.download(MODEL_VERSION, onState ?? (() => undefined));
}
