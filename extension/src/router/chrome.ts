/**
 * Chrome router: MV3 service worker, stateless (scope.md 6.2).
 *
 * Holds no cache and no queue. Two jobs only:
 *
 *   1. Make sure the offscreen document exists, then relay content-script ports to it.
 *      The offscreen document owns the Host and persists until the extension is reloaded
 *      or the browser exits -- it is not subject to the 30 s idle rule, which applies only
 *      to AUDIO_PLAYBACK. So the service worker being killed is a non-event.
 *   2. Download the model, because it is the only context that can. Extension pages are
 *      cross-origin isolated by the manifest COOP/COEP keys so ORT can use threads, and
 *      under require-corp a cross-origin fetch needs CORS or CORP, which GitHub release
 *      assets do not send. A service worker is not an extension *page* and is unaffected.
 */

import { handleCommand } from "./shared.js";
import { OFFSCREEN_PORT, PORT_NAME } from "../shared/protocol.js";

const OFFSCREEN_URL = "host.html";

let creating: Promise<void> | null = null;

async function ensureOffscreen(): Promise<void> {
  const existing = await chrome.runtime.getContexts({
    contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT],
    documentUrls: [chrome.runtime.getURL(OFFSCREEN_URL)],
  });
  if (existing.length > 0) return;

  creating ??= chrome.offscreen
    .createDocument({
      url: OFFSCREEN_URL,
      reasons: [chrome.offscreen.Reason.WORKERS],
      justification:
        "Runs the ONNX Runtime Web session in a dedicated Web Worker and keeps it alive " +
        "across service worker restarts.",
    })
    .catch((err: unknown) => {
      // Two tabs can race here; a document created by the other one is fine.
      if (!String(err).includes("Only a single offscreen document")) throw err;
    })
    .finally(() => {
      creating = null;
    });
  await creating;
}

/**
 * Relay a content-script port to the offscreen document, verbatim, in both directions.
 *
 * The listener is attached SYNCHRONOUSLY and buffers, which is not a nicety. A content
 * script posts `hello` in the same turn it calls `connect()`, while this side is still
 * awaiting `ensureOffscreen()` -- creating an offscreen document is slow the first time.
 * Chrome does not queue port messages for a listener that does not exist yet, so without
 * the buffer that first `hello` is dropped and the page is simply never scored. It fails
 * silently and only on the first page after a browser start, which is the worst shape a
 * bug can have.
 */
function relay(port: chrome.runtime.Port): void {
  const pending: unknown[] = [];
  let downstream: chrome.runtime.Port | null = null;
  let closed = false;

  port.onMessage.addListener((message: unknown) => {
    if (downstream === null) pending.push(message);
    else {
      try {
        downstream.postMessage(message);
      } catch {
        /* offscreen restarting */
      }
    }
  });
  port.onDisconnect.addListener(() => {
    closed = true;
    downstream?.disconnect();
  });

  void ensureOffscreen().then(
    () => {
      if (closed) return;
      downstream = chrome.runtime.connect({ name: OFFSCREEN_PORT });
      downstream.onMessage.addListener((message: unknown) => {
        try {
          port.postMessage(message);
        } catch {
          /* page went away */
        }
      });
      downstream.onDisconnect.addListener(() => port.disconnect());
      for (const message of pending) downstream.postMessage(message);
      pending.length = 0;
    },
    (err: unknown) => {
      console.error("slop-marker: could not create the offscreen document:", err);
      void chrome.storage.local.set({
        lastError: { at: Date.now(), where: "offscreen", detail: String(err) },
      });
      port.disconnect();
    },
  );
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== PORT_NAME) return;
  relay(port);
});

/**
 * Every page command is answered here, not in the offscreen document.
 *
 * The offscreen document is created lazily, and `sendMessage` that no context handles
 * resolves to `undefined` rather than failing -- so routing status through the offscreen
 * meant the first page to ask silently got nothing back. The service worker always exists
 * when a message arrives, and it is also the only context that may fetch.
 */
chrome.runtime.onMessage.addListener((message: { type: string }, _sender, sendResponse) =>
  handleCommand(message, sendResponse, () => {
    // Best-effort: if the offscreen document is holding a session for a model we just
    // deleted, ask it to drop it. It may not exist, and that is fine.
    void chrome.runtime.sendMessage({ type: "clearModel" }).catch(() => undefined);
  }),
);

chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason !== "install") return;
  // Chrome grants <all_urls> at install, so the first-run page's permission request is a
  // no-op here. It runs anyway: it is also where the model download starts (scope.md 6.5).
  void chrome.tabs.create({ url: chrome.runtime.getURL("first-run.html") });
});
