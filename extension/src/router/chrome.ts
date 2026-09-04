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

import { downloadHere } from "./shared.js";
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

/** Relay a content-script port to the offscreen document, verbatim, in both directions. */
function relay(port: chrome.runtime.Port): void {
  const downstream = chrome.runtime.connect({ name: OFFSCREEN_PORT });
  const up = (message: unknown): void => {
    try {
      port.postMessage(message);
    } catch {
      /* page went away */
    }
  };
  const down = (message: unknown): void => {
    try {
      downstream.postMessage(message);
    } catch {
      /* offscreen restarting */
    }
  };
  downstream.onMessage.addListener(up);
  port.onMessage.addListener(down);
  downstream.onDisconnect.addListener(() => port.disconnect());
  port.onDisconnect.addListener(() => downstream.disconnect());
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== PORT_NAME) return;
  void ensureOffscreen().then(
    () => relay(port),
    (err: unknown) => {
      console.error("slop-marker: could not create the offscreen document:", err);
      port.disconnect();
    },
  );
});

/**
 * `downloadModel` is answered here rather than being relayed, since this is the context
 * that may fetch. Everything else is the offscreen document's business, so it is left
 * unhandled and the offscreen listener answers it.
 */
chrome.runtime.onMessage.addListener((message: { type: string }, _sender, sendResponse) => {
  if (message.type !== "downloadModel") return false;
  void downloadHere().then(
    () => sendResponse({ ok: true }),
    (err: unknown) => sendResponse({ ok: false, message: String(err) }),
  );
  return true;
});

chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason !== "install") return;
  // Chrome grants <all_urls> at install, so the first-run page's permission request is a
  // no-op here. It runs anyway: it is also where the model download starts (scope.md 6.5).
  void chrome.tabs.create({ url: chrome.runtime.getURL("first-run.html") });
});
