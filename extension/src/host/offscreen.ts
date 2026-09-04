/**
 * Chrome only: the offscreen document, which is where the Host actually lives.
 *
 * It exists for one reason -- an MV3 service worker cannot hold a Web Worker or an ORT
 * session across its own restarts, and this document can. It is created on demand by
 * router/chrome.ts and lives until the extension is reloaded.
 *
 * It cannot download. Extension pages are cross-origin isolated here (manifest COOP/COEP,
 * which is also what buys ORT its threads), and require-corp blocks the cross-origin fetch.
 * So it asks the service worker to download and then reads the verified bytes out of the
 * Cache API, which is per-origin and shared between the two contexts.
 */

import { OFFSCREEN_PORT } from "../shared/protocol.js";
import { Router } from "../router/shared.js";
import { Host } from "./host.js";

// Recorded before anything else can throw. Nothing shows this document's console -- devtools
// does not list offscreen documents by default -- so if the module fails to evaluate, its
// onConnect listener is never registered and every port silently gets no reply. The options
// page reads this back, and it is the difference between "broken" and "broken, here".
void chrome.storage.local.set({ offscreenBootedAt: Date.now() });
self.addEventListener("error", (event) => {
  void chrome.storage.local.set({
    lastError: { at: Date.now(), where: "offscreen", detail: String(event.message) },
  });
});
self.addEventListener("unhandledrejection", (event) => {
  void chrome.storage.local.set({
    lastError: { at: Date.now(), where: "offscreen", detail: String(event.reason) },
  });
});

const router = new Router({
  ensureModel: async () => {
    const response: { ok: boolean; message?: string } | undefined =
      await chrome.runtime.sendMessage({ type: "downloadModel" });
    if (response === undefined || !response.ok) {
      throw new Error(response?.message ?? "the service worker did not answer");
    }
  },
  createHost: () =>
    new Host(chrome.runtime.getURL("worker.js"), chrome.runtime.getURL("ort/")),
});

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== OFFSCREEN_PORT) return;
  router.attachPort(port);
});

// Commands are answered by the service worker, which is the context that may fetch (see
// router/shared.ts). The only thing this document needs to hear about is the model being
// cleared, which invalidates the session it is holding.
chrome.runtime.onMessage.addListener((message: { type: string }, _sender, sendResponse) => {
  if (message.type !== "clearModel") return false;
  void router.disposeHost().then(() => sendResponse({ ok: true }));
  return true;
});

void router.refresh();
