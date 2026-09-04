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

chrome.runtime.onMessage.addListener((message: { type: string }, _sender, sendResponse) => {
  // downloadModel belongs to the service worker; everything else is answered here.
  if (message.type === "downloadModel") return false;
  return router.handleCommand(message, sendResponse);
});

void router.refresh();
