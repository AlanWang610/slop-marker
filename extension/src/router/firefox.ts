/**
 * Firefox router: MV3 event page, which is also the host (scope.md 6.2).
 *
 * No offscreen document exists on Firefox, so the Router and the Host live here together.
 * The event page unloads after ~30 s without events, taking the ORT session with it.
 * Two mitigations, in layers:
 *
 *   1. Content scripts hold the port open and send a keepalive every 20 s while a tab has
 *      unscored on-screen blocks. Port traffic resets the idle timer.
 *   2. Warm restore: when the page reloads, the Host re-creates the session from the Cache
 *      API. Measured at 0.6 s from cached bytes (docs/measurements/extension-latency.md),
 *      well inside the "low single-digit seconds" scope.md 6.2 budgets for it. Content
 *      scripts tolerate the gap by re-sending pending requests on port reconnect.
 *
 * COOP/COEP are ignored here, so this page is not cross-origin isolated and may be
 * single-threaded -- the observed mode is recorded and shown on the options page.
 */

import { Host } from "../host/host.js";
import { PORT_NAME } from "../shared/protocol.js";
import { downloadHere, Router } from "./shared.js";

const router = new Router({
  // Nothing constrains fetching here: Firefox ignores the COOP/COEP manifest keys.
  ensureModel: () => downloadHere(),
  createHost: () =>
    new Host(chrome.runtime.getURL("worker.js"), chrome.runtime.getURL("ort/")),
});

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== PORT_NAME) return;
  router.attachPort(port);
});

chrome.runtime.onMessage.addListener((message: { type: string }, _sender, sendResponse) =>
  router.handleCommand(message, sendResponse),
);

chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason !== "install") return;
  // Firefox MV3 grants no host permissions at install, so content scripts do nothing until
  // the user enables site access. The first-run page explains that and asks (scope.md 6.5).
  void chrome.tabs.create({ url: chrome.runtime.getURL("first-run.html") });
});

void router.refresh();
