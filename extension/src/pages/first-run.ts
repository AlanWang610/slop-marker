/**
 * First run (scope.md 6.5).
 *
 * The interesting half is Firefox: MV3 there grants no host permissions at install, so
 * content scripts do nothing at all until the user enables site access. That has to be
 * requested from a user gesture, which is what the button is for. On Chrome <all_urls> is
 * granted at install and the same request is a harmless no-op, so one page serves both.
 */

import type { ModelState } from "../shared/protocol.js";

const ORIGINS = { origins: ["<all_urls>"] };

const grant = document.getElementById("grant") as HTMLButtonElement;
const permissionStatus = document.getElementById("permission-status")!;
const explainer = document.getElementById("permission-explainer")!;
const download = document.getElementById("download") as HTMLButtonElement;
const modelStatus = document.getElementById("model-status")!;
const progress = document.getElementById("progress") as HTMLProgressElement;

function describe(state: ModelState): string {
  switch (state.phase) {
    case "absent":
      return "Not downloaded yet.";
    case "downloading":
      return state.total > 0
        ? `Downloading… ${(state.received / 1e6).toFixed(0)} of ${(state.total / 1e6).toFixed(0)} MB`
        : `Downloading… ${(state.received / 1e6).toFixed(0)} MB`;
    case "verifying":
      return "Verifying checksum…";
    case "ready":
      return `Ready (${state.version}).`;
    case "error":
      return `Failed: ${state.message}`;
  }
}

function paintModel(state: ModelState): void {
  modelStatus.textContent = describe(state);
  modelStatus.className =
    state.phase === "ready" ? "status-ok" : state.phase === "error" ? "status-error" : "";
  progress.hidden = state.phase !== "downloading";
  if (state.phase === "downloading" && state.total > 0) {
    progress.value = state.received / state.total;
  }
  download.disabled = state.phase === "downloading" || state.phase === "verifying";
  download.textContent = state.phase === "ready" ? "Re-download model" : "Download model";
}

async function refreshPermission(): Promise<boolean> {
  const granted = await chrome.permissions.contains(ORIGINS);
  if (granted) {
    explainer.textContent = "Granted. slop-marker can read page text to score it.";
    permissionStatus.textContent = "✓ Granted";
    permissionStatus.className = "status-ok";
    grant.disabled = true;
  } else {
    explainer.textContent =
      "Firefox does not grant site access to MV3 extensions at install, so nothing is " +
      "scored until you allow it. Page text is read locally and never leaves the browser.";
    permissionStatus.textContent = "";
    grant.disabled = false;
  }
  return granted;
}

grant.addEventListener("click", () => {
  // Must be called synchronously from the gesture, so no await before this point.
  void chrome.permissions.request(ORIGINS).then(
    () => void refreshPermission(),
    (err: unknown) => {
      permissionStatus.textContent = String(err);
      permissionStatus.className = "status-error";
    },
  );
});

download.addEventListener("click", () => {
  download.disabled = true;
  modelStatus.textContent = "Starting…";
  void chrome.runtime.sendMessage({ type: "downloadModel" }).then(
    (response: { ok: boolean; message?: string } | undefined) => {
      if (response !== undefined && !response.ok) {
        paintModel({ phase: "error", message: response.message ?? "unknown error" });
      }
      void poll();
    },
    (err: unknown) => paintModel({ phase: "error", message: String(err) }),
  );
});

async function poll(): Promise<void> {
  const stored = await chrome.storage.local.get("modelState");
  const state = stored["modelState"] as ModelState | undefined;
  if (state !== undefined) paintModel(state);
}

chrome.storage.onChanged.addListener((changes) => {
  const change = changes["modelState"];
  if (change !== undefined) paintModel(change.newValue as ModelState);
});

void refreshPermission();
void poll();
setInterval(() => void poll(), 1000);
