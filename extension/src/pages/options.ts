/**
 * Options (scope.md 6.3): per-site allowlist, threshold override, cache controls, and the
 * observed threading mode. Nothing else, deliberately -- there is no feedback UI, no
 * telemetry and no reporting anywhere in this project.
 */

import { SHIPPED_CALIBRATION_JSON } from "../shared/bundle-config.js";
import { parseCalibration } from "../shared/calibration.js";
import type { ModelState, ThreadingInfo } from "../shared/protocol.js";

interface Status {
  modelState: ModelState;
  modelVersion: string;
  cachedScores: number;
  threading: ThreadingInfo | null;
}

const $ = (id: string): HTMLElement => document.getElementById(id)!;

function describe(state: ModelState): string {
  switch (state.phase) {
    case "absent":
      return "not downloaded";
    case "downloading":
      return `downloading (${(state.received / 1e6).toFixed(0)} MB)`;
    case "verifying":
      return "verifying checksum";
    case "ready":
      return "ready";
    case "error":
      return `failed — ${state.message}`;
  }
}

async function refresh(): Promise<void> {
  // Annotated rather than asserted: sendMessage is typed `any`, and an annotation carries
  // the same information without tripping no-unnecessary-type-assertion.
  const status: Status | undefined = await chrome.runtime.sendMessage({ type: "getStatus" });
  if (status === undefined) return;

  $("model-version").textContent = status.modelVersion;
  $("model-state").textContent = describe(status.modelState);
  $("model-state").className = status.modelState.phase === "ready" ? "status-ok" : "";
  $("cached-scores").textContent = status.cachedScores.toLocaleString();

  const threading = status.threading;
  if (threading === null) {
    $("threading").textContent = "not yet observed";
    $("threading-hint").textContent = "";
  } else {
    $("threading").textContent = `${threading.numThreads} thread${
      threading.numThreads === 1 ? "" : "s"
    } (${threading.crossOriginIsolated ? "cross-origin isolated" : "not isolated"}, ${
      threading.hardwareConcurrency
    } cores available)`;
    $("threading-hint").textContent =
      threading.numThreads === 1
        ? "Single-threaded: scoring is roughly 3-4x slower here. This browser did not make " +
          "the inference context cross-origin isolated, which is what threads require."
        : "";
  }
}

/* ------------------------------------------------------------------ allowlist */

const allowlist = $("allowlist") as HTMLTextAreaElement;

async function loadAllowlist(): Promise<void> {
  const stored = await chrome.storage.local.get("allowlist");
  allowlist.value = ((stored["allowlist"] as string[] | undefined) ?? []).join("\n");
}

$("save-allowlist").addEventListener("click", () => {
  const origins = allowlist.value
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  void chrome.storage.local.set({ allowlist: origins }).then(() => {
    $("allowlist-status").textContent = `Saved ${origins.length} origin${origins.length === 1 ? "" : "s"}.`;
    $("allowlist-status").className = "status-ok";
    setTimeout(() => ($("allowlist-status").textContent = ""), 2500);
  });
});

/* ------------------------------------------------------------------ threshold */

const threshold = $("threshold") as HTMLInputElement;

async function loadThreshold(): Promise<void> {
  const stored = await chrome.storage.local.get("thresholdOverride");
  const value = stored["thresholdOverride"] as number | undefined;
  if (value === undefined) {
    // The slider starts at the shipped t_on so moving it is a relative act, not a jump.
    const shipped = parseCalibration(SHIPPED_CALIBRATION_JSON);
    $("threshold-value").textContent = `default (${shipped.t_on.toFixed(3)})`;
    threshold.value = String(shipped.t_on);
  } else {
    $("threshold-value").textContent = value.toFixed(3);
    threshold.value = String(value);
  }
}

threshold.addEventListener("input", () => {
  const value = Number(threshold.value);
  $("threshold-value").textContent = value.toFixed(3);
  void chrome.storage.local.set({ thresholdOverride: value });
});

$("reset-threshold").addEventListener("click", () => {
  void chrome.storage.local.remove("thresholdOverride").then(() => void loadThreshold());
});

/* ------------------------------------------------------------------ storage */

function wire(id: string, type: string, label: string): void {
  $(id).addEventListener("click", () => {
    $("storage-status").textContent = "Working…";
    void chrome.runtime.sendMessage({ type }).then(() => {
      $("storage-status").textContent = label;
      $("storage-status").className = "status-ok";
      void refresh();
      setTimeout(() => ($("storage-status").textContent = ""), 2500);
    });
  });
}

wire("clear-scores", "clearScores", "Score cache cleared.");
wire("clear-model", "clearModel", "Model cleared. It downloads again on the next page.");

void refresh();
void loadAllowlist();
void loadThreshold();
setInterval(() => void refresh(), 2000);
