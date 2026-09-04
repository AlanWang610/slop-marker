/**
 * Messages on the content-script <-> background port, and between the host and its worker.
 *
 * One file so the two ends cannot disagree. The content script never talks to the worker
 * directly: it speaks to a router (scope.md 6.2), which is the only browser-specific piece.
 */

import type { Calibration } from "./calibration.js";

export const PORT_NAME = "slop-marker";

/**
 * Chrome only: the service worker relays each content-script port to the offscreen
 * document over a second port with this name. Firefox has no such hop.
 */
export const OFFSCREEN_PORT = "slop-marker:offscreen";

/** Content scripts ping this often while on-screen work is outstanding (scope.md 6.2). */
export const KEEPALIVE_MS = 20_000;

/** Lower is more urgent. Viewport distance in screens; 0 means on-screen. */
export type Priority = number;

export interface ScoreRequest {
  /** sha256(normalizeForHash(text)) -- content-keyed, so syndicated text scores once. */
  readonly hash: string;
  readonly text: string;
  readonly words: number;
  readonly priority: Priority;
}

export type ModelState =
  | { readonly phase: "absent" }
  | { readonly phase: "downloading"; readonly received: number; readonly total: number }
  | { readonly phase: "verifying" }
  | { readonly phase: "ready"; readonly version: string }
  | { readonly phase: "error"; readonly message: string };

export interface ThreadingInfo {
  readonly crossOriginIsolated: boolean;
  readonly numThreads: number;
  readonly hardwareConcurrency: number;
}

/** content script -> router -> host */
export type ContentMessage =
  | { readonly type: "hello" }
  | { readonly type: "score"; readonly items: readonly ScoreRequest[] }
  | { readonly type: "reprioritize"; readonly priorities: Readonly<Record<string, Priority>> }
  | { readonly type: "cancel"; readonly hashes: readonly string[] }
  | { readonly type: "keepalive" };

/** host -> router -> content script */
export type HostMessage =
  | {
      readonly type: "ready";
      readonly calibration: Calibration;
      readonly threading: ThreadingInfo;
    }
  | { readonly type: "model"; readonly state: ModelState }
  | {
      readonly type: "score";
      readonly hash: string;
      readonly logit: number;
      readonly words: number;
      readonly modelVersion: string;
    }
  | { readonly type: "failed"; readonly hash: string; readonly message: string };

/** host <-> inference worker */
export type WorkerRequest =
  | {
      readonly type: "init";
      readonly model: ArrayBuffer;
      readonly tokenizerJson: string;
      readonly tokenizerConfigJson: string;
      readonly maxLength: number;
      readonly numThreads: number;
      readonly wasmPaths: string;
    }
  | { readonly type: "run"; readonly id: number; readonly text: string }
  | { readonly type: "close" };

export type WorkerResponse =
  | { readonly type: "ready"; readonly threading: ThreadingInfo }
  | { readonly type: "result"; readonly id: number; readonly logit: number; readonly nTokens: number }
  | { readonly type: "error"; readonly id?: number; readonly message: string };
