/**
 * The content script's wiring: port lifecycle, observers, storage, painting.
 *
 * All the logic that decides *what* gets scored and *what* gets highlighted lives in
 * pipeline.ts, which has no browser dependencies and is unit-tested. This file is the part
 * that can only be exercised in a real browser, so it is kept as thin as it can be.
 *
 * Nothing here is stateful across passes. Re-scoring after a DOM mutation re-runs the same
 * pure function over the same inputs, which is what makes scope.md 8's hysteresis spatial
 * rather than temporal: the same page always renders the same way.
 */

import { type Calibration, withThresholdOverride } from "../shared/calibration.js";
import { contentHash } from "../shared/normalize.js";
import {
  KEEPALIVE_MS,
  PORT_NAME,
  type ContentMessage,
  type HostMessage,
  type ScoreRequest,
} from "../shared/protocol.js";
import { extractBlocks, pinRoot } from "./extract.js";
import { buildChunks, computeRuns, priorityFor, type PageChunk } from "./pipeline.js";
import { render } from "./render.js";

/** Silence longer than this, with chunks still unscored, means the port is not working. */
const STALL_MS = 20_000;

let calibration: Calibration | null = null;
/** Set from the options page; re-derives t_off so the hysteresis band survives. */
let thresholdOverride: number | null = null;
let chunks: PageChunk[] = [];
let port: chrome.runtime.Port | null = null;
let allowed = true;
let renderScheduled = false;

/* ------------------------------------------------------------------ rendering */

function paint(): void {
  if (calibration === null) return;
  const cal =
    thresholdOverride === null
      ? calibration
      : withThresholdOverride(calibration, thresholdOverride);
  render(computeRuns(chunks, cal));
}

function schedulePaint(): void {
  if (renderScheduled) return;
  renderScheduled = true;
  requestAnimationFrame(() => {
    renderScheduled = false;
    paint();
  });
}

/* ------------------------------------------------------------------ priority */

let viewport: IntersectionObserver | null = null;

/**
 * Viewport-distance priority (scope.md 7.4). The observer's one-screen `rootMargin` is what
 * makes it fire *before* a block scrolls in, so the chunk is already queued at priority 0 by
 * the time the reader reaches it.
 *
 * Priority is measured against the real viewport rather than `entry.rootBounds`, which is
 * the margin-expanded box: against that, everything within a screen would tie at 0 and the
 * queue would lose the ordering it exists for.
 */
/** Block element -> its chunks, so a callback costs O(entries) rather than O(entries x chunks). */
let byElement = new Map<Element, PageChunk[]>();

function watchViewport(): void {
  viewport?.disconnect();
  viewport = new IntersectionObserver(onIntersect, { rootMargin: "100% 0px" });

  byElement = new Map();
  for (const chunk of chunks) {
    const existing = byElement.get(chunk.block.element);
    if (existing === undefined) byElement.set(chunk.block.element, [chunk]);
    else existing.push(chunk);
  }
  for (const element of byElement.keys()) viewport.observe(element);
}

function onIntersect(entries: IntersectionObserverEntry[]): void {
  const height = window.innerHeight || 1;
  const changed: Record<string, number> = {};
  for (const entry of entries) {
    const priority = priorityFor(entry.boundingClientRect, height);
    for (const chunk of byElement.get(entry.target) ?? []) {
      chunk.priority = priority;
      if (chunk.logit === null && chunk.hash !== "") changed[chunk.hash] = priority;
    }
  }
  if (Object.keys(changed).length > 0) post({ type: "reprioritize", priorities: changed });
}

/**
 * Seed priorities synchronously before the first request goes out. The observer's first
 * callback is asynchronous, and without this the whole first batch would be queued at
 * Infinity and drained in arbitrary order -- which is exactly the case viewport-first
 * scheduling is for.
 */
function seedPriorities(): void {
  const height = window.innerHeight || 1;
  for (const chunk of chunks) {
    chunk.priority = priorityFor(chunk.block.element.getBoundingClientRect(), height);
  }
}

/* ------------------------------------------------------------------ transport */

/**
 * Send, or notice that we cannot and start getting the port back.
 *
 * Waiting for `onDisconnect` alone is not enough. A terminated service worker can leave the
 * port looking present until the next write throws, and a write attempted while `port` is
 * null used to be a silent no-op with nothing to re-send it -- so a rescan that happened to
 * land in that window was lost for good and the new content was never scored. Every path
 * out of here either delivers the message or schedules a reconnect that re-asks for
 * everything still outstanding (scope.md 7.4).
 */
function post(message: ContentMessage): void {
  if (port === null) {
    scheduleReconnect(0);
    return;
  }
  try {
    port.postMessage(message);
  } catch {
    port = null;
    scheduleReconnect(0);
  }
}

/**
 * When the host last said anything, or we last asked it something. Drives the stall
 * watchdog in `observe()`.
 */
let lastProgressAt = Date.now();

function requestOutstanding(): void {
  const items: ScoreRequest[] = chunks
    .filter((c) => c.logit === null)
    .map((c) => ({ hash: c.hash, text: c.text, words: c.words, priority: c.priority }));
  if (items.length === 0) return;
  lastProgressAt = Date.now();
  post({ type: "score", items });
}

function onMessage(raw: unknown): void {
  lastProgressAt = Date.now();
  const message = raw as HostMessage;
  switch (message.type) {
    case "ready":
      if (calibration === null) {
        calibration = message.calibration;
        void start();
      }
      return;
    case "score": {
      let changed = false;
      for (const chunk of chunks) {
        if (chunk.hash === message.hash && chunk.logit === null) {
          chunk.logit = message.logit;
          changed = true;
        }
      }
      if (changed) schedulePaint();
      return;
    }
    case "failed":
      // Leave the logit null: the page simply stays un-highlighted, which is the safe
      // direction for a high-precision detector.
      console.debug("slop-marker: chunk failed —", message.message);
      return;
    case "model":
      return;
  }
}

let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

/**
 * Reconnect and re-ask for whatever has not come back yet. Covers a Chrome service worker
 * restart and a Firefox event page unload with the same code (scope.md 7.4).
 */
function scheduleReconnect(delay = 500): void {
  if (reconnectTimer !== null || !allowed) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    if (!allowed) return;
    connect();
    if (calibration !== null) requestOutstanding();
  }, delay);
}

function connect(): void {
  try {
    port = chrome.runtime.connect({ name: PORT_NAME });
  } catch (err) {
    // The extension was reloaded or removed under us, so this content script is orphaned
    // and no amount of retrying will help. Stop rather than spinning. Worth one line: an
    // orphaned content script and a page with no AI text on it look identical.
    console.warn("slop-marker: extension context is gone, stopping —", err);
    allowed = false;
    port = null;
    return;
  }
  port.onMessage.addListener(onMessage);
  port.onDisconnect.addListener(() => {
    port = null;
    scheduleReconnect();
  });
  post({ type: "hello" });
}

/* ------------------------------------------------------------------ lifecycle */

let root: Element | null = null;

/** The held content root; see `pinRoot` for why it is held rather than re-derived. */
function pinnedRoot(): Element {
  root = pinRoot(root);
  return root;
}

async function start(): Promise<void> {
  if (calibration === null || !allowed) return;
  chunks = buildChunks(calibration, extractBlocks(pinnedRoot()));

  await Promise.all(
    chunks.map(async (chunk) => {
      chunk.hash = await contentHash(chunk.text);
    }),
  );
  seedPriorities();
  requestOutstanding();

  // Unconditionally, even with nothing to score yet. A client-rendered page has no prose at
  // document_idle -- which is precisely the case scope.md 7.1's MutationObserver exists for
  // -- so returning early here would leave the observer uninstalled on exactly those pages
  // and they would never be scored at all.
  observe();
}

let mutationTimer: ReturnType<typeof setTimeout> | null = null;
let mutations: MutationObserver | null = null;

/** Watch the pinned root for added subtrees, debounced (scope.md 7.1). */
function watchMutations(): void {
  mutations ??= new MutationObserver((records) => {
    const added = records.some((r) => r.addedNodes.length > 0);
    if (!added) return;
    if (mutationTimer !== null) clearTimeout(mutationTimer);
    mutationTimer = setTimeout(() => {
      mutationTimer = null;
      void rescan();
    }, 500);
  });
  mutations.disconnect();
  mutations.observe(pinnedRoot(), { childList: true, subtree: true });
}

function observe(): void {
  watchViewport();
  watchMutations();

  setInterval(() => {
    const outstanding = chunks.filter((c) => c.logit === null);
    if (outstanding.length === 0) return;

    // Port traffic resets the Firefox event-page idle timer (scope.md 6.2). Only while
    // there is on-screen work, so an idle tab lets the page unload as it should.
    if (outstanding.some((c) => c.priority <= 1)) post({ type: "keepalive" });

    if (Date.now() - lastProgressAt < STALL_MS) return;

    /**
     * Nothing has come back for a while with work still outstanding, so start the port
     * over.
     *
     * `onDisconnect` is not a sufficient signal on its own. A port to a service worker that
     * has been terminated can stay writable and simply swallow what is posted into it --
     * measured: the worker restarts, the content script's port never reports a disconnect,
     * and every score request after that point is lost. The failure is invisible, because a
     * page that is never scored looks exactly like a page with nothing to flag. So the only
     * reliable evidence is silence, and this acts on it.
     */
    lastProgressAt = Date.now();
    try {
      port?.disconnect();
    } catch {
      /* already gone */
    }
    port = null;
    scheduleReconnect(0);
  }, KEEPALIVE_MS);
}

/** Re-extract, carrying forward scores we already have. Content-keyed, so cheap. */
async function rescan(): Promise<void> {
  if (calibration === null || !allowed) return;
  const known = new Map(chunks.filter((c) => c.logit !== null).map((c) => [c.hash, c.logit!]));
  const previousRoot = root;
  const next = buildChunks(calibration, extractBlocks(pinnedRoot()));
  // A wholesale client-side navigation replaces the root; follow it with the observer.
  if (root !== previousRoot) watchMutations();
  await Promise.all(
    next.map(async (chunk) => {
      chunk.hash = await contentHash(chunk.text);
      chunk.logit = known.get(chunk.hash) ?? null;
    }),
  );
  chunks = next;
  seedPriorities();
  watchViewport();
  requestOutstanding();
  schedulePaint();
}

async function main(): Promise<void> {
  const stored = await chrome.storage.local.get(["allowlist", "thresholdOverride"]);
  const allowlist = (stored["allowlist"] as string[] | undefined) ?? [];
  if (allowlist.includes(location.origin)) {
    allowed = false;
    return;
  }
  thresholdOverride = (stored["thresholdOverride"] as number | undefined) ?? null;

  // Repaint when the override changes, so the options page slider is live on open tabs
  // rather than needing a reload.
  chrome.storage.onChanged.addListener((changes) => {
    if (!("thresholdOverride" in changes)) return;
    thresholdOverride = (changes["thresholdOverride"]?.newValue as number | undefined) ?? null;
    schedulePaint();
  });

  connect();
}

void main();
