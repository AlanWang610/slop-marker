/**
 * The content script: extract -> language gate -> chunk -> hash -> score -> aggregate ->
 * highlight (scope.md 7, 8, 9).
 *
 * Aggregation is per *document*, not per block. scope.md 8's document prior counts how many
 * of the page's chunks clear `t_off`, and runs extend across neighbouring chunks -- so the
 * whole page's chunks form one ordered sequence, and a run may span block boundaries.
 *
 * Nothing here is stateful across passes. Re-scoring after a DOM mutation re-runs the same
 * pure function over the same inputs, which is what makes scope.md 8's hysteresis spatial
 * rather than temporal: the same page always renders the same way.
 */

import { aggregate, type Chunk } from "../shared/aggregate.js";
import { type Calibration, withThresholdOverride } from "../shared/calibration.js";
import { chunkSpans } from "../shared/chunking.js";
import { contentHash, countWords, toCodePoints } from "../shared/normalize.js";
import {
  KEEPALIVE_MS,
  PORT_NAME,
  type ContentMessage,
  type HostMessage,
  type ScoreRequest,
} from "../shared/protocol.js";
import { type Block, contentRoot, extractBlocks, rangeFor } from "./extract.js";
import { isScoreable } from "./langgate.js";
import { render, type RenderedRun } from "./render.js";

/** One scoreable unit: a span of one block's collapsed text. */
interface PageChunk {
  readonly block: Block;
  readonly start: number;
  readonly end: number;
  readonly text: string;
  readonly words: number;
  hash: string;
  logit: number | null;
  /** Viewport distance in screens; 0 is on-screen. Lower is scored first. */
  priority: number;
}

let calibration: Calibration | null = null;
/** Set from the options page; re-derives t_off so the hysteresis band survives. */
let thresholdOverride: number | null = null;
let chunks: PageChunk[] = [];
let port: chrome.runtime.Port | null = null;
let allowed = true;
let renderScheduled = false;

/* ------------------------------------------------------------------ extraction */

function buildChunks(cal: Calibration): PageChunk[] {
  const out: PageChunk[] = [];
  for (const block of extractBlocks()) {
    if (countWords(block.text) < cal.min_words) continue;
    if (!isScoreable(block.text)) continue;

    const cp = toCodePoints(block.text);
    for (const [start, end] of chunkSpans(block.text, cal.min_words, 400)) {
      const text = cp.slice(start, end).join("");
      out.push({
        block,
        start,
        end,
        text,
        words: countWords(text),
        hash: "",
        logit: null,
        priority: Number.POSITIVE_INFINITY,
      });
    }
  }
  return out;
}

/* ------------------------------------------------------------------ priority */

function updatePriorities(): void {
  const height = window.innerHeight || 1;
  for (const chunk of chunks) {
    const rect = chunk.block.element.getBoundingClientRect();
    if (rect.bottom >= 0 && rect.top <= height) {
      chunk.priority = 0;
    } else if (rect.top > height) {
      chunk.priority = (rect.top - height) / height;
    } else {
      chunk.priority = -rect.bottom / height;
    }
  }
}

/* ------------------------------------------------------------------ rendering */

/**
 * Aggregate and paint. Runs only once every chunk has a score, because scope.md 8's
 * document prior depends on the whole page: painting a partial page would flash highlights
 * that the prior later withdraws.
 */
function paint(): void {
  if (calibration === null || chunks.length === 0) return;
  if (chunks.some((c) => c.logit === null)) return;

  const cal =
    thresholdOverride === null
      ? calibration
      : withThresholdOverride(calibration, thresholdOverride);
  const input: Chunk[] = chunks.map((c) => ({ logit: c.logit!, words: c.words }));
  const result = aggregate(input, cal);

  const runs: RenderedRun[] = [];
  for (const run of result.runs) {
    if (!run.flagged) continue;
    const ranges: Range[] = [];
    const blocks = new Set<Element>();
    for (let i = run.start; i <= run.end; i++) {
      const chunk = chunks[i]!;
      const range = rangeFor(chunk.block, chunk.start, chunk.end);
      if (range !== null) ranges.push(range);
      blocks.add(chunk.block.element);
    }
    if (ranges.length === 0) continue;
    runs.push({
      ranges,
      blocks: [...blocks],
      score: run.score,
      words: run.words,
      modelVersion: cal.version,
    });
  }
  render(runs);
}

function schedulePaint(): void {
  if (renderScheduled) return;
  renderScheduled = true;
  requestAnimationFrame(() => {
    renderScheduled = false;
    paint();
  });
}

/* ------------------------------------------------------------------ transport */

function post(message: ContentMessage): void {
  try {
    port?.postMessage(message);
  } catch {
    // Disconnected; onDisconnect will reconnect and re-send.
  }
}

function requestOutstanding(): void {
  const items: ScoreRequest[] = chunks
    .filter((c) => c.logit === null)
    .map((c) => ({ hash: c.hash, text: c.text, words: c.words, priority: c.priority }));
  if (items.length > 0) post({ type: "score", items });
}

function onMessage(raw: unknown): void {
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

function connect(): void {
  port = chrome.runtime.connect({ name: PORT_NAME });
  port.onMessage.addListener(onMessage);
  port.onDisconnect.addListener(() => {
    port = null;
    // Covers a Chrome service worker restart and a Firefox event page unload identically
    // (scope.md 7.4). Reconnect and re-ask for whatever has not come back yet.
    setTimeout(() => {
      if (!allowed) return;
      connect();
      post({ type: "hello" });
      if (calibration !== null) requestOutstanding();
    }, 500);
  });
  post({ type: "hello" });
}

/* ------------------------------------------------------------------ lifecycle */

async function start(): Promise<void> {
  if (calibration === null || !allowed) return;
  chunks = buildChunks(calibration);
  if (chunks.length === 0) return;

  await Promise.all(
    chunks.map(async (chunk) => {
      chunk.hash = await contentHash(chunk.text);
    }),
  );
  updatePriorities();
  requestOutstanding();
  observe();
}

let scrollTimer: ReturnType<typeof setTimeout> | null = null;
function onScroll(): void {
  if (scrollTimer !== null) return;
  scrollTimer = setTimeout(() => {
    scrollTimer = null;
    updatePriorities();
    const priorities: Record<string, number> = {};
    for (const chunk of chunks) if (chunk.logit === null) priorities[chunk.hash] = chunk.priority;
    if (Object.keys(priorities).length > 0) post({ type: "reprioritize", priorities });
  }, 200);
}

let mutationTimer: ReturnType<typeof setTimeout> | null = null;
function observe(): void {
  window.addEventListener("scroll", onScroll, { passive: true });

  // SPAs: re-run extraction on added subtrees, debounced (scope.md 7.1).
  const observer = new MutationObserver((records) => {
    const added = records.some((r) => r.addedNodes.length > 0);
    if (!added) return;
    if (mutationTimer !== null) clearTimeout(mutationTimer);
    mutationTimer = setTimeout(() => {
      mutationTimer = null;
      void rescan();
    }, 500);
  });
  observer.observe(contentRoot(), { childList: true, subtree: true });

  // Port traffic resets the Firefox event-page idle timer (scope.md 6.2). Only while there
  // is on-screen work outstanding, so an idle tab lets the page unload as it should.
  setInterval(() => {
    const pending = chunks.some((c) => c.logit === null && c.priority <= 1);
    if (pending) post({ type: "keepalive" });
  }, KEEPALIVE_MS);
}

/** Re-extract, carrying forward scores we already have. Content-keyed, so cheap. */
async function rescan(): Promise<void> {
  if (calibration === null || !allowed) return;
  const known = new Map(chunks.filter((c) => c.logit !== null).map((c) => [c.hash, c.logit!]));
  const next = buildChunks(calibration);
  await Promise.all(
    next.map(async (chunk) => {
      chunk.hash = await contentHash(chunk.text);
      chunk.logit = known.get(chunk.hash) ?? null;
    }),
  );
  chunks = next;
  updatePriorities();
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
