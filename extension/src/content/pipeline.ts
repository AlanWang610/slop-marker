/**
 * The content script's pure middle: blocks in, chunks out, runs out (scope.md 7.3, 8, 9).
 *
 * Split out of index.ts so it can be tested without a browser. index.ts owns everything
 * that talks to the page or the port -- observers, the port lifecycle, storage -- and
 * nothing in here touches either. Given the same blocks and the same logits, these
 * functions always produce the same runs, which is the property scope.md 8 relies on when
 * it calls the hysteresis *spatial* rather than temporal.
 */

import { aggregate, type Chunk } from "../shared/aggregate.js";
import type { Calibration } from "../shared/calibration.js";
import { chunkSpans, MAX_WORDS } from "../shared/chunking.js";
import { countWords, toCodePoints } from "../shared/normalize.js";
import { type Block, rangeFor } from "./extract.js";
import { isScoreable } from "./langgate.js";
import type { RenderedRun } from "./render.js";

/** One scoreable unit: a span of one block's collapsed text. */
export interface PageChunk {
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

/**
 * Blocks -> chunks, dropping what must never be scored: blocks under `min_words`
 * (scope.md 7.1) and blocks the language gate does not accept (scope.md 7.2).
 *
 * Takes blocks rather than calling `extractBlocks` itself, so a test can hand it a page.
 */
export function buildChunks(cal: Calibration, blocks: readonly Block[]): PageChunk[] {
  const out: PageChunk[] = [];
  for (const block of blocks) {
    if (countWords(block.text) < cal.min_words) continue;
    if (!isScoreable(block.text)) continue;

    const cp = toCodePoints(block.text);
    for (const [start, end] of chunkSpans(block.text, cal.min_words, MAX_WORDS)) {
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

/**
 * Queue priority from viewport distance, in screens (scope.md 7.4). 0 is on-screen; 1 is
 * one screen away in either direction. Never negative -- the host drains lowest-first, so a
 * negative would sort ahead of what the reader is actually looking at.
 */
export function priorityFor(
  rect: { readonly top: number; readonly bottom: number },
  viewportHeight: number,
): number {
  const height = Math.max(viewportHeight, 1);
  if (rect.bottom >= 0 && rect.top <= height) return 0;
  if (rect.top > height) return (rect.top - height) / height;
  return -rect.bottom / height;
}

/**
 * Score every chunk, pool into runs, and turn the flagged ones into something renderable.
 *
 * Returns [] until every chunk has a logit: scope.md 8's document prior counts how many of
 * the *page's* chunks clear `t_off`, so painting a partial page would flash highlights the
 * prior then withdraws.
 */
export function computeRuns(chunks: readonly PageChunk[], cal: Calibration): RenderedRun[] {
  if (chunks.length === 0) return [];
  if (chunks.some((c) => c.logit === null)) return [];

  const input: Chunk[] = chunks.map((c) => ({ logit: c.logit!, words: c.words }));
  const runs: RenderedRun[] = [];

  for (const run of aggregate(input, cal).runs) {
    if (!run.flagged) continue;
    const ranges: Range[] = [];
    const blocks = new Set<Element>();
    for (let i = run.start; i <= run.end; i++) {
      const chunk = chunks[i]!;
      const range = rangeFor(chunk.block, chunk.start, chunk.end);
      // A null range means the DOM moved under us; skip it rather than throwing away the
      // whole run, which may still cover several other blocks.
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
  return runs;
}
