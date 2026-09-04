/**
 * Deterministic chunking of a paragraph block into scoreable units (scope.md 7.3).
 *
 * Port of training/src/slopmarker/data/chunking.py, pinned by fixtures/windows.json
 * (`chunk_cases`). Keep it separate in your head from the *training* window sampler
 * (data/windows.py), which is stochastic and is never ported.
 */

import { collapseWhitespace, countWords, toCodePoints } from "./normalize.js";
import { splitSentences } from "./sentences.js";

export const MIN_WORDS = 40;
export const MAX_WORDS = 400;

/**
 * Split one block into <= maxWords chunks at sentence boundaries.
 *
 * Blocks under `minWords` are skipped entirely (return []). A single sentence longer than
 * `maxWords` is emitted alone rather than cut mid-sentence -- that is the only way a chunk
 * may exceed the cap.
 *
 * A trailing chunk may come out under `minWords`. That is deliberate: merging it back into
 * the previous chunk would breach `maxWords`, and scope.md 8 already shrinks short chunks
 * toward 0.5 by `words / 80`, so a short tail can only extend a run, never start one.
 */
export function chunkBlock(
  text: string,
  minWords: number = MIN_WORDS,
  maxWords: number = MAX_WORDS,
): string[] {
  const collapsed = collapseWhitespace(text);
  if (countWords(collapsed) < minWords) return [];

  const cp = toCodePoints(collapsed);
  const chunks: string[] = [];
  let current: string[] = [];
  let currentWords = 0;

  for (const [start, end] of splitSentences(collapsed)) {
    const sentence = cp.slice(start, end).join("");
    const words = countWords(sentence);
    if (current.length > 0 && currentWords + words > maxWords) {
      chunks.push(current.join(" "));
      current = [];
      currentWords = 0;
    }
    current.push(sentence);
    currentWords += words;
  }
  if (current.length > 0) chunks.push(current.join(" "));
  return chunks;
}
