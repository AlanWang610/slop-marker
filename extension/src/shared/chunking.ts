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
  const cp = toCodePoints(collapsed);
  return chunkSpans(collapsed, minWords, maxWords).map(([a, b]) => cp.slice(a, b).join(""));
}

/**
 * The same chunking, as [start, end) codepoint spans into `collapseWhitespace(text)`.
 *
 * The content script needs this: to highlight a run it has to map chunks back to DOM
 * ranges, and it cannot do that from the strings alone. Spans are safe because
 * `splitSentences` partitions the input and its sentences rejoin with exactly one space
 * (`" ".join(sentences) === text.trim()`), so a group of consecutive sentences is always a
 * contiguous span rather than something that merely stringifies the same.
 *
 * chunking.test.ts asserts the two agree on every fixture case, so this cannot drift into
 * being a second, subtly different chunker.
 */
export function chunkSpans(
  collapsed: string,
  minWords: number = MIN_WORDS,
  maxWords: number = MAX_WORDS,
): Array<[number, number]> {
  if (countWords(collapsed) < minWords) return [];

  const cp = toCodePoints(collapsed);
  const spans: Array<[number, number]> = [];
  let start: number | null = null;
  let end = 0;
  let currentWords = 0;

  for (const [sentenceStart, sentenceEnd] of splitSentences(collapsed)) {
    const words = countWords(cp.slice(sentenceStart, sentenceEnd).join(""));
    if (start !== null && currentWords + words > maxWords) {
      spans.push([start, end]);
      start = null;
      currentWords = 0;
    }
    start ??= sentenceStart;
    end = sentenceEnd;
    currentWords += words;
  }
  if (start !== null) spans.push([start, end]);
  return spans;
}
