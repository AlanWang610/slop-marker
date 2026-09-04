/**
 * Sentence splitting. Port of training/src/slopmarker/data/sentences.py.
 *
 * Hand-written on purpose, on both sides. This is a cross-language coupling point: the
 * extension has to produce identical boundaries, and every off-the-shelf splitter fails
 * that test. `Intl.Segmenter` implements the same spec as PyICU but against ICU data that
 * differs per browser build and per OS, so its boundaries drift silently between a user's
 * Firefox and the training box.
 *
 * Offsets are CODEPOINT indices, matching Python's string indexing. Everything here works
 * on the `string[]` from `toCodePoints`, never on raw `text[i]`, which would index UTF-16
 * units and shift every offset after an astral character.
 *
 * Pinned by fixtures/windows.json (`sentence_cases`).
 */

import { ELLIPSIS, toCodePoints, WHITESPACE } from "./normalize.js";

/**
 * Tokens that end in a period without ending a sentence. Compared case-sensitively
 * against the word preceding the period, with that period stripped.
 */
export const ABBREVIATIONS: ReadonlySet<string> = new Set(
  `Mr Mrs Ms Dr Prof Sr Jr St Mt Rev Gen Col Lt Sgt Capt Gov Sen Rep Adm
   Inc Ltd Co Corp Dept Univ Ave Blvd Rd No Fig Vol Ch Sec
   Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec
   Mon Tue Tues Wed Thu Thurs Fri Sat Sun
   vs etc al approx ca esp min max ibid cf
   e.g i.e a.m p.m U.S U.K Ph.D M.D B.A M.A`
    .split(/[ \t\r\n]+/)
    .filter((w) => w.length > 0),
);

const TERMINATORS: ReadonlySet<string> = new Set([...".!?", ELLIPSIS]);

/** Closing punctuation that may sit between the terminator and the space. */
const CLOSERS: ReadonlySet<string> = new Set([
  ...`"')]}`,
  String.fromCodePoint(0x2019),
  String.fromCodePoint(0x201d),
]);

/** A sentence may only start with a capital, a digit, or one of these openers. */
const OPENERS: ReadonlySet<string> = new Set([
  ...`"'([{`,
  String.fromCodePoint(0x2018),
  String.fromCodePoint(0x201c),
]);

/**
 * Portable uppercase test.
 *
 * Not `isupper()`/`/\p{Lu}/`: this exact form is what the Python side uses, and it behaves
 * identically in both languages, including for oddities like the German sharp s whose
 * uppercase is two characters.
 */
function isUpper(ch: string): boolean {
  return ch !== ch.toLowerCase() && ch === ch.toUpperCase();
}

/** ASCII digits only, matching the Python side; `\d` and `isdigit()` are both wider. */
function isDigit(ch: string): boolean {
  return ch >= "0" && ch <= "9";
}

/** True if the period at `dot` is part of an abbreviation or an initial. */
function blocksBoundary(cp: readonly string[], dot: number): boolean {
  if (cp[dot] !== ".") return false; // only periods are ambiguous; ! and ? are not
  let start = 0;
  for (let j = dot - 1; j >= 0; j--) {
    if (cp[j] === " ") {
      start = j + 1;
      break;
    }
  }
  const word = cp.slice(start, dot).join("");
  const isInitial = Array.from(word).length === 1 && word.toLowerCase() !== word.toUpperCase();
  return isInitial || ABBREVIATIONS.has(word);
}

/** Return [start, end) codepoint spans, one per sentence, covering all of `text`. */
export function splitSentences(text: string): Array<[number, number]> {
  const cp = toCodePoints(text);
  const spans: Array<[number, number]> = [];
  let start = 0;
  let i = 0;
  const n = cp.length;

  while (i < n) {
    const ch = cp[i]!;
    if (!TERMINATORS.has(ch)) {
      i += 1;
      continue;
    }
    if (blocksBoundary(cp, i)) {
      i += 1;
      continue;
    }
    let end = i + 1;
    while (end < n && TERMINATORS.has(cp[end]!)) end += 1; // "?!", "..."
    while (end < n && CLOSERS.has(cp[end]!)) end += 1; // He said "Hello."
    // A boundary needs a space and then something that can open a sentence.
    if (end < n && cp[end] === " " && end + 1 < n) {
      const nxt = cp[end + 1]!;
      if (isUpper(nxt) || isDigit(nxt) || OPENERS.has(nxt)) {
        spans.push([start, end]);
        start = end + 1;
        i = start;
        continue;
      }
    }
    i = end;
  }

  // Trim trailing whitespace off the last span rather than emitting it.
  let end = n;
  while (end > start && WHITESPACE.has(cp[end - 1]!)) end -= 1;
  if (end > start) spans.push([start, end]);
  return spans;
}

export function sentences(text: string): string[] {
  const cp = toCodePoints(text);
  return splitSentences(text).map(([a, b]) => cp.slice(a, b).join(""));
}
