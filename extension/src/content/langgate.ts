/**
 * The language gate (scope.md 7.2). Non-English blocks are never scored.
 *
 * The backbone is English-only, so scoring other languages produces confident noise. This
 * gate is explicitly NOT one of the coupling points -- it need not agree with the
 * training-side py3langid gate block for block -- but scope.md 7.2 names an interaction
 * worth watching, and it is a real one:
 *
 *   franc is weak on short non-native English, and "unknown -> skip" means the extension
 *   may silently never score exactly the prose scope.md 4.2 over-samples as its hardest
 *   negative. Trained on, never scored, and invisible in every false-positive number.
 *
 * So the bias here is deliberately asymmetric, but only where that is safe:
 *
 *   Non-Latin script  -> rejected outright, before franc is consulted at all.
 *   Latin script      -> undetermined counts as English.
 *
 * The split matters. A false accept on Latin text costs one scored chunk that the length
 * penalty and the 150-word run minimum usually discard anyway, and it keeps second-language
 * English in view. A false accept on Chinese is the scope.md 1 non-goal happening.
 *
 * Pinned by fixtures/langgate.json, whose English cases are real corpus windows.
 */

import { franc } from "franc-min";

/**
 * Below this many characters franc is guessing. Blocks this short are already under
 * `min_words` and dropped before they reach here, so in practice this only affects text we
 * would not have scored anyway.
 */
const MIN_CHARS_FOR_DETECTION = 120;

/**
 * Share of a block's *letters* that may sit outside the Latin script before it is refused.
 *
 * This runs first, and it is not a redundant belt to franc's braces. CJK is dense: a
 * paragraph of Chinese is far shorter in characters than the same content in English, so it
 * lands under MIN_CHARS_FOR_DETECTION, comes back "undetermined", and the asymmetric bias
 * above would then score it. Measured on fixtures/langgate.json, which is how that was found.
 *
 * Latin-script English tolerates well above this: loanwords, names and quoted fragments are
 * a few letters, and punctuation, digits and emoji are not letters at all.
 */
const MAX_NON_LATIN_LETTER_FRACTION = 0.2;

const LETTER = /\p{L}/gu;
const LATIN_LETTER = /\p{Script=Latin}/gu;

export type LanguageVerdict = "english" | "undetermined" | "other";

/** Fraction of letters outside the Latin script. 0 when there are no letters at all. */
export function nonLatinLetterFraction(text: string): number {
  const letters = text.match(LETTER)?.length ?? 0;
  if (letters === 0) return 0;
  const latin = text.match(LATIN_LETTER)?.length ?? 0;
  return (letters - latin) / letters;
}

export function detect(text: string): LanguageVerdict {
  // Script first: this is the one judgement that does not need a statistical model, and it
  // is the one franc's length sensitivity gets wrong.
  if (nonLatinLetterFraction(text) > MAX_NON_LATIN_LETTER_FRACTION) return "other";
  if (text.length < MIN_CHARS_FOR_DETECTION) return "undetermined";
  const code = franc(text, { minLength: MIN_CHARS_FOR_DETECTION });
  if (code === "eng") return "english";
  if (code === "und") return "undetermined";
  return "other";
}

/**
 * Whether to score a block.
 *
 * `undetermined` passes. See the module comment: skipping it would preferentially drop
 * non-native English, the population scope.md 2 names as the dominant false-positive risk
 * and the one the corpus was built to handle.
 */
export function isScoreable(text: string): boolean {
  return detect(text) !== "other";
}
