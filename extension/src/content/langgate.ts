/**
 * The language gate (scope.md 7.2). Non-English blocks are never scored.
 *
 * The backbone is English-only, so scoring other languages would produce confident noise.
 * This gate is explicitly NOT one of the coupling points -- it need not agree with the
 * training-side py3langid gate block for block -- but scope.md 7.2 names an interaction
 * worth watching, and it is a real one:
 *
 *   franc is weak on short non-native English, and "unknown -> skip" means we may silently
 *   never score exactly the prose scope.md 4.2 over-samples as its hardest negative. Text
 *   trained on but never scored is a gap that shows up nowhere in the FPR numbers.
 *
 * So the bias here is deliberately asymmetric. `und` on text long enough to have been
 * judged is treated as English rather than skipped: a false accept costs one scored block
 * that the length penalty and the 150-word run minimum will usually discard anyway, while
 * a false skip silently removes the exact genre we most need to get right.
 */

import { franc } from "franc-min";

/**
 * Below this many characters franc is guessing. scope.md 7.2 wants "unknown -> skip", but
 * a block this short is already under `min_words` and is dropped before it gets here, so
 * in practice the threshold only affects blocks we would not score anyway.
 */
const MIN_CHARS_FOR_DETECTION = 120;

export type LanguageVerdict = "english" | "undetermined" | "other";

export function detect(text: string): LanguageVerdict {
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
 * non-native English, which is the population scope.md 2 names as the dominant
 * false-positive risk and the one the corpus was built to handle.
 */
export function isScoreable(text: string): boolean {
  return detect(text) !== "other";
}
