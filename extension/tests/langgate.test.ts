/**
 * scope.md 7.2's language-gate assertion.
 *
 * The gate is deliberately not a coupling point: the extension bundles a franc-class
 * detector, the harvest uses py3langid, and they need not agree block for block. What
 * scope.md 7.2 does ask for is narrower and much more useful:
 *
 *   "asserts both that the two gates agree on acceptance, and, more importantly, that the
 *    cases where they disagree are not systematically non-native."
 *
 * The second half is the one with teeth. franc is weak on short non-native English, and a
 * gate that skips what it cannot identify would silently never score exactly the prose
 * scope.md 4.2 over-samples as its hardest negative -- trained on, never scored, and
 * invisible in every false-positive number this project has. A gap like that cannot be
 * found by looking at FPR; it can only be found here.
 *
 * The English cases are real corpus windows, not constructed ones, because the question is
 * whether a detector trips on real second-language prose.
 */

import { describe, expect, it } from "vitest";

import { detect, isScoreable } from "../src/content/langgate.js";
import { loadFixture } from "./fixtures.js";

interface LangGateFixture {
  version: number;
  min_score: number;
  counts: Record<string, number>;
  cases: Array<{
    group: "non_native_english" | "native_english" | "non_english";
    text: string;
    note: string | null;
    python: { lang: string; score: number; is_english: boolean };
  }>;
}

const fx = loadFixture<LangGateFixture>("langgate.json");
const group = (name: string) => fx.cases.filter((c) => c.group === name);

/** Share of a group the extension is willing to score. */
function acceptRate(name: string): number {
  const rows = group(name);
  return rows.filter((c) => isScoreable(c.text)).length / rows.length;
}

describe("language gate", () => {
  it("has enough of each group to say anything", () => {
    expect(group("non_native_english").length).toBeGreaterThanOrEqual(40);
    expect(group("native_english").length).toBeGreaterThanOrEqual(40);
    expect(group("non_english").length).toBeGreaterThanOrEqual(10);
  });

  it("never scores non-English", () => {
    const leaked = group("non_english").filter((c) => isScoreable(c.text));
    expect(leaked.map((c) => c.note)).toEqual([]);
  });

  it("agrees with the training gate on non-English", () => {
    // Both must reject; the fixture records that Python does.
    for (const c of group("non_english")) {
      expect(c.python.is_english, `python accepted ${c.note}`).toBe(false);
      expect(detect(c.text), `extension accepted ${c.note}`).not.toBe("english");
    }
  });

  it("scores the great majority of English", () => {
    expect(acceptRate("native_english")).toBeGreaterThan(0.9);
    expect(acceptRate("non_native_english")).toBeGreaterThan(0.9);
  });

  /**
   * THE assertion. Not "the gates agree" -- they need not -- but "the extension is not
   * quietly discarding second-language English at a higher rate than native English."
   *
   * The margin is 5 points. Anything larger means the extension is systematically refusing
   * to score the genre the corpus was built to handle, and the shipped false-positive
   * numbers would be describing a population the extension never actually sees.
   */
  it("does not skip non-native English more than native English", () => {
    const native = acceptRate("native_english");
    const nonNative = acceptRate("non_native_english");
    const gap = native - nonNative;
    console.log(
      `  accept: native ${(native * 100).toFixed(1)}%, ` +
        `non-native ${(nonNative * 100).toFixed(1)}%, gap ${(gap * 100).toFixed(1)}pp`,
    );
    expect(gap).toBeLessThan(0.05);
  });

  it("disagrees with the training gate rarely, and not preferentially on non-native", () => {
    const disagreements = fx.cases.filter(
      (c) => c.python.is_english !== (detect(c.text) === "english"),
    );
    const rate = disagreements.length / fx.cases.length;
    const nonNativeShare =
      disagreements.length === 0
        ? 0
        : disagreements.filter((c) => c.group === "non_native_english").length /
          disagreements.length;
    const baseline = group("non_native_english").length / fx.cases.length;
    console.log(
      `  disagreement rate ${(rate * 100).toFixed(1)}%, ` +
        `non-native share ${(nonNativeShare * 100).toFixed(1)}% ` +
        `(baseline ${(baseline * 100).toFixed(1)}%)`,
    );
    // Disagreement is allowed; concentration on non-native is what would be damning.
    if (disagreements.length >= 5) {
      expect(nonNativeShare).toBeLessThan(baseline + 0.25);
    }
  });

  it("treats undetermined as scoreable, on purpose", () => {
    // Documented in langgate.ts: a false accept costs one chunk the length penalty usually
    // discards; a false skip removes the hardest negative from view entirely.
    expect(isScoreable("Short.")).toBe(true);
    expect(detect("Short.")).toBe("undetermined");
  });
});
