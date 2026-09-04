import { describe, expect, it } from "vitest";

import { aggregate } from "../src/shared/aggregate.js";
import { logit, parseCalibration, withThresholdOverride } from "../src/shared/calibration.js";
import { loadFixture, type AggregateFixture } from "./fixtures.js";

const fixture = loadFixture<AggregateFixture>("aggregate.json");
const base = parseCalibration(JSON.stringify(fixture.cases[0]!.calibration));

/**
 * The options page exposes a threshold override (scope.md 6.3). The thing to get right is
 * that t_off is not independent: scope.md 4.5 derives it from t_on by a fixed offset in
 * log-odds space, because a fixed offset in probability space is not scale-free.
 */
describe("withThresholdOverride", () => {
  it("preserves the hysteresis band in log-odds space", () => {
    const shipped = logit(base.t_on) - logit(base.t_off);
    for (const t of [0.55, 0.7, 0.84, 0.95, 0.99]) {
      const moved = withThresholdOverride(base, t);
      expect(logit(moved.t_on) - logit(moved.t_off)).toBeCloseTo(shipped, 10);
    }
  });

  it("keeps 0 < t_off < t_on < 1 at every reachable setting", () => {
    for (let t = 0.02; t < 0.999; t += 0.01) {
      const moved = withThresholdOverride(base, t);
      expect(moved.t_off).toBeGreaterThan(0);
      expect(moved.t_off).toBeLessThan(moved.t_on);
      expect(moved.t_on).toBeLessThan(1);
    }
  });

  it("changes nothing else about the calibration", () => {
    const moved = withThresholdOverride(base, 0.95);
    expect(moved.version).toBe(base.version);
    expect(moved.temperature).toBe(base.temperature);
    expect(moved.min_words).toBe(base.min_words);
    expect(moved.aggregate).toEqual(base.aggregate);
  });

  it("raising the threshold never flags more than the default does", () => {
    // The safe direction: a stricter threshold can only remove runs, never add them.
    for (const c of fixture.cases) {
      const cal = parseCalibration(JSON.stringify(c.calibration));
      const shipped = aggregate(c.chunks, cal).runs.filter((r) => r.flagged).length;
      const stricter = aggregate(c.chunks, withThresholdOverride(cal, 0.995)).runs.filter(
        (r) => r.flagged,
      ).length;
      expect(stricter).toBeLessThanOrEqual(shipped);
    }
  });

  it("refuses a value that would invert the band", () => {
    // Clamped, not thrown: an options page cannot be allowed to brick scoring.
    expect(withThresholdOverride(base, 0).t_on).toBeGreaterThan(0);
    expect(withThresholdOverride(base, 1).t_on).toBeLessThan(1);
  });
});
