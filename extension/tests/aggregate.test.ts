import { describe, expect, it } from "vitest";

import { aggregate } from "../src/shared/aggregate.js";
import { parseCalibration } from "../src/shared/calibration.js";
import { type AggregateFixture, loadFixture } from "./fixtures.js";

const fixture = loadFixture<AggregateFixture>("aggregate.json");

/**
 * The fixture carries full-precision floats produced by CPython. Both languages use IEEE
 * 754 doubles and the same operation order, so agreement should be near-exact; 1e-12 leaves
 * room for the last-bit differences in Math.exp/Math.log between libm implementations
 * without letting a real divergence through.
 */
const TOL = 1e-12;

describe("aggregate parity with fixtures/aggregate.json", () => {
  it("has cases to check", () => {
    expect(fixture.cases.length).toBeGreaterThan(0);
  });

  for (const c of fixture.cases) {
    describe(c.name, () => {
      const cal = parseCalibration(JSON.stringify(c.calibration));
      const got = aggregate(c.chunks, cal);

      it("t_on_effective", () => {
        expect(got.t_on_effective).toBeCloseTo(c.expected.t_on_effective, 12);
      });

      it("chunk_p", () => {
        expect(got.chunk_p).toHaveLength(c.expected.chunk_p.length);
        got.chunk_p.forEach((p, i) => {
          expect(Math.abs(p - c.expected.chunk_p[i]!)).toBeLessThan(TOL);
        });
      });

      it("chunk_p_penalized", () => {
        expect(got.chunk_p_penalized).toHaveLength(c.expected.chunk_p_penalized.length);
        got.chunk_p_penalized.forEach((p, i) => {
          expect(Math.abs(p - c.expected.chunk_p_penalized[i]!)).toBeLessThan(TOL);
        });
      });

      it("runs", () => {
        expect(got.runs).toHaveLength(c.expected.runs.length);
        got.runs.forEach((run, i) => {
          const want = c.expected.runs[i]!;
          expect({
            start: run.start,
            end: run.end,
            words: run.words,
            flagged: run.flagged,
          }).toEqual({
            start: want.start,
            end: want.end,
            words: want.words,
            flagged: want.flagged,
          });
          expect(Math.abs(run.score - want.score)).toBeLessThan(TOL);
        });
      });
    });
  }
});

describe("invariants the Python suite also asserts", () => {
  it("runs are disjoint and ordered", () => {
    for (const c of fixture.cases) {
      const cal = parseCalibration(JSON.stringify(c.calibration));
      const runs = aggregate(c.chunks, cal).runs;
      for (let i = 1; i < runs.length; i++) {
        expect(runs[i]!.start).toBeGreaterThan(runs[i - 1]!.end);
      }
      for (const r of runs) expect(r.end).toBeGreaterThanOrEqual(r.start);
    }
  });

  it("a run under run_min_words never flags", () => {
    for (const c of fixture.cases) {
      const cal = parseCalibration(JSON.stringify(c.calibration));
      for (const r of aggregate(c.chunks, cal).runs) {
        if (r.words < cal.aggregate.run_min_words) expect(r.flagged).toBe(false);
      }
    }
  });

  it("the doc-prior bump stays inside (0, 1)", () => {
    for (const c of fixture.cases) {
      const cal = parseCalibration(JSON.stringify(c.calibration));
      const t = aggregate(c.chunks, cal).t_on_effective;
      expect(t).toBeGreaterThan(0);
      expect(t).toBeLessThan(1);
      expect(t).toBeGreaterThanOrEqual(cal.t_on - 1e-12);
    }
  });
});
