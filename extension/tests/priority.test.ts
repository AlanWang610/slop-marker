/**
 * Viewport-distance priority (scope.md 7.4).
 *
 * The host drains the queue lowest-priority-first, so the only thing that matters here is
 * that the number grows with distance from the viewport and never goes below zero. A
 * negative would sort a block the reader has already scrolled past ahead of the one in
 * front of them.
 */

import { describe, expect, it } from "vitest";

import { priorityFor } from "../src/content/pipeline.js";

const HEIGHT = 800;
const rect = (top: number, height = 100): { top: number; bottom: number } => ({
  top,
  bottom: top + height,
});

describe("on screen", () => {
  it("is 0 for a block filling the viewport", () => {
    expect(priorityFor({ top: 0, bottom: HEIGHT }, HEIGHT)).toBe(0);
  });

  it("is 0 for a block in the middle", () => {
    expect(priorityFor(rect(300), HEIGHT)).toBe(0);
  });

  it("is 0 for a block straddling the top edge", () => {
    expect(priorityFor(rect(-50), HEIGHT)).toBe(0);
  });

  it("is 0 for a block straddling the bottom edge", () => {
    expect(priorityFor(rect(HEIGHT - 10), HEIGHT)).toBe(0);
  });

  it("is 0 for a block taller than the viewport", () => {
    expect(priorityFor({ top: -5000, bottom: 5000 }, HEIGHT)).toBe(0);
  });
});

describe("below the viewport", () => {
  it("is 1 one screen down", () => {
    expect(priorityFor({ top: 2 * HEIGHT, bottom: 2 * HEIGHT + 100 }, HEIGHT)).toBe(1);
  });

  it("is 0.5 half a screen down", () => {
    expect(priorityFor({ top: 1.5 * HEIGHT, bottom: 1.5 * HEIGHT + 10 }, HEIGHT)).toBe(0.5);
  });

  it("grows with distance", () => {
    const near = priorityFor(rect(HEIGHT + 100), HEIGHT);
    const far = priorityFor(rect(HEIGHT + 5000), HEIGHT);
    expect(far).toBeGreaterThan(near);
  });
});

describe("above the viewport", () => {
  it("is 1 one screen up", () => {
    expect(priorityFor({ top: -2 * HEIGHT, bottom: -HEIGHT }, HEIGHT)).toBe(1);
  });

  it("is 0.5 half a screen up", () => {
    expect(priorityFor({ top: -1000, bottom: -0.5 * HEIGHT }, HEIGHT)).toBe(0.5);
  });

  it("is symmetric with the same distance below", () => {
    const above = priorityFor({ top: -3 * HEIGHT, bottom: -2 * HEIGHT }, HEIGHT);
    const below = priorityFor({ top: 3 * HEIGHT, bottom: 4 * HEIGHT }, HEIGHT);
    expect(above).toBe(below);
  });
});

describe("invariants", () => {
  it("is never negative, at any offset", () => {
    for (let top = -20_000; top <= 20_000; top += 137) {
      expect(priorityFor(rect(top), HEIGHT)).toBeGreaterThanOrEqual(0);
    }
  });

  it("is monotone in distance on each side", () => {
    let previous = 0;
    for (let top = HEIGHT + 1; top < 20_000; top += 199) {
      const p = priorityFor(rect(top, 1), HEIGHT);
      expect(p).toBeGreaterThanOrEqual(previous);
      previous = p;
    }
  });

  it("ranks the on-screen block ahead of everything else", () => {
    const onScreen = priorityFor(rect(10), HEIGHT);
    for (const top of [-9000, -2000, HEIGHT + 1, 9000]) {
      expect(priorityFor(rect(top, 1), HEIGHT)).toBeGreaterThan(onScreen);
    }
  });

  it("survives a zero-height viewport rather than dividing by zero", () => {
    expect(Number.isFinite(priorityFor(rect(500), 0))).toBe(true);
  });
});
