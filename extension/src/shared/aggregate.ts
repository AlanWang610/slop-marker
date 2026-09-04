/**
 * scope.md 8: chunk logits -> highlighted runs.
 *
 * Port of training/src/slopmarker/eval/aggregate.py, pinned by fixtures/aggregate.json.
 * Every ambiguity in scope.md 8 is resolved on the Python side and THE FIXTURE IS THE
 * SPECIFICATION -- if you disagree with a choice, change it there and regenerate, don't
 * reinterpret it here.
 *
 * Resolved ambiguities:
 *   - "shrunk toward 0.5 by words/80" is linear in probability space, capped at 1.
 *   - "fewer than 20% of chunks exceed t_off" uses a strict `>`.
 *   - the document-prior bump is applied in log-odds space (see calibration.ts).
 *   - a run's score is the unweighted mean of its chunks' log-odds, after the length
 *     penalty, converted back to a probability.
 *   - scope.md 8's "hysteresis" is spatial, not temporal: a run opens at a chunk >= t_on
 *     and extends through neighbours >= t_off. There is no state carried between calls, so
 *     re-scoring a page after a DOM mutation always gives the same answer.
 *
 * Watch the asymmetry between the three comparisons; it is deliberate:
 *   doc prior   strict `>`  against t_off
 *   run extend        `>=`  against t_off
 *   run open          `>=`  against the *effective* t_on
 */

import { type Calibration, logit, sigmoid } from "./calibration.js";

export interface Chunk {
  readonly logit: number;
  readonly words: number;
}

export interface Run {
  /** index of the first chunk, inclusive */
  readonly start: number;
  /** index of the last chunk, inclusive */
  readonly end: number;
  readonly words: number;
  readonly score: number;
  readonly flagged: boolean;
}

export interface AggregateResult {
  readonly t_on_effective: number;
  /** after temperature, before the length penalty */
  readonly chunk_p: number[];
  /** what the thresholds are compared against */
  readonly chunk_p_penalized: number[];
  readonly runs: Run[];
}

export function flaggedRuns(result: AggregateResult): Run[] {
  return result.runs.filter((r) => r.flagged);
}

/** Shrink short chunks toward 0.5. Detectors are near chance on short spans. */
export function applyLengthPenalty(p: number, words: number, pivot: number): number {
  if (words >= pivot) return p;
  return 0.5 + (p - 0.5) * (words / pivot);
}

export function aggregate(chunks: readonly Chunk[], cal: Calibration): AggregateResult {
  const params = cal.aggregate;
  const chunkP = chunks.map((c) => sigmoid(c.logit / cal.temperature));
  const penalized = chunkP.map((p, i) =>
    applyLengthPenalty(p, chunks[i]!.words, params.length_penalty_words),
  );

  // Document prior: on a page with little flagged material, demand more confidence.
  let tOn = cal.t_on;
  if (chunks.length > 0) {
    const above = penalized.filter((p) => p > cal.t_off).length; // strict >
    if (above < params.doc_prior_min_fraction * chunks.length) {
      tOn = sigmoid(logit(cal.t_on) + params.doc_prior_bump_logodds);
    }
  }

  const runs: Run[] = [];
  let i = 0;
  while (i < penalized.length) {
    if (penalized[i]! < tOn) {
      i += 1;
      continue;
    }
    // A run opens here and extends over neighbours that clear the lower threshold.
    let start = i;
    let end = i;
    while (end + 1 < penalized.length && penalized[end + 1]! >= cal.t_off) end += 1;
    while (start - 1 >= 0 && penalized[start - 1]! >= cal.t_off) {
      const last = runs[runs.length - 1];
      // don't absorb chunks already claimed by the previous run
      if (last !== undefined && start - 1 <= last.end) break;
      start -= 1;
    }

    let words = 0;
    for (let k = start; k <= end; k++) words += chunks[k]!.words; // RAW words, not penalized
    let sumLogOdds = 0;
    for (let k = start; k <= end; k++) sumLogOdds += logit(penalized[k]!);
    const meanLogOdds = sumLogOdds / (end - start + 1); // unweighted

    runs.push({
      start,
      end,
      words,
      score: sigmoid(meanLogOdds),
      flagged: words >= params.run_min_words,
    });
    i = end + 1;
  }

  return {
    t_on_effective: tOn,
    chunk_p: chunkP,
    chunk_p_penalized: penalized,
    runs,
  };
}
