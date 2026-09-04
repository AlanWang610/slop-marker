/**
 * The calibration.json contract (scope.md 5), shared with the training side.
 *
 * Port of training/src/slopmarker/eval/calibration.py. Everything the extension needs to
 * turn a logit into a highlight lives here, so that a model refresh is a new bundle and
 * not an extension release (scope.md 4.7).
 *
 * NO CONSTANT IN THIS FILE MAY BE HARD-CODED AT A CALL SITE. The defaults below exist only
 * to fill an absent `aggregate` block, exactly as the Python dataclass defaults do.
 */

export const SCHEMA_VERSION = 1;

export function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

export function logit(p: number, eps = 1e-6): number {
  const clamped = Math.min(Math.max(p, eps), 1 - eps);
  return Math.log(clamped / (1 - clamped));
}

/** Constants for scope.md 8. Defaults are the values scope.md states. */
export interface AggregateParams {
  readonly length_penalty_words: number;
  readonly run_min_words: number;
  readonly doc_prior_min_fraction: number;
  /**
   * scope.md 8 says "raise t_on by +0.05" in probability space. At a <=1% FPR operating
   * point t_on lands near 0.95-0.99, so +0.05 yields an unreachable threshold above 1.0 and
   * every sparse document silently stops being scoreable. The bump is applied in log-odds
   * space instead, where it is scale-free and cannot leave the unit interval.
   */
  readonly doc_prior_bump_logodds: number;
}

export const DEFAULT_AGGREGATE_PARAMS: AggregateParams = {
  length_penalty_words: 80,
  run_min_words: 150,
  doc_prior_min_fraction: 0.2,
  doc_prior_bump_logodds: 0.25,
};

export interface Calibration {
  readonly version: string;
  readonly temperature: number;
  readonly t_on: number;
  readonly t_off: number;
  readonly min_words: number;
  readonly max_length: number;
  readonly aggregate: AggregateParams;
}

function requireNumber(raw: Record<string, unknown>, key: string): number {
  const value = raw[key];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`calibration.json: "${key}" must be a finite number, got ${String(value)}`);
  }
  return value;
}

/**
 * Parse and validate a calibration.json payload.
 *
 * Mirrors the Python `__post_init__` invariants. Unlike the Python reader this tolerates
 * unknown top-level keys, so a future bundle carrying an extra field does not brick an
 * installed extension -- the version string is what gates behaviour.
 */
export function parseCalibration(text: string): Calibration {
  const raw: unknown = JSON.parse(text);
  if (typeof raw !== "object" || raw === null) {
    throw new Error("calibration.json: expected a JSON object");
  }
  const obj = raw as Record<string, unknown>;

  const schema = obj["schema_version"];
  if (schema !== undefined && schema !== SCHEMA_VERSION) {
    throw new Error(`calibration.json: unsupported schema_version ${JSON.stringify(schema)}`);
  }
  if (typeof obj["version"] !== "string" || obj["version"].length === 0) {
    throw new Error('calibration.json: "version" must be a non-empty string');
  }

  const rawAgg = (obj["aggregate"] ?? {}) as Record<string, unknown>;
  const aggregate: AggregateParams = {
    length_penalty_words:
      (rawAgg["length_penalty_words"] as number | undefined) ??
      DEFAULT_AGGREGATE_PARAMS.length_penalty_words,
    run_min_words:
      (rawAgg["run_min_words"] as number | undefined) ?? DEFAULT_AGGREGATE_PARAMS.run_min_words,
    doc_prior_min_fraction:
      (rawAgg["doc_prior_min_fraction"] as number | undefined) ??
      DEFAULT_AGGREGATE_PARAMS.doc_prior_min_fraction,
    doc_prior_bump_logodds:
      (rawAgg["doc_prior_bump_logodds"] as number | undefined) ??
      DEFAULT_AGGREGATE_PARAMS.doc_prior_bump_logodds,
  };

  const cal: Calibration = {
    version: obj["version"],
    temperature: requireNumber(obj, "temperature"),
    t_on: requireNumber(obj, "t_on"),
    t_off: requireNumber(obj, "t_off"),
    min_words: (obj["min_words"] as number | undefined) ?? 40,
    max_length: (obj["max_length"] as number | undefined) ?? 512,
    aggregate,
  };

  if (cal.temperature <= 0) {
    throw new Error(`calibration.json: temperature must be positive, got ${cal.temperature}`);
  }
  if (!(cal.t_off > 0 && cal.t_off < cal.t_on && cal.t_on < 1)) {
    throw new Error(
      `calibration.json: need 0 < t_off < t_on < 1, got t_off=${cal.t_off} t_on=${cal.t_on}`,
    );
  }
  if (cal.aggregate.length_penalty_words <= 0) {
    throw new Error("calibration.json: aggregate.length_penalty_words must be positive");
  }
  return cal;
}

/** Derive t_off from t_on in log-odds space. */
export function tOffFrom(tOn: number, deltaLogOdds: number): number {
  return sigmoid(logit(tOn) - deltaLogOdds);
}
