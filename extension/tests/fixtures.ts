/** Loader for the cross-language fixtures in <repo>/fixtures, shared by both test suites. */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
export const FIXTURE_DIR = join(here, "..", "..", "fixtures");

export function loadFixture<T>(name: string): T {
  return JSON.parse(readFileSync(join(FIXTURE_DIR, name), "utf-8")) as T;
}

export interface NormalizeFixture {
  version: number;
  note: string;
  cases: Array<{
    name: string;
    input: string;
    collapsed: string;
    normalized: string;
    sha256: string;
    words: number;
  }>;
}

export interface WindowsFixture {
  version: number;
  note: string;
  sentence_cases: Array<{ name: string; input: string; sentences: string[] }>;
  chunk_cases: Array<{
    name: string;
    input: string;
    min_words: number;
    max_words: number;
    chunks: Array<{ text: string; words: number }>;
  }>;
}

export interface AggregateFixture {
  version: number;
  note: string;
  cases: Array<{
    name: string;
    calibration: unknown;
    chunks: Array<{ logit: number; words: number }>;
    expected: {
      t_on_effective: number;
      chunk_p: number[];
      chunk_p_penalized: number[];
      runs: Array<{
        start: number;
        end: number;
        words: number;
        score: number;
        flagged: boolean;
      }>;
    };
  }>;
}
