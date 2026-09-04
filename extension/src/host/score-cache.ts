/**
 * The score cache (scope.md 11): `{hash -> {logit, words, modelVersion, ts}}` in
 * storage.local, LRU-capped.
 *
 * Content-keyed on sha256(normalizeForHash(text)), so text syndicated across dozens of
 * domains -- press releases, the case scope.md 4.5 calls out for leaking across a
 * domain-level split -- is scored once.
 *
 * Entries carry the model version and are dropped when it changes, which is what makes
 * scope.md 4.7 true: a new model invalidates old scores automatically on both browsers.
 */

const KEY = "scoreCache";
const MAX_ENTRIES = 50_000;
/** Evict in bulk: rewriting 50k entries on every insert would be the dominant cost. */
const EVICT_TO = 45_000;

export interface CachedScore {
  readonly logit: number;
  readonly words: number;
  readonly modelVersion: string;
  /** Last access, epoch ms. Drives LRU eviction. */
  ts: number;
}

type CacheMap = Record<string, CachedScore>;

let memo: CacheMap | null = null;
let flushTimer: ReturnType<typeof setTimeout> | null = null;
let dirty = false;

async function load(): Promise<CacheMap> {
  if (memo !== null) return memo;
  const stored = await chrome.storage.local.get(KEY);
  memo = (stored[KEY] as CacheMap | undefined) ?? {};
  return memo;
}

/** Coalesce writes: scoring a page touches hundreds of entries in a few seconds. */
function scheduleFlush(): void {
  dirty = true;
  if (flushTimer !== null) return;
  flushTimer = setTimeout(() => {
    flushTimer = null;
    if (!dirty || memo === null) return;
    dirty = false;
    void chrome.storage.local.set({ [KEY]: memo });
  }, 2_000);
}

export async function get(hash: string, modelVersion: string): Promise<CachedScore | null> {
  const cache = await load();
  const hit = cache[hash];
  if (hit === undefined || hit.modelVersion !== modelVersion) return null;
  hit.ts = Date.now();
  scheduleFlush();
  return hit;
}

export async function put(hash: string, entry: CachedScore): Promise<void> {
  const cache = await load();
  cache[hash] = entry;
  const keys = Object.keys(cache);
  if (keys.length > MAX_ENTRIES) {
    keys.sort((a, b) => cache[a]!.ts - cache[b]!.ts);
    for (const key of keys.slice(0, keys.length - EVICT_TO)) delete cache[key];
  }
  scheduleFlush();
}

/** Drop everything. Backs the options page's "clear cache". */
export async function clear(): Promise<void> {
  memo = {};
  dirty = false;
  await chrome.storage.local.remove(KEY);
}

/** Drop entries from other model versions. Cheap to call at startup. */
export async function evictOtherVersions(modelVersion: string): Promise<number> {
  const cache = await load();
  let removed = 0;
  for (const [hash, entry] of Object.entries(cache)) {
    if (entry.modelVersion !== modelVersion) {
      delete cache[hash];
      removed += 1;
    }
  }
  if (removed > 0) scheduleFlush();
  return removed;
}

export async function size(): Promise<number> {
  return Object.keys(await load()).length;
}
