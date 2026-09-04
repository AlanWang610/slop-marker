/**
 * The score cache (scope.md 11): content-keyed, model-versioned, LRU-capped.
 *
 * Two properties are load-bearing and neither had a test. Entries must be invalidated when
 * the model version changes -- that is what makes scope.md 4.7 true, that a new model
 * silently discards old scores rather than mixing operating points on one page. And
 * eviction must drop the *least recently used*, since a cache that evicts the wrong end is
 * worse than no cache at all on exactly the pages a reader keeps returning to.
 *
 * fake-indexeddb rather than a hand-rolled stub, so the cursor walk, the `ts` index and the
 * transaction boundaries are the real ones.
 */

import "fake-indexeddb/auto";

import { beforeEach, describe, expect, it } from "vitest";

import {
  clear,
  evictOldest,
  evictOtherVersions,
  get,
  limits,
  put,
  size,
  type CachedScore,
} from "../src/host/score-cache.js";

const VERSION = "mb-base-0.2.0-dev";

const entry = (over: Partial<CachedScore> = {}): CachedScore => ({
  logit: 1.5,
  words: 210,
  modelVersion: VERSION,
  ts: Date.now(),
  ...over,
});

/** Read a record straight from IndexedDB, to see what the module actually stored. */
function raw(hash: string): Promise<(CachedScore & { hash: string }) | undefined> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("slop-marker", 1);
    request.onsuccess = () => {
      const db = request.result;
      const read = db.transaction("scores", "readonly").objectStore("scores").get(hash);
      read.onsuccess = () => {
        db.close();
        resolve(read.result as (CachedScore & { hash: string }) | undefined);
      };
      read.onerror = () => reject(read.error ?? new Error("read failed"));
    };
    request.onerror = () => reject(request.error ?? new Error("open failed"));
  });
}

const settle = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

beforeEach(async () => {
  await clear();
  limits.max = 50_000;
  limits.evictTo = 45_000;
});

describe("round trip", () => {
  it("returns what was put", async () => {
    await put("h1", entry({ logit: 2.25, words: 300 }));
    const hit = await get("h1", VERSION);
    expect(hit).toMatchObject({ logit: 2.25, words: 300, modelVersion: VERSION });
  });

  it("returns null for a hash that was never stored", async () => {
    expect(await get("missing", VERSION)).toBeNull();
  });

  it("counts what it holds", async () => {
    expect(await size()).toBe(0);
    await put("h1", entry());
    await put("h2", entry());
    expect(await size()).toBe(2);
  });

  it("overwrites rather than duplicating the same hash", async () => {
    await put("h1", entry({ logit: 1 }));
    await put("h1", entry({ logit: 9 }));
    expect(await size()).toBe(1);
    expect((await get("h1", VERSION))!.logit).toBe(9);
  });

  it("clear empties the store", async () => {
    await put("h1", entry());
    await clear();
    expect(await size()).toBe(0);
    expect(await get("h1", VERSION)).toBeNull();
  });
});

describe("model version invalidation (scope.md 4.7)", () => {
  it("misses an entry written by a different model", async () => {
    await put("h1", entry({ modelVersion: "older-model" }));
    expect(await get("h1", VERSION)).toBeNull();
  });

  it("keeps the row so a downgrade still hits it", async () => {
    await put("h1", entry({ modelVersion: "older-model" }));
    await get("h1", VERSION);
    expect(await get("h1", "older-model")).not.toBeNull();
  });

  it("evictOtherVersions removes only foreign versions and reports how many", async () => {
    await put("keep1", entry());
    await put("keep2", entry());
    await put("drop1", entry({ modelVersion: "old-a" }));
    await put("drop2", entry({ modelVersion: "old-b" }));

    expect(await evictOtherVersions(VERSION)).toBe(2);
    expect(await size()).toBe(2);
    expect(await get("keep1", VERSION)).not.toBeNull();
    expect(await get("drop1", "old-a")).toBeNull();
  });

  it("evictOtherVersions is a no-op on a clean cache", async () => {
    await put("h1", entry());
    expect(await evictOtherVersions(VERSION)).toBe(0);
    expect(await size()).toBe(1);
  });
});

describe("LRU bookkeeping", () => {
  it("touches the timestamp on a hit, so a re-read is not evicted first", async () => {
    await put("h1", entry({ ts: 1000 }));
    expect((await raw("h1"))!.ts).toBe(1000);

    await get("h1", VERSION);
    await settle(); // the touch is deliberately fire-and-forget

    expect((await raw("h1"))!.ts).toBeGreaterThan(1000);
  });

  it("does not touch a version miss", async () => {
    await put("h1", entry({ ts: 1000, modelVersion: "old" }));
    await get("h1", VERSION);
    await settle();
    expect((await raw("h1"))!.ts).toBe(1000);
  });
});

describe("eviction", () => {
  const seed = async (n: number): Promise<void> => {
    for (let i = 0; i < n; i++) await put(`h${i}`, entry({ ts: 1000 + i }));
  };

  it("evictOldest drops the least recently used first", async () => {
    await seed(10);
    await evictOldest(4);

    expect(await size()).toBe(6);
    for (let i = 0; i < 4; i++) expect(await get(`h${i}`, VERSION)).toBeNull();
    for (let i = 4; i < 10; i++) expect(await get(`h${i}`, VERSION)).not.toBeNull();
  });

  it("evictOldest ignores a non-positive count", async () => {
    await seed(5);
    await evictOldest(0);
    await evictOldest(-3);
    expect(await size()).toBe(5);
  });

  it("evictOldest asked for more than it holds empties the store", async () => {
    await seed(3);
    await evictOldest(99);
    expect(await size()).toBe(0);
  });

  it("put trims down to evictTo once it passes the cap", async () => {
    limits.max = 10;
    limits.evictTo = 6;
    await seed(10);
    expect(await size()).toBe(10);

    await put("overflow", entry({ ts: 9999 }));
    expect(await size()).toBe(6);
  });

  it("keeps the most recently used entries when it trims", async () => {
    limits.max = 8;
    limits.evictTo = 4;
    await seed(8);
    await put("newest", entry({ ts: 9999 }));

    expect(await get("newest", VERSION)).not.toBeNull();
    expect(await get("h0", VERSION)).toBeNull();
    expect(await get("h7", VERSION)).not.toBeNull();
  });

  it("keeps an entry that was read recently even though it was written first", async () => {
    limits.max = 6;
    limits.evictTo = 3;
    for (let i = 0; i < 6; i++) await put(`h${i}`, entry({ ts: 1000 + i }));

    await get("h0", VERSION); // touch the oldest
    await settle();
    await put("newest", entry({ ts: 9999 }));

    expect(await get("h0", VERSION)).not.toBeNull();
    expect(await get("h1", VERSION)).toBeNull();
  });

  it("does not evict while under the cap", async () => {
    limits.max = 100;
    await seed(20);
    expect(await size()).toBe(20);
  });
});
