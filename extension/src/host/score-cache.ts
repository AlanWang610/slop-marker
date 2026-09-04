/**
 * The score cache (scope.md 11): `{hash -> {logit, words, modelVersion, ts}}`, LRU-capped.
 *
 * Content-keyed on `sha256(normalizeForHash(text))`, so text syndicated across dozens of
 * domains -- press releases, the case scope.md 4.5 calls out for leaking across a
 * domain-level split -- is scored once. Entries carry the model version and are dropped
 * when it changes, which is what makes scope.md 4.7 true: a new model invalidates old
 * scores automatically on both browsers.
 *
 * IndexedDB, not `storage.local`, which is a deliberate deviation from scope.md 11 and
 * has two independent reasons:
 *
 *   1. It has to work in a Chrome offscreen document, and those are granted only
 *      `chrome.runtime` -- `chrome.storage` is undefined there (see shared/storage.ts).
 *      The small keys route around that with a service-worker proxy; a 50k-entry cache
 *      cannot, because every flush would ship the whole map across a message port.
 *   2. `storage.local` has no partial access. Reading one score means deserialising the
 *      entire cache and writing one means reserialising it, so a 50k-entry LRU costs more
 *      per hit than it saves. IndexedDB reads and writes single records, and its `ts`
 *      index makes eviction a cursor rather than a full sort.
 *
 * The intent scope.md 11 states -- persisted locally, LRU-capped near 50k, keyed by model
 * version, nothing sent anywhere -- is unchanged.
 */

const DB_NAME = "slop-marker";
const DB_VERSION = 1;
const STORE = "scores";
const TS_INDEX = "ts";

/**
 * LRU bounds (scope.md 11). Mutable so a test can exercise eviction without inserting
 * fifty thousand records; nothing but tests ever writes to it. Evicting in bulk down to
 * `evictTo` is deliberate -- trimming one record per insert would thrash the cursor.
 */
export const limits = { max: 50_000, evictTo: 45_000 };

export interface CachedScore {
  readonly logit: number;
  readonly words: number;
  readonly modelVersion: string;
  /** Last access, epoch ms. Drives LRU eviction. */
  ts: number;
}

interface Record_ extends CachedScore {
  hash: string;
}

let dbPromise: Promise<IDBDatabase> | null = null;

function open(): Promise<IDBDatabase> {
  dbPromise ??= new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: "hash" });
        store.createIndex(TS_INDEX, "ts");
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("indexedDB.open failed"));
  });
  return dbPromise;
}

function tx<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return open().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const transaction = db.transaction(STORE, mode);
        const request = run(transaction.objectStore(STORE));
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error ?? new Error("indexedDB request failed"));
      }),
  );
}

export async function get(hash: string, modelVersion: string): Promise<CachedScore | null> {
  // store.get is typed IDBRequest<any>; the cast is where that becomes a Record_.
  const hit = await tx<Record_ | undefined>(
    "readonly",
    (store) => store.get(hash) as IDBRequest<Record_ | undefined>,
  );
  if (hit === undefined || hit.modelVersion !== modelVersion) return null;
  // Touch for LRU. Fire and forget: a lost touch costs one early eviction, nothing more.
  void tx("readwrite", (store) => store.put({ ...hit, ts: Date.now() })).catch(() => undefined);
  return hit;
}

export async function put(hash: string, entry: CachedScore): Promise<void> {
  await tx("readwrite", (store) => store.put({ hash, ...entry }));
  const count = await size();
  if (count > limits.max) await evictOldest(count - limits.evictTo);
}

/** Delete the `n` least recently used entries, oldest first, via the ts index. */
export async function evictOldest(n: number): Promise<void> {
  if (n <= 0) return;
  const db = await open();
  await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(STORE, "readwrite");
    const cursorRequest = transaction.objectStore(STORE).index(TS_INDEX).openCursor();
    let remaining = n;
    cursorRequest.onsuccess = () => {
      const cursor = cursorRequest.result;
      if (cursor === null || remaining <= 0) return;
      cursor.delete();
      remaining -= 1;
      cursor.continue();
    };
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error("eviction failed"));
  });
}

/** Drop everything. Backs the options page's "clear cache". */
export async function clear(): Promise<void> {
  await tx("readwrite", (store) => store.clear());
}

/** Drop entries from other model versions. Cheap to call at startup. */
export async function evictOtherVersions(modelVersion: string): Promise<number> {
  const db = await open();
  return new Promise<number>((resolve, reject) => {
    const transaction = db.transaction(STORE, "readwrite");
    const cursorRequest = transaction.objectStore(STORE).openCursor();
    let removed = 0;
    cursorRequest.onsuccess = () => {
      const cursor = cursorRequest.result;
      if (cursor === null) return;
      if ((cursor.value as Record_).modelVersion !== modelVersion) {
        cursor.delete();
        removed += 1;
      }
      cursor.continue();
    };
    transaction.oncomplete = () => resolve(removed);
    transaction.onerror = () => reject(transaction.error ?? new Error("eviction failed"));
  });
}

export async function size(): Promise<number> {
  return tx<number>("readonly", (store) => store.count());
}

/**
 * The surface the Host depends on. Declared so a test can hand it a Map-backed stand-in and
 * exercise the queue without an IndexedDB at all.
 */
export interface ScoreCache {
  get(hash: string, modelVersion: string): Promise<CachedScore | null>;
  put(hash: string, entry: CachedScore): Promise<void>;
  clear(): Promise<void>;
  evictOtherVersions(modelVersion: string): Promise<number>;
  size(): Promise<number>;
}

/** Drop the memoized connection. Tests that swap the `indexedDB` global need this. */
export function resetForTests(): void {
  dbPromise = null;
}
