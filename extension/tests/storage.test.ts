/**
 * `chrome.storage.local`, or a service-worker proxy where it does not exist.
 *
 * This module exists because of the bug that hid longest: a Chrome offscreen document is
 * granted only `chrome.runtime`, so `chrome.storage` is undefined there. `offscreen.js`
 * threw a TypeError on its first statement, before registering `onConnect`, and a page that
 * is never scored looks exactly like a page with no AI text on it.
 *
 * `local` is resolved once at import, so each case here re-imports the module against a
 * different global. That is the point: the two branches are chosen at load time, and the
 * one that was never exercised is the one that was broken in production.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type * as StorageNamespace from "../src/shared/storage.js";

type StorageModule = typeof StorageNamespace;

/** Import the module fresh, with whatever `chrome` shape the case needs. */
async function importWith(chrome: unknown): Promise<StorageModule> {
  vi.resetModules();
  vi.stubGlobal("chrome", chrome);
  return import("../src/shared/storage.js");
}

function realStorage(): {
  chrome: unknown;
  store: Map<string, unknown>;
  removed: string[][];
} {
  const store = new Map<string, unknown>();
  const removed: string[][] = [];
  return {
    store,
    removed,
    chrome: {
      storage: {
        local: {
          get: (keys: string | string[] | null) => {
            if (keys === null) return Promise.resolve(Object.fromEntries(store));
            const wanted = typeof keys === "string" ? [keys] : keys;
            const out: Record<string, unknown> = {};
            for (const key of wanted) if (store.has(key)) out[key] = store.get(key);
            return Promise.resolve(out);
          },
          set: (items: Record<string, unknown>) => {
            for (const [k, v] of Object.entries(items)) store.set(k, v);
            return Promise.resolve();
          },
          remove: (keys: string | string[]) => {
            const list = typeof keys === "string" ? [keys] : keys;
            removed.push(list);
            for (const key of list) store.delete(key);
            return Promise.resolve();
          },
        },
      },
    },
  };
}

/** The offscreen document's world: runtime only, no storage. */
function proxied(answer: (message: unknown) => unknown): { chrome: unknown; sent: unknown[] } {
  const sent: unknown[] = [];
  return {
    sent,
    chrome: {
      runtime: {
        sendMessage: (message: unknown) => {
          sent.push(message);
          return Promise.resolve(answer(message));
        },
      },
    },
  };
}

beforeEach(() => vi.resetModules());
afterEach(() => vi.unstubAllGlobals());

describe("where chrome.storage exists", () => {
  it("reports itself as not proxied", async () => {
    const storage = await importWith(realStorage().chrome);
    expect(storage.isProxied).toBe(false);
  });

  it("reads and writes through the real API", async () => {
    const real = realStorage();
    const storage = await importWith(real.chrome);
    await storage.set({ threading: { numThreads: 4 } });
    expect(real.store.get("threading")).toEqual({ numThreads: 4 });
    expect(await storage.get("threading")).toEqual({ threading: { numThreads: 4 } });
  });

  it("removes through the real API", async () => {
    const real = realStorage();
    const storage = await importWith(real.chrome);
    await storage.set({ a: 1, b: 2 });
    await storage.remove(["a"]);
    expect(real.removed).toEqual([["a"]]);
    expect(await storage.get(null)).toEqual({ b: 2 });
  });

  it("sends no runtime messages at all", async () => {
    const real = realStorage();
    const sendMessage = vi.fn();
    const storage = await importWith({
      ...(real.chrome as object),
      runtime: { sendMessage },
    });
    await storage.set({ a: 1 });
    await storage.get("a");
    expect(sendMessage).not.toHaveBeenCalled();
  });
});

describe("where chrome.storage is missing (the offscreen document)", () => {
  it("reports itself as proxied", async () => {
    const storage = await importWith(proxied(() => ({ ok: true })).chrome);
    expect(storage.isProxied).toBe(true);
  });

  it("importing does not throw, which is the bug this file exists for", async () => {
    await expect(importWith({ runtime: {} })).resolves.toBeDefined();
  });

  it("routes get through the service worker and unwraps items", async () => {
    const proxy = proxied(() => ({ ok: true, items: { threading: 4 } }));
    const storage = await importWith(proxy.chrome);
    expect(await storage.get("threading")).toEqual({ threading: 4 });
    expect(proxy.sent).toEqual([{ type: "storage", op: "get", keys: "threading" }]);
  });

  it("routes set through the service worker", async () => {
    const proxy = proxied(() => ({ ok: true }));
    const storage = await importWith(proxy.chrome);
    await storage.set({ a: 1 });
    expect(proxy.sent).toEqual([{ type: "storage", op: "set", items: { a: 1 } }]);
  });

  it("routes remove through the service worker", async () => {
    const proxy = proxied(() => ({ ok: true }));
    const storage = await importWith(proxy.chrome);
    await storage.remove(["a", "b"]);
    expect(proxy.sent).toEqual([{ type: "storage", op: "remove", keys: ["a", "b"] }]);
  });

  it("returns an empty object when the worker answers without items", async () => {
    const storage = await importWith(proxied(() => ({ ok: true })).chrome);
    expect(await storage.get(null)).toEqual({});
  });

  it("throws, naming the op, when nothing answers", async () => {
    const storage = await importWith(proxied(() => undefined).chrome);
    await expect(storage.get("a")).rejects.toThrow(/storage\.get failed: no answer/);
  });

  it("throws, naming the op, when the worker reports failure", async () => {
    const storage = await importWith(
      proxied(() => ({ ok: false, message: "quota exceeded" })).chrome,
    );
    await expect(storage.set({ a: 1 })).rejects.toThrow(/storage\.set failed: quota exceeded/);
  });
});

describe("handleProxyMessage, served by the router", () => {
  const serve = async () => {
    const real = realStorage();
    const storage = await importWith(real.chrome);
    return { storage, ...real };
  };

  it("ignores messages that are not storage requests", async () => {
    const { storage } = await serve();
    const sendResponse = vi.fn();
    expect(storage.handleProxyMessage({ type: "getStatus" }, sendResponse)).toBe(false);
    expect(sendResponse).not.toHaveBeenCalled();
  });

  it("ignores an unknown storage op rather than answering wrongly", async () => {
    const { storage } = await serve();
    expect(
      storage.handleProxyMessage({ type: "storage", op: "drop" } as never, vi.fn()),
    ).toBe(false);
  });

  it("serves get, keeping the channel open for the async answer", async () => {
    const { storage, store } = await serve();
    store.set("threading", 4);
    const sendResponse = vi.fn();

    expect(
      storage.handleProxyMessage({ type: "storage", op: "get", keys: "threading" } as never, sendResponse),
    ).toBe(true);
    await vi.waitFor(() => expect(sendResponse).toHaveBeenCalled());
    expect(sendResponse).toHaveBeenCalledWith({ ok: true, items: { threading: 4 } });
  });

  it("serves set", async () => {
    const { storage, store } = await serve();
    const sendResponse = vi.fn();
    storage.handleProxyMessage(
      { type: "storage", op: "set", items: { a: 1 } } as never,
      sendResponse,
    );
    await vi.waitFor(() => expect(sendResponse).toHaveBeenCalledWith({ ok: true }));
    expect(store.get("a")).toBe(1);
  });

  it("serves remove", async () => {
    const { storage, store, removed } = await serve();
    store.set("a", 1);
    const sendResponse = vi.fn();
    storage.handleProxyMessage(
      { type: "storage", op: "remove", keys: ["a"] } as never,
      sendResponse,
    );
    await vi.waitFor(() => expect(sendResponse).toHaveBeenCalledWith({ ok: true }));
    expect(removed).toEqual([["a"]]);
  });

  it("reports a failure back rather than leaving the caller hanging", async () => {
    vi.resetModules();
    vi.stubGlobal("chrome", {
      storage: {
        local: {
          get: () => Promise.reject(new Error("disk is on fire")),
          set: () => Promise.resolve(),
          remove: () => Promise.resolve(),
        },
      },
    });
    const storage = await import("../src/shared/storage.js");
    const sendResponse = vi.fn();
    storage.handleProxyMessage({ type: "storage", op: "get", keys: null } as never, sendResponse);
    await vi.waitFor(() =>
      expect(sendResponse).toHaveBeenCalledWith({ ok: false, message: "disk is on fire" }),
    );
  });

  it("round-trips through the proxy, end to end", async () => {
    const real = realStorage();
    const router = await importWith(real.chrome);

    // The offscreen side latches its (missing) storage at import...
    const client = await importWith({
      runtime: {
        sendMessage: (message: unknown) =>
          new Promise((resolve) => {
            router.handleProxyMessage(message as { type: string }, resolve);
          }),
      },
    });

    // ...and only then does the global regain storage, standing in for the fact that these
    // are two contexts with two globals. `handleProxyMessage` reads it live; the client's
    // `local` stays undefined, so it keeps proxying.
    vi.stubGlobal("chrome", {
      ...(real.chrome as object),
      runtime: (globalThis.chrome as unknown as { runtime: unknown }).runtime,
    });

    await client.set({ threading: { numThreads: 4 } });
    expect(real.store.get("threading")).toEqual({ numThreads: 4 });
    expect(await client.get("threading")).toEqual({ threading: { numThreads: 4 } });
  });
});
