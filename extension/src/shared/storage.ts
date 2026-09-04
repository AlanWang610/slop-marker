/**
 * `chrome.storage.local`, or a service-worker proxy where it does not exist.
 *
 * A Chrome offscreen document is granted **only `chrome.runtime`**. Measured, by attaching
 * to the live document over CDP:
 *
 *     Object.keys(chrome)  ->  csi, loadTimes, runtime
 *     typeof chrome.storage -> undefined
 *
 * Everything else the host needs is a web API and is present: `caches`, `indexedDB`,
 * `Worker`, `SharedArrayBuffer`, and `crossOriginIsolated === true`. So the architecture
 * survives; only storage has to be routed around.
 *
 * This is the failure that hid the longest. `offscreen.js` threw a TypeError on its very
 * first statement, before registering `onConnect`, so every port got no reply -- and a page
 * that is never scored looks exactly like a page with no AI text on it. The e2e harness had
 * been opening `host.html` as an ordinary *tab*, which does have `chrome.storage`, so it
 * reproduced everything except the bug.
 *
 * Not used by the content script or the extension pages: those are ordinary contexts with
 * the real API, and they reach it directly.
 */

const local = globalThis.chrome?.storage?.local;

/** True in a Chrome offscreen document, false everywhere else. */
export const isProxied = local === undefined;

interface ProxyRequest {
  readonly type: "storage";
  readonly op: "get" | "set" | "remove";
  readonly keys?: string | string[] | null;
  readonly items?: Record<string, unknown>;
}

async function proxy(request: ProxyRequest): Promise<Record<string, unknown>> {
  const response: { ok: boolean; items?: Record<string, unknown>; message?: string } | undefined =
    await chrome.runtime.sendMessage(request);
  if (response === undefined || !response.ok) {
    throw new Error(`storage.${request.op} failed: ${response?.message ?? "no answer"}`);
  }
  return response.items ?? {};
}

export async function get(keys: string | string[] | null): Promise<Record<string, unknown>> {
  if (local !== undefined) return local.get(keys);
  return proxy({ type: "storage", op: "get", keys });
}

export async function set(items: Record<string, unknown>): Promise<void> {
  if (local !== undefined) {
    await local.set(items);
    return;
  }
  await proxy({ type: "storage", op: "set", items });
}

export async function remove(keys: string | string[]): Promise<void> {
  if (local !== undefined) {
    await local.remove(keys);
    return;
  }
  await proxy({ type: "storage", op: "remove", keys });
}

/**
 * Serve a proxied request. Called by the router, which is always a context that has the
 * real API: the service worker on Chrome, the event page on Firefox.
 */
export function handleProxyMessage(
  message: { type: string },
  sendResponse: (response: unknown) => void,
): boolean {
  if (message.type !== "storage") return false;
  const request = message as ProxyRequest;
  const fail = (err: unknown): void =>
    sendResponse({ ok: false, message: err instanceof Error ? err.message : String(err) });

  switch (request.op) {
    case "get":
      void chrome.storage.local
        .get(request.keys ?? null)
        .then((items) => sendResponse({ ok: true, items }), fail);
      return true;
    case "set":
      void chrome.storage.local
        .set(request.items ?? {})
        .then(() => sendResponse({ ok: true }), fail);
      return true;
    case "remove":
      void chrome.storage.local
        .remove(request.keys ?? [])
        .then(() => sendResponse({ ok: true }), fail);
      return true;
    default:
      return false;
  }
}
