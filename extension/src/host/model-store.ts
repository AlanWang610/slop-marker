/**
 * Fetching, verifying and caching the model bundle (scope.md 6.4).
 *
 * The bundle is not packaged in the extension: it is ~140 MB, and decoupling it from the
 * extension means a model refresh is a new bundle rather than a new release (scope.md 4.7).
 *
 * WHERE THIS RUNS MATTERS. `download()` must run in the router context -- the Chrome
 * service worker or the Firefox event page -- and never in the offscreen document. The
 * manifest sets `cross_origin_embedder_policy: require-corp` so extension *pages* can be
 * cross-origin isolated and ORT can use threads, and under require-corp a cross-origin
 * fetch needs the response to be CORS-successful or carry Cross-Origin-Resource-Policy.
 * GitHub release assets send neither (verified: release-assets.githubusercontent.com
 * returns no Access-Control-Allow-Origin). The Cache API is per-origin and shared across
 * extension contexts, so the router downloads and the offscreen document reads. That costs
 * nothing and sidesteps COEP entirely.
 *
 * Every file is checked against the SHA256SUMS shipped *inside* the extension, not the one
 * downloaded beside it. A tampered or truncated upload fails closed rather than scoring
 * with the wrong weights.
 */

import { BUNDLE_FILES, MODEL_BASE_URL, SHIPPED_SHA256SUMS } from "../shared/bundle-config.js";
import type { ModelState } from "../shared/protocol.js";

const CACHE_NAME = "slop-marker-model";

/** Parse coreutils `sha256sum` output: "<64 hex><two spaces><name>" per line. */
export function parseSha256Sums(text: string): Map<string, string> {
  const out = new Map<string, string>();
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (trimmed.length === 0) continue;
    const match = /^([0-9a-f]{64})\s{1,2}\*?(.+)$/.exec(trimmed);
    if (match === null) throw new Error(`malformed SHA256SUMS line: ${line}`);
    out.set(match[2]!.trim(), match[1]!);
  }
  return out;
}

async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function fileUrl(version: string, file: string): string {
  return `${MODEL_BASE_URL}/${encodeURIComponent(version)}/${encodeURIComponent(file)}`;
}

/**
 * What `download` checks against. Injected only by tests -- production always reads the
 * SHA256SUMS compiled into the extension, which is the whole point of scope.md 6.4: a
 * tampered upload cannot also rewrite the expectation.
 */
export interface BundleSpec {
  readonly files: readonly string[];
  readonly expected: ReadonlyMap<string, string>;
}

export function shippedBundle(): BundleSpec {
  return { files: BUNDLE_FILES, expected: parseSha256Sums(SHIPPED_SHA256SUMS) };
}

/** True when every bundle file for `version` is already cached. */
export async function isCached(
  version: string,
  files: readonly string[] = BUNDLE_FILES,
): Promise<boolean> {
  const cache = await caches.open(CACHE_NAME);
  for (const file of files) {
    if ((await cache.match(fileUrl(version, file))) === undefined) return false;
  }
  return true;
}

/**
 * Download the bundle, verify it, and put it in the Cache API. Router context only.
 * Idempotent: returns immediately when everything is already cached and verified.
 */
export async function download(
  version: string,
  onState: (state: ModelState) => void,
  spec: BundleSpec = shippedBundle(),
): Promise<void> {
  if (await isCached(version, spec.files)) {
    onState({ phase: "ready", version });
    return;
  }

  const expected = spec.expected;
  const cache = await caches.open(CACHE_NAME);

  // Sizes are unknown until each response arrives, so progress is reported against the
  // number of bytes actually seen plus content-length of the file in flight. Good enough
  // for a progress bar and honest about what it knows.
  let received = 0;
  let total = 0;

  for (const file of spec.files) {
    const url = fileUrl(version, file);
    const response = await fetch(url, { cache: "no-store", credentials: "omit" });
    if (!response.ok) {
      throw new Error(`${file}: HTTP ${response.status} from ${url}`);
    }
    const declared = Number(response.headers.get("content-length") ?? 0);
    total += declared;

    const bytes = await response.arrayBuffer();
    received += bytes.byteLength;
    onState({ phase: "downloading", received, total: Math.max(total, received) });

    onState({ phase: "verifying" });
    const want = expected.get(file);
    if (want === undefined) {
      throw new Error(`${file} is not listed in the extension's SHA256SUMS`);
    }
    const got = await sha256Hex(bytes);
    if (got !== want) {
      // Do not cache a file that failed. Fail closed.
      throw new Error(`${file}: checksum mismatch (expected ${want.slice(0, 12)}…, got ${got.slice(0, 12)}…)`);
    }

    await cache.put(url, new Response(bytes));
  }

  onState({ phase: "ready", version });
}

/** Read one cached bundle file as bytes. Throws if absent -- callers should download first. */
export async function readCached(version: string, file: string): Promise<ArrayBuffer> {
  const cache = await caches.open(CACHE_NAME);
  const hit = await cache.match(fileUrl(version, file));
  if (hit === undefined) throw new Error(`${file} is not cached for model ${version}`);
  return hit.arrayBuffer();
}

export async function readCachedText(version: string, file: string): Promise<string> {
  return new TextDecoder().decode(await readCached(version, file));
}

/** Drop every cached bundle file. Backs the options page's "re-download model". */
export async function clearCache(): Promise<void> {
  await caches.delete(CACHE_NAME);
}
