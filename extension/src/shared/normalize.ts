/**
 * Text normalization. Two functions, two different jobs.
 *
 * Port of training/src/slopmarker/data/normalize.py. Pinned by fixtures/normalize.json;
 * both test suites read that file. If you disagree with a choice here, change the Python
 * side and regenerate the fixture -- do not reinterpret it.
 *
 * `collapseWhitespace` is what the *model* sees: it only regularizes whitespace, so curly
 * quotes and em dashes survive. Those are real signal for AI text.
 *
 * `normalizeForHash` is what the *score cache key* is built from (scope.md 7.4). It folds
 * quotes, dashes and invisibles too, so syndicated text differing only in typography is
 * scored once.
 *
 * Every character class is an explicit codepoint set and no regex `\s` appears anywhere:
 * Python's `\s` matches U+001C-U+001F, JavaScript's matches U+FEFF and not those. Relying
 * on either would guarantee a drift bug that only fires on scraped web text.
 */

/** Iterate codepoints, not UTF-16 code units. Python's `for ch in text` does the same. */
export function toCodePoints(text: string): string[] {
  return Array.from(text);
}

/** Every character we treat as whitespace. Nothing else counts. */
export const WHITESPACE: ReadonlySet<string> = new Set(
  [
    0x09, // tab
    0x0a, // line feed
    0x0b, // vertical tab
    0x0c, // form feed
    0x0d, // carriage return
    0x20, // space
    0x00a0, // no-break space
    0x1680, // ogham space mark
    // en quad .. hair space (0x2000..0x200A); note 0x200B is INVISIBLE, not whitespace
    0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a,
    0x2028, // line separator
    0x2029, // paragraph separator
    0x202f, // narrow no-break space
    0x205f, // medium mathematical space
    0x3000, // ideographic space
  ].map((cp) => String.fromCodePoint(cp)),
);

/**
 * Zero-width and format characters deleted outright. U+200C ZWNJ and U+200D ZWJ are
 * deliberately NOT here: they are load-bearing in emoji sequences and in Indic and Arabic
 * scripts, and deleting them would mangle real text.
 */
export const INVISIBLE: ReadonlySet<string> = new Set(
  [
    0x00ad, // soft hyphen
    0x200b, // zero-width space
    0x2060, // word joiner
    0xfeff, // zero-width no-break space / BOM
  ].map((cp) => String.fromCodePoint(cp)),
);

const SINGLE_QUOTES = [0x2018, 0x2019, 0x201a, 0x201b, 0x2032]; // curly quotes, prime
const DOUBLE_QUOTES = [0x201c, 0x201d, 0x201e, 0x201f, 0x2033]; // curly quotes, double prime
const DASHES = [0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212]; // hyphen .. minus

export const ELLIPSIS = String.fromCodePoint(0x2026);

const FOLD: ReadonlyMap<string, string> = new Map([
  ...SINGLE_QUOTES.map((cp) => [String.fromCodePoint(cp), "'"] as const),
  ...DOUBLE_QUOTES.map((cp) => [String.fromCodePoint(cp), '"'] as const),
  ...DASHES.map((cp) => [String.fromCodePoint(cp), "-"] as const),
]);

/** Map every whitespace character to a space, collapse runs, strip the ends. */
export function collapseWhitespace(text: string): string {
  const out: string[] = [];
  let pendingSpace = false;
  for (const ch of text) {
    if (WHITESPACE.has(ch)) {
      pendingSpace = out.length > 0; // leading whitespace never sets it
    } else {
      if (pendingSpace) {
        out.push(" ");
        pendingSpace = false;
      }
      out.push(ch);
    }
  }
  return out.join("");
}

/** Aggressive normalization used only to build the score-cache key. */
export function normalizeForHash(text: string): string {
  const out: string[] = [];
  for (const ch of text.normalize("NFC")) {
    if (INVISIBLE.has(ch)) continue;
    if (ch === ELLIPSIS) {
      out.push("...");
    } else {
      out.push(FOLD.get(ch) ?? ch);
    }
  }
  return collapseWhitespace(out.join(""));
}

/**
 * sha256 of the normalized form, hex. The score-cache key of scope.md 7.4.
 *
 * Async because it uses Web Crypto, which is what both the content script and the host
 * have. Node 18+ exposes the same `crypto.subtle`, so the fixture test runs unchanged.
 */
export async function contentHash(text: string): Promise<string> {
  const bytes = new TextEncoder().encode(normalizeForHash(text));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Word count, defined on whitespace-collapsed text so it needs no regex at all.
 * This is the one definition used by training, by windowing and by the extension.
 */
export function countWords(text: string): number {
  const collapsed = collapseWhitespace(text);
  return collapsed ? collapsed.split(" ").length : 0;
}
