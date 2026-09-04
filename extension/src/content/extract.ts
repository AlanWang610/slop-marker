/**
 * Pulling scoreable text out of a page (scope.md 7.1), keeping enough provenance to put a
 * highlight back exactly where the text came from.
 *
 * The hard part is not choosing elements, it is the mapping. The model and the chunker
 * both work on `collapseWhitespace(text)`, but a highlight has to land on DOM text nodes at
 * their real offsets. So extraction records, for every codepoint of the collapsed text,
 * which text node it came from and where -- and `rangeFor` walks that back.
 */

import { toCodePoints, WHITESPACE } from "../shared/normalize.js";

/** Containers whose text is never scored. */
const EXCLUDED = [
  "nav",
  "header",
  "footer",
  "aside",
  "code",
  "pre",
  "form",
  "textarea",
  "select",
  "button",
  "figure",
  "figcaption",
  "script",
  "style",
  "noscript",
  "svg",
  "table",
].join(",");

const EXCLUDED_ROLES = '[role="navigation"],[role="banner"],[role="contentinfo"],[role="search"],[role="form"],[role="complementary"]';

/** Comment widgets, which scope.md 7.1 excludes by name. */
const COMMENT_WIDGETS =
  "#disqus_thread,.disqus,#comments,.comments,.comment-list,[id^='comment-'],[class*='giscus'],[class*='utterances']";

const CANDIDATES = "p,li,blockquote,dd,td";

/** One text node's contribution to the collapsed string. */
interface Provenance {
  readonly node: Text;
  /** Offset within `node.data`, in UTF-16 units -- what Range wants. */
  readonly offset: number;
}

export interface Block {
  readonly element: Element;
  /** collapseWhitespace(element text). What the chunker and the model see. */
  readonly text: string;
  /** One entry per codepoint of `text`. */
  readonly provenance: readonly Provenance[];
}

function isExcluded(element: Element): boolean {
  if (element.closest(EXCLUDED) !== null) return true;
  if (element.closest(EXCLUDED_ROLES) !== null) return true;
  if (element.closest("[contenteditable]") !== null) return true;
  if (element.closest(COMMENT_WIDGETS) !== null) return true;
  return false;
}

/** Readability-ish content root. Cheap heuristics, in descending confidence. */
export function contentRoot(doc: Document = document): Element {
  return (
    doc.querySelector("article") ??
    doc.querySelector("main") ??
    doc.querySelector('[role="main"]') ??
    doc.body
  );
}

/**
 * Collapse an element's text while recording where each surviving codepoint came from.
 *
 * This mirrors `collapseWhitespace` exactly -- same explicit whitespace set, same
 * leading-space suppression, same run collapsing. It is duplicated rather than reused
 * because the shared version has no way to report provenance, and content-parity.test.ts
 * asserts the two agree on the resulting string.
 */
export function collapseWithProvenance(element: Element): {
  text: string;
  provenance: Provenance[];
} {
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
  const out: string[] = [];
  const provenance: Provenance[] = [];
  let pendingSpace = false;
  let pendingFrom: Provenance | null = null;

  for (let node = walker.nextNode(); node !== null; node = walker.nextNode()) {
    const text = node as Text;
    const parent = text.parentElement;
    if (parent !== null && isExcluded(parent)) continue;

    // Iterate codepoints, tracking the UTF-16 offset separately: a Range needs UTF-16.
    let unit = 0;
    for (const ch of text.data) {
      if (WHITESPACE.has(ch)) {
        if (out.length > 0) {
          pendingSpace = true;
          pendingFrom ??= { node: text, offset: unit };
        }
      } else {
        if (pendingSpace) {
          out.push(" ");
          provenance.push(pendingFrom ?? { node: text, offset: unit });
          pendingSpace = false;
          pendingFrom = null;
        }
        out.push(ch);
        provenance.push({ node: text, offset: unit });
      }
      unit += ch.length;
    }
  }

  return { text: out.join(""), provenance };
}

/**
 * Scoreable blocks under `root`, in document order.
 *
 * A candidate nested inside another candidate (a `li` inside a `td`) is dropped, so text is
 * never scored twice and never appears in two runs.
 */
export function extractBlocks(root: Element = contentRoot()): Block[] {
  const seen = new Set<Element>();
  const blocks: Block[] = [];

  for (const element of root.querySelectorAll(CANDIDATES)) {
    if (isExcluded(element)) continue;
    if (element.closest(CANDIDATES) !== element) continue; // nested candidate
    if (seen.has(element)) continue;
    seen.add(element);

    const { text, provenance } = collapseWithProvenance(element);
    if (text.length === 0) continue;
    blocks.push({ element, text, provenance });
  }
  return blocks;
}

/**
 * A DOM Range covering [start, end) of a block's collapsed text.
 *
 * Returns null when the span is empty or its provenance was lost to a DOM mutation, which
 * the caller treats as "do not highlight" rather than as an error.
 */
export function rangeFor(block: Block, start: number, end: number): Range | null {
  if (end <= start) return null;
  const first = block.provenance[start];
  const last = block.provenance[end - 1];
  if (first === undefined || last === undefined) return null;
  if (!first.node.isConnected || !last.node.isConnected) return null;

  const range = document.createRange();
  try {
    range.setStart(first.node, Math.min(first.offset, first.node.length));
    // The end offset is exclusive, and the last codepoint may be astral.
    const lastChar = toCodePoints(last.node.data.slice(last.offset))[0] ?? "";
    range.setEnd(last.node, Math.min(last.offset + lastChar.length, last.node.length));
  } catch {
    return null;
  }
  return range;
}
