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

/**
 * Containers whose text is never scored.
 *
 * `table` is deliberately absent. scope.md 7.1 lists `td` as a *candidate*, and because
 * `isExcluded` asks `closest()`, putting `table` here matched every cell's own ancestor and
 * silently dropped every table cell on the web -- `td` had never once been scored. Layout
 * tables are the residual risk, and the `min_words` floor (~40, from calibration.json)
 * already screens navigation, pricing and spec cells: a cell holding forty words of prose
 * is prose.
 */
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
].join(",");

const EXCLUDED_ROLES = '[role="navigation"],[role="banner"],[role="contentinfo"],[role="search"],[role="form"],[role="complementary"]';

/** Comment widgets, which scope.md 7.1 excludes by name. */
const COMMENT_WIDGETS =
  "#disqus_thread,.disqus,#comments,.comments,.comment-list,[id^='comment-'],[class*='giscus'],[class*='utterances']";

const CANDIDATES = "p,li,blockquote,dd,td";

/**
 * Elements that force a word boundary, because a reader sees one.
 *
 * `<li><p>Alpha.</p><p>Beta.</p></li>` has no whitespace text node between the paragraphs
 * unless the HTML happens to be pretty-printed, so a naive text walk yields "Alpha.Beta."
 * -- one word short, with a token the model never saw in training. Minified markup is the
 * common case on the real web, and it is exactly what a hand-written test page does not
 * reproduce.
 */
const BLOCK_LEVEL = new Set([
  "ADDRESS", "ARTICLE", "ASIDE", "BLOCKQUOTE", "DD", "DETAILS", "DIALOG", "DIV", "DL", "DT",
  "FIELDSET", "FIGCAPTION", "FIGURE", "FOOTER", "FORM", "H1", "H2", "H3", "H4", "H5", "H6",
  "HEADER", "HR", "LI", "MAIN", "NAV", "OL", "P", "PRE", "SECTION", "SUMMARY", "TABLE",
  "TBODY", "TD", "TFOOT", "TH", "THEAD", "TR", "UL",
]);

/** Nearest block-level ancestor of `node`, bounded by `root`. */
function blockAncestor(node: Node, root: Element): Element {
  for (let el = node.parentElement; el !== null && el !== root; el = el.parentElement) {
    if (BLOCK_LEVEL.has(el.tagName)) return el;
  }
  return root;
}

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
 * Within a single block this mirrors `collapseWhitespace` exactly -- same explicit
 * whitespace set, same leading-space suppression, same run collapsing -- and
 * provenance.test.ts pins that equality, because the logic is duplicated rather than reused
 * (the shared version has no way to report provenance).
 *
 * It differs in one deliberate way: crossing a block-level boundary or a `<br>` inserts a
 * space even where the markup has no whitespace node, because a reader sees a word boundary
 * there. `collapseWhitespace(element.textContent)` does not, and would glue "Alpha." to
 * "Beta." on any minified page.
 */
export function collapseWithProvenance(element: Element): {
  text: string;
  provenance: Provenance[];
} {
  const walker = document.createTreeWalker(
    element,
    NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT,
    {
      acceptNode: (node) =>
        node.nodeType === Node.ELEMENT_NODE && isExcluded(node as Element)
          ? NodeFilter.FILTER_REJECT // skips the whole subtree, not just this node
          : NodeFilter.FILTER_ACCEPT,
    },
  );
  const out: string[] = [];
  const provenance: Provenance[] = [];
  let pendingSpace = false;
  let pendingFrom: Provenance | null = null;
  let lastBlock: Element | null = null;

  /** Nothing to anchor an inserted space to yet; the next real codepoint supplies it. */
  const breakWord = (): void => {
    if (out.length === 0) return;
    pendingSpace = true;
    pendingFrom = null;
  };

  for (let node = walker.nextNode(); node !== null; node = walker.nextNode()) {
    if (node.nodeType === Node.ELEMENT_NODE) {
      if ((node as Element).tagName === "BR") breakWord();
      continue;
    }
    const text = node as Text;

    // Entering *or leaving* a block is a word boundary; comparing the nearest block
    // ancestor of consecutive text nodes catches both with one test.
    const block = blockAncestor(text, element);
    if (lastBlock !== null && block !== lastBlock) breakWord();
    lastBlock = block;

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
 * The content root to use for this pass, given the one held from the last.
 *
 * `contentRoot()` prefers the first `<article>`, so re-deriving it every pass is a trap: an
 * SPA that appends `<article>` elements as the reader scrolls narrows the root to the first
 * one it added, and every later article falls outside it and is silently never scored.
 * Measured on a three-batch feed -- batch one highlighted, batches two and three invisible.
 *
 * So the root is chosen once and held. It is re-derived only when the held element has left
 * the document, which is a wholesale client-side navigation rather than an append.
 */
export function pinRoot(held: Element | null, doc: Document = document): Element {
  return held !== null && held.isConnected ? held : contentRoot(doc);
}

/**
 * Scoreable blocks under `root`, in document order.
 *
 * A candidate nested inside another candidate (a `li` inside a `td`) is dropped, so text is
 * never scored twice and never appears in two runs.
 */
export function extractBlocks(root: Element = contentRoot()): Block[] {
  const blocks: Block[] = [];

  for (const element of root.querySelectorAll(CANDIDATES)) {
    if (isExcluded(element)) continue;

    // A candidate inside another candidate: the outer one already covers this text, and
    // `collapseWithProvenance` walks descendants, so keeping both would score it twice --
    // inflating scope.md 8's document prior and letting one paragraph open two runs.
    //
    // The test is on the *parent*. `element.closest(CANDIDATES)` matches the element
    // itself, so the obvious-looking `closest(CANDIDATES) !== element` can never fire.
    if (element.parentElement?.closest(CANDIDATES) != null) continue;

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
