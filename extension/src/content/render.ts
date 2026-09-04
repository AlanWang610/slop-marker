/**
 * Rendering (scope.md 9). De-emphasize, never hide.
 *
 * Uses the CSS Custom Highlight API so the DOM is not mutated -- no wrapper elements, no
 * layout shift, nothing for a page's own scripts to trip over. Falls back to wrapper spans
 * where `CSS.highlights` is missing.
 *
 * The colour is mixed against the *effective* background rather than assumed to be white,
 * which is what makes this survive dark and tinted themes. No `opacity` and no `filter`:
 * both would dim any images, badges or inline SVG inside the run as well as the text.
 */

const HIGHLIGHT_NAME = "slop-marker";
const STYLE_ID = "slop-marker-style";
const FALLBACK_CLASS = "slop-marker-run";
const BORDER_ATTR = "data-slop-marker";

export interface RenderedRun {
  readonly ranges: readonly Range[];
  /** Blocks the run touches, which carry the secondary cue. */
  readonly blocks: readonly Element[];
  readonly score: number;
  readonly words: number;
  readonly modelVersion: string;
}

const supportsHighlights = typeof CSS !== "undefined" && "highlights" in CSS;

/**
 * First non-transparent computed background-color walking up ancestors.
 *
 * Returns a colour string; `Canvas` is the system default, so an element with no painted
 * ancestor still mixes against whatever the user's theme actually uses.
 */
export function effectiveBackground(element: Element): string {
  for (let node: Element | null = element; node !== null; node = node.parentElement) {
    const background = getComputedStyle(node).backgroundColor;
    if (background === "" || background === "transparent") continue;
    // rgba(...) with a zero alpha is transparent whatever the channels say.
    const alpha = /rgba?\([^)]*,\s*([\d.]+)\s*\)/.exec(background);
    if (alpha !== null && Number(alpha[1]) === 0) continue;
    return background;
  }
  return "Canvas";
}

function ensureStyle(): void {
  if (document.getElementById(STYLE_ID) !== null) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  // color-mix in oklab keeps the mix perceptually even on both light and dark grounds.
  style.textContent = `
::highlight(${HIGHLIGHT_NAME}) {
  color: color-mix(in oklab, currentColor 45%, var(--slop-marker-bg, Canvas));
}
.${FALLBACK_CLASS} {
  color: color-mix(in oklab, currentColor 45%, var(--slop-marker-bg, Canvas));
}
[${BORDER_ATTR}] {
  border-inline-start: 2px dashed
    color-mix(in oklab, currentColor 45%, var(--slop-marker-bg, Canvas));
  padding-inline-start: 0.5em;
}
`;
  (document.head ?? document.documentElement).append(style);
}

let highlight: Highlight | null = null;
const decorated = new Set<Element>();

/** Replace everything currently shown. Rendering is a pure function of the runs. */
export function render(runs: readonly RenderedRun[]): void {
  ensureStyle();
  clear();

  for (const run of runs) {
    for (const block of run.blocks) {
      // Each block mixes against its own background, so a run spanning a callout and body
      // text de-emphasizes correctly in both.
      (block as HTMLElement).style.setProperty("--slop-marker-bg", effectiveBackground(block));
      block.setAttribute(BORDER_ATTR, "");
      block.setAttribute(
        "title",
        `Likely AI-generated — score ${run.score.toFixed(2)}, ${run.words} words, ${run.modelVersion}`,
      );
      decorated.add(block);
    }
  }

  if (supportsHighlights) {
    highlight = new Highlight();
    for (const run of runs) for (const range of run.ranges) highlight.add(range);
    CSS.highlights.set(HIGHLIGHT_NAME, highlight);
    return;
  }

  // Fallback: wrap in a span. Mutates the DOM, so it is second choice.
  for (const run of runs) {
    for (const range of run.ranges) {
      try {
        const span = document.createElement("span");
        span.className = FALLBACK_CLASS;
        range.surroundContents(span);
      } catch {
        // surroundContents throws on ranges that cross element boundaries; skip those
        // rather than restructuring a page we do not own.
      }
    }
  }
}

export function clear(): void {
  if (supportsHighlights) {
    CSS.highlights.delete(HIGHLIGHT_NAME);
    highlight = null;
  } else {
    for (const span of document.querySelectorAll(`.${FALLBACK_CLASS}`)) {
      span.replaceWith(...span.childNodes);
    }
  }
  for (const block of decorated) {
    block.removeAttribute(BORDER_ATTR);
    block.removeAttribute("title");
    (block as HTMLElement).style.removeProperty("--slop-marker-bg");
  }
  decorated.clear();
}
