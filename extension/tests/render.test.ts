/**
 * @vitest-environment jsdom
 *
 * Rendering (scope.md 9). De-emphasize, never hide.
 *
 * Both paths are covered here. jsdom has a `CSS` object without `highlights`, so by default
 * these exercise the wrapper-span fallback -- the one a real browser almost never takes and
 * which is therefore the one most likely to rot. The Highlight API path is covered by
 * stubbing `CSS.highlights`, which only became possible once the capability check stopped
 * being latched at import time.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import { clear, effectiveBackground, render, type RenderedRun } from "../src/content/render.js";

const BORDER_ATTR = "data-slop-marker";
const FALLBACK_CLASS = "slop-marker-run";

function paragraph(text = "alpha beta gamma delta"): HTMLParagraphElement {
  document.body.replaceChildren();
  document.body.removeAttribute("style");
  const p = document.createElement("p");
  p.append(document.createTextNode(text));
  document.body.append(p);
  return p;
}

/**
 * The first non-empty text node under `el`. After a fallback paint the block's firstChild is
 * an empty text node left behind by `surroundContents`, followed by the wrapper span -- so a
 * range has to be built against whatever the DOM looks like right now, skipping the empties.
 */
function firstText(el: Element): Text {
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node !== null; node = walker.nextNode()) {
    if ((node as Text).length > 0) return node as Text;
  }
  throw new Error("no text under the element");
}

function rangeOver(node: Node, start: number, end: number): Range {
  const range = document.createRange();
  range.setStart(node, start);
  range.setEnd(node, end);
  return range;
}

function run(element: Element, ranges: Range[], over: Partial<RenderedRun> = {}): RenderedRun {
  return {
    ranges,
    blocks: [element],
    score: 0.93,
    words: 210,
    modelVersion: "mb-base-0.2.0-dev",
    ...over,
  };
}

/** Give `CSS.highlights` a Map and a Highlight class, as a supporting browser would. */
function withHighlightApi(): Map<string, { ranges: Range[] }> {
  const store = new Map<string, { ranges: Range[] }>();
  class FakeHighlight {
    readonly ranges: Range[] = [];
    add(range: Range): void {
      this.ranges.push(range);
    }
  }
  vi.stubGlobal("Highlight", FakeHighlight);
  Object.defineProperty(CSS, "highlights", { value: store, configurable: true });
  return store;
}

afterEach(() => {
  clear();
  vi.unstubAllGlobals();
  if ("highlights" in CSS) {
    Reflect.deleteProperty(CSS, "highlights");
  }
  document.body.replaceChildren();
  document.body.removeAttribute("style");
  document.documentElement.removeAttribute("style");
  document.getElementById("slop-marker-style")?.remove();
});

describe("effectiveBackground", () => {
  it("uses the element's own painted background", () => {
    const p = paragraph();
    p.style.backgroundColor = "rgb(17, 17, 17)";
    expect(effectiveBackground(p)).toBe("rgb(17, 17, 17)");
  });

  it("walks up to the first painted ancestor", () => {
    const p = paragraph();
    p.parentElement!.style.backgroundColor = "rgb(240, 240, 230)";
    expect(effectiveBackground(p)).toBe("rgb(240, 240, 230)");
  });

  it("looks past an explicitly transparent element", () => {
    const p = paragraph();
    p.style.backgroundColor = "transparent";
    p.parentElement!.style.backgroundColor = "rgb(1, 2, 3)";
    expect(effectiveBackground(p)).toBe("rgb(1, 2, 3)");
  });

  it("looks past a zero-alpha rgba, whatever its channels say", () => {
    const p = paragraph();
    p.style.backgroundColor = "rgba(255, 0, 0, 0)";
    p.parentElement!.style.backgroundColor = "rgb(4, 5, 6)";
    expect(effectiveBackground(p)).toBe("rgb(4, 5, 6)");
  });

  it("keeps a partially transparent background rather than looking through it", () => {
    const p = paragraph();
    p.style.backgroundColor = "rgba(255, 0, 0, 0.5)";
    expect(effectiveBackground(p)).toBe("rgba(255, 0, 0, 0.5)");
  });

  it("falls back to the system canvas when nothing on the page paints", () => {
    expect(effectiveBackground(paragraph())).toBe("Canvas");
  });

  it("finds a dark theme set on the root, not just on the block", () => {
    const p = paragraph();
    document.documentElement.style.backgroundColor = "rgb(9, 9, 11)";
    expect(effectiveBackground(p)).toBe("rgb(9, 9, 11)");
  });
});

describe("the secondary cue on run blocks (scope.md 9)", () => {
  it("marks each block and mixes against that block's own background", () => {
    const p = paragraph();
    p.style.backgroundColor = "rgb(17, 17, 17)";
    render([run(p, [rangeOver(p.firstChild!, 0, 5)])]);

    expect(p.hasAttribute(BORDER_ATTR)).toBe(true);
    expect(p.style.getPropertyValue("--slop-marker-bg")).toBe("rgb(17, 17, 17)");
  });

  it("carries the tooltip scope.md 7.5 asks for: score, words, model version", () => {
    const p = paragraph();
    render([run(p, [rangeOver(p.firstChild!, 0, 5)])]);
    const title = p.getAttribute("title")!;
    expect(title).toContain("0.93");
    expect(title).toContain("210");
    expect(title).toContain("mb-base-0.2.0-dev");
  });

  it("installs its stylesheet once, not once per paint", () => {
    const p = paragraph();
    for (let i = 0; i < 3; i++) render([run(p, [rangeOver(firstText(p), 0, 5)])]);
    expect(document.querySelectorAll("#slop-marker-style")).toHaveLength(1);
  });

  it("uses no opacity and no filter, which would dim images too", () => {
    const p = paragraph();
    render([run(p, [rangeOver(p.firstChild!, 0, 5)])]);
    const css = document.getElementById("slop-marker-style")!.textContent ?? "";
    expect(css).not.toMatch(/(^|[^-\w])opacity\s*:/);
    expect(css).not.toMatch(/(^|[^-\w])filter\s*:/);
    expect(css).toContain("color-mix(in oklab");
  });
});

describe("clear", () => {
  it("removes every trace from the blocks it decorated", () => {
    const p = paragraph();
    p.style.backgroundColor = "rgb(17, 17, 17)";
    render([run(p, [rangeOver(p.firstChild!, 0, 5)])]);
    clear();

    expect(p.hasAttribute(BORDER_ATTR)).toBe(false);
    expect(p.hasAttribute("title")).toBe(false);
    expect(p.style.getPropertyValue("--slop-marker-bg")).toBe("");
  });

  it("is safe to call when nothing was ever rendered", () => {
    expect(() => {
      clear();
      clear();
    }).not.toThrow();
  });

  it("leaves earlier blocks clean when a repaint drops them", () => {
    document.body.replaceChildren();
    const first = document.createElement("p");
    first.append(document.createTextNode("alpha beta"));
    const second = document.createElement("p");
    second.append(document.createTextNode("gamma delta"));
    document.body.append(first, second);

    render([run(first, [rangeOver(first.firstChild!, 0, 5)])]);
    render([run(second, [rangeOver(second.firstChild!, 0, 5)])]);

    expect(first.hasAttribute(BORDER_ATTR)).toBe(false);
    expect(second.hasAttribute(BORDER_ATTR)).toBe(true);
  });
});

describe("wrapper-span fallback", () => {
  it("wraps the range when CSS.highlights is missing", () => {
    const p = paragraph();
    render([run(p, [rangeOver(p.firstChild!, 0, 5)])]);
    const spans = document.querySelectorAll(`.${FALLBACK_CLASS}`);
    expect(spans).toHaveLength(1);
    expect(spans[0]!.textContent).toBe("alpha");
  });

  it("unwraps on clear, restoring the text", () => {
    const p = paragraph();
    render([run(p, [rangeOver(p.firstChild!, 0, 5)])]);
    clear();
    expect(document.querySelectorAll(`.${FALLBACK_CLASS}`)).toHaveLength(0);
    expect(p.textContent).toBe("alpha beta gamma delta");
  });

  it("skips a range that crosses element boundaries instead of restructuring the page", () => {
    document.body.replaceChildren();
    const p = document.createElement("p");
    p.append(document.createTextNode("alpha "));
    const em = document.createElement("em");
    em.append(document.createTextNode("beta"));
    p.append(em, document.createTextNode(" gamma"));
    document.body.append(p);

    const range = document.createRange();
    range.setStart(p.firstChild!, 0);
    range.setEnd(em.firstChild!, 4);

    expect(() => render([run(p, [range])])).not.toThrow();
    expect(p.textContent).toBe("alpha beta gamma");
  });

  it("does not accumulate wrappers across repaints", () => {
    const p = paragraph();
    for (let i = 0; i < 3; i++) render([run(p, [rangeOver(firstText(p), 0, 5)])]);
    expect(document.querySelectorAll(`.${FALLBACK_CLASS}`)).toHaveLength(1);
  });

  it("re-joins the text nodes it split, so the page is not fragmented by repainting", () => {
    const p = paragraph();
    render([run(p, [rangeOver(firstText(p), 0, 5)])]);
    clear();
    expect(p.childNodes).toHaveLength(1);
    expect(p.firstChild!.textContent).toBe("alpha beta gamma delta");
  });
});

describe("CSS Custom Highlight API path", () => {
  it("registers one highlight holding every range", () => {
    const store = withHighlightApi();
    const p = paragraph();
    render([
      run(p, [rangeOver(p.firstChild!, 0, 5), rangeOver(p.firstChild!, 6, 10)]),
    ]);

    const highlight = store.get("slop-marker")!;
    expect(highlight.ranges).toHaveLength(2);
    expect(document.querySelectorAll(`.${FALLBACK_CLASS}`)).toHaveLength(0);
  });

  it("does not mutate the DOM", () => {
    withHighlightApi();
    const p = paragraph();
    const before = p.innerHTML;
    render([run(p, [rangeOver(p.firstChild!, 0, 5)])]);
    expect(p.innerHTML).toBe(before);
  });

  it("deletes the highlight on clear", () => {
    const store = withHighlightApi();
    const p = paragraph();
    render([run(p, [rangeOver(p.firstChild!, 0, 5)])]);
    clear();
    expect(store.has("slop-marker")).toBe(false);
  });

  it("replaces rather than accumulates across paints", () => {
    const store = withHighlightApi();
    const p = paragraph();
    render([run(p, [rangeOver(p.firstChild!, 0, 5)])]);
    render([run(p, [rangeOver(p.firstChild!, 6, 10)])]);
    expect(store.get("slop-marker")!.ranges).toHaveLength(1);
  });

  it("still applies the block-level border cue", () => {
    withHighlightApi();
    const p = paragraph();
    render([run(p, [rangeOver(p.firstChild!, 0, 5)])]);
    expect(p.hasAttribute(BORDER_ATTR)).toBe(true);
  });
});

describe("rendering nothing", () => {
  it("clears a previous paint", () => {
    const p = paragraph();
    render([run(p, [rangeOver(p.firstChild!, 0, 5)])]);
    render([]);
    expect(p.hasAttribute(BORDER_ATTR)).toBe(false);
    expect(document.querySelectorAll(`.${FALLBACK_CLASS}`)).toHaveLength(0);
  });
});
