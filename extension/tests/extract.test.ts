/**
 * @vitest-environment jsdom
 *
 * Element selection (scope.md 7.1). The candidate list, the exclusion list, and the rule
 * that no text is ever scored twice.
 *
 * Two of these started as failing tests:
 *   - `td` is a scope.md 7.1 candidate, but `table` had been added to the exclusion list,
 *     so `td.closest(EXCLUDED)` always matched and no table cell was ever scored.
 *   - the nested-candidate guard read `element.closest(CANDIDATES) !== element`, and
 *     `closest` matches the element itself, so it never fired and `<li><p>x</p></li>`
 *     produced two blocks over the same text.
 */

import { beforeEach, describe, expect, it } from "vitest";

import {
  collapseWithProvenance,
  contentRoot,
  extractBlocks,
  pinRoot,
} from "../src/content/extract.js";

/** Enough words that nothing downstream would drop the block for being short. */
const PROSE =
  "This paragraph is deliberately long enough to be treated as a real block of prose " +
  "rather than a stray fragment of interface text that happens to sit in the markup.";

function page(html: string): void {
  document.body.innerHTML = html;
}

const texts = (root?: Element): string[] =>
  extractBlocks(root ?? document.body).map((b) => b.text);

beforeEach(() => {
  document.body.innerHTML = "";
});

describe("candidates", () => {
  const candidates: Array<[string, string]> = [
    ["p", `<p>${PROSE}</p>`],
    ["li", `<ul><li>${PROSE}</li></ul>`],
    ["blockquote", `<blockquote>${PROSE}</blockquote>`],
    ["dd", `<dl><dt>Term</dt><dd>${PROSE}</dd></dl>`],
    ["td", `<table><tbody><tr><td>${PROSE}</td></tr></tbody></table>`],
  ];

  for (const [name, html] of candidates) {
    it(`extracts ${name}, which scope.md 7.1 lists as a candidate`, () => {
      page(html);
      expect(texts()).toEqual([PROSE]);
    });
  }

  it("extracts several candidates in document order", () => {
    page(`<p>First ${PROSE}</p><blockquote>Second ${PROSE}</blockquote><p>Third ${PROSE}</p>`);
    const out = texts();
    expect(out).toHaveLength(3);
    expect(out[0]!.startsWith("First")).toBe(true);
    expect(out[1]!.startsWith("Second")).toBe(true);
    expect(out[2]!.startsWith("Third")).toBe(true);
  });

  it("skips elements whose text is empty or whitespace only", () => {
    page(`<p></p><p>   \n\t  </p><p>${PROSE}</p>`);
    expect(texts()).toEqual([PROSE]);
  });

  it("collapses whitespace, so what comes out is what the model is fed", () => {
    page("<p>  spaced   out\n\ttext  </p>");
    expect(texts()).toEqual(["spaced out text"]);
  });
});

describe("exclusions", () => {
  const containers = [
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "figure",
    "code",
    "pre",
    "button",
  ];

  for (const tag of containers) {
    it(`never scores text inside <${tag}>`, () => {
      page(`<${tag}><p>${PROSE}</p></${tag}>`);
      expect(texts()).toEqual([]);
    });
  }

  const roles = ["navigation", "banner", "contentinfo", "search", "form", "complementary"];
  for (const role of roles) {
    it(`never scores text inside [role="${role}"]`, () => {
      page(`<div role="${role}"><p>${PROSE}</p></div>`);
      expect(texts()).toEqual([]);
    });
  }

  it("never scores editable text", () => {
    page(`<div contenteditable="true"><p>${PROSE}</p></div>`);
    expect(texts()).toEqual([]);
  });

  const widgets = [
    '<div id="disqus_thread"><p>P</p></div>',
    '<div class="disqus"><p>P</p></div>',
    '<section id="comments"><p>P</p></section>',
    '<section class="comments"><p>P</p></section>',
    '<ol class="comment-list"><li>P</li></ol>',
    '<div id="comment-1421"><p>P</p></div>',
    '<div class="giscus-frame"><p>P</p></div>',
    '<div class="utterances"><p>P</p></div>',
  ];
  for (const widget of widgets) {
    it(`never scores the comment widget ${widget.slice(0, 34)}…`, () => {
      page(widget.replace(/>P</g, `>${PROSE}<`));
      expect(texts()).toEqual([]);
    });
  }

  it("drops an excluded descendant's text but keeps the rest of the block", () => {
    page(`<p>before <code>const x = 1;</code> after</p>`);
    expect(texts()).toEqual(["before after"]);
  });

  it("keeps a candidate that merely follows an excluded sibling", () => {
    page(`<nav><p>nav text</p></nav><p>${PROSE}</p>`);
    expect(texts()).toEqual([PROSE]);
  });
});

describe("no text is scored twice", () => {
  it("keeps only the outer candidate when one nests inside another", () => {
    page(`<ul><li><p>${PROSE}</p></li></ul>`);
    expect(texts()).toEqual([PROSE]);
  });

  it("merges several nested candidates into the one outer block", () => {
    page(`<ul><li><p>Alpha alpha.</p><p>Beta beta.</p></li></ul>`);
    expect(texts()).toEqual(["Alpha alpha. Beta beta."]);
  });

  it("keeps the outer blockquote rather than its paragraphs", () => {
    page(`<blockquote><p>${PROSE}</p></blockquote>`);
    expect(texts()).toEqual([PROSE]);
  });

  it("keeps sibling candidates that do not nest", () => {
    page(`<ul><li>Alpha alpha.</li><li>Beta beta.</li></ul>`);
    expect(texts()).toEqual(["Alpha alpha.", "Beta beta."]);
  });

  it("never reports the same element twice", () => {
    page(`<table><tr><td><ul><li><p>${PROSE}</p></li></ul></td></tr></table>`);
    const blocks = extractBlocks(document.body);
    expect(blocks).toHaveLength(1);
    expect(blocks[0]!.element.tagName).toBe("TD");
  });
});

describe("contentRoot", () => {
  it("prefers article", () => {
    page(`<main><article id="a"><p>x</p></article></main>`);
    expect(contentRoot(document).id).toBe("a");
  });

  it("falls back to main", () => {
    page(`<main id="m"><p>x</p></main>`);
    expect(contentRoot(document).id).toBe("m");
  });

  it("falls back to [role=main]", () => {
    page(`<div role="main" id="r"><p>x</p></div>`);
    expect(contentRoot(document).id).toBe("r");
  });

  it("falls back to body", () => {
    page(`<p>x</p>`);
    expect(contentRoot(document)).toBe(document.body);
  });

  it("scopes extraction to the root it is given", () => {
    page(`<article id="a"><p>inside</p></article><p>outside</p>`);
    expect(texts(document.getElementById("a")!)).toEqual(["inside"]);
  });
});

describe("collapseWithProvenance", () => {
  it("records one entry per codepoint of the collapsed text", () => {
    page(`<p>alpha <em>beta</em> gamma</p>`);
    const { text, provenance } = collapseWithProvenance(document.querySelector("p")!);
    expect(text).toBe("alpha beta gamma");
    expect(provenance).toHaveLength([...text].length);
  });

  it("attributes each codepoint to the text node it came from", () => {
    page(`<p>ab<em>cd</em></p>`);
    const { text, provenance } = collapseWithProvenance(document.querySelector("p")!);
    expect(text).toBe("abcd");
    expect(provenance[0]!.node.data).toBe("ab");
    expect(provenance[1]!.offset).toBe(1);
    expect(provenance[2]!.node.data).toBe("cd");
    expect(provenance[2]!.offset).toBe(0);
  });
});

describe("word boundaries the markup does not spell out", () => {
  it("separates adjacent block children with no whitespace node between them", () => {
    page(`<ul><li><p>Alpha alpha.</p><p>Beta beta.</p></li></ul>`);
    expect(texts()).toEqual(["Alpha alpha. Beta beta."]);
  });

  it("separates a block child from the text that follows it", () => {
    page(`<li><p>Alpha alpha.</p>tail text</li>`);
    expect(texts()).toEqual(["Alpha alpha. tail text"]);
  });

  it("separates text that precedes a block child", () => {
    page(`<li>lead text<p>Alpha alpha.</p></li>`);
    expect(texts()).toEqual(["lead text Alpha alpha."]);
  });

  it("treats <br> as a word boundary", () => {
    page(`<p>first line<br>second line</p>`);
    expect(texts()).toEqual(["first line second line"]);
  });

  it("does not insert a space at an inline boundary", () => {
    page(`<p>anti<em>dis</em>establishment</p>`);
    expect(texts()).toEqual(["antidisestablishment"]);
  });

  it("does not double the space when whitespace is already there", () => {
    page(`<ul><li><p>Alpha.</p>\n  <p>Beta.</p></li></ul>`);
    expect(texts()).toEqual(["Alpha. Beta."]);
  });

  it("does not open a block with a leading space", () => {
    page(`<ul><li><p>Alpha.</p><p>Beta.</p></li></ul>`);
    expect(texts()[0]!.startsWith("Alpha")).toBe(true);
  });

  it("keeps provenance aligned across an inserted boundary space", () => {
    page(`<li><p>ab</p><p>cd</p></li>`);
    const { text, provenance } = collapseWithProvenance(document.querySelector("li")!);
    expect(text).toBe("ab cd");
    expect(provenance).toHaveLength(5);
    expect(provenance[2]!.node.data).toBe("cd"); // the inserted space anchors forward
    expect(provenance[3]!.node.data).toBe("cd");
  });

  it("skips an excluded subtree entirely, including its nested elements", () => {
    page(`<p>before <code>let <span>x</span> = 1;</code> after</p>`);
    expect(texts()).toEqual(["before after"]);
  });
});

describe("pinRoot", () => {
  it("derives a root when nothing is held yet", () => {
    page(`<main id="m"><p>${PROSE}</p></main>`);
    expect(pinRoot(null, document).id).toBe("m");
  });

  it("keeps the held root even once an article appears inside it", () => {
    page(`<main id="m"><p>${PROSE}</p></main>`);
    const held = pinRoot(null, document);

    document.querySelector("#m")!.insertAdjacentHTML(
      "beforeend",
      `<article id="first"><p>${PROSE}</p></article>`,
    );
    // contentRoot() would now answer #first, which is the whole bug.
    expect(contentRoot(document).id).toBe("first");
    expect(pinRoot(held, document).id).toBe("m");
  });

  it("keeps every later article in scope, which is what the feed case needs", () => {
    page(`<main id="m"></main>`);
    const held = pinRoot(null, document);
    const main = document.querySelector("#m")!;

    for (const id of ["one", "two", "three"]) {
      main.insertAdjacentHTML("beforeend", `<article id="${id}"><p>${PROSE}</p></article>`);
    }
    const found = extractBlocks(pinRoot(held, document));
    expect(found).toHaveLength(3);
    expect(found.map((b) => b.element.closest("article")!.id)).toEqual(["one", "two", "three"]);
  });

  it("re-derives once the held root leaves the document", () => {
    page(`<main id="m"><p>${PROSE}</p></main>`);
    const held = pinRoot(null, document);

    // A wholesale client-side navigation, not an append.
    page(`<main id="replaced"><p>${PROSE}</p></main>`);
    expect(held.isConnected).toBe(false);
    expect(pinRoot(held, document).id).toBe("replaced");
  });

  it("is stable across repeated calls", () => {
    page(`<main id="m"><p>${PROSE}</p></main>`);
    const first = pinRoot(null, document);
    expect(pinRoot(first, document)).toBe(first);
    expect(pinRoot(pinRoot(first, document), document)).toBe(first);
  });
});
