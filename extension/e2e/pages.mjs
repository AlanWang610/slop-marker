/**
 * The pages both browser harnesses load, in one place so Chrome and Firefox cannot drift.
 *
 * Every page is built from `fixtures/documents.json`, so the expected outcome is never a
 * guess: Python already scored this exact text through the same pipeline, and the fixture
 * records which documents come out flagged.
 *
 * The structured page is the important one. scope.md 7.1 names five candidate elements and
 * a list of exclusions, and until now the harness only ever rendered `<p>` inside
 * `<article>` -- so `li`, `blockquote`, `dd` and `td` were never once exercised in a
 * browser, and neither was any exclusion beyond `nav` and `footer`. The decoys carry the
 * *same flagged text* as the scoreable blocks, which is the strong form of the assertion:
 * it proves the text was excluded, not merely that the model scored that copy low.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..", "..");

const escape = (s) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/**
 * Paragraphs of at least ~110 words. Blocks under `min_words` (40) are correctly skipped by
 * the content script, so a page of short paragraphs is scored not at all -- which on screen
 * looks identical to a page the model declined to flag.
 */
function paras(text) {
  const out = [];
  let current = [];
  let words = 0;
  for (const sentence of text.split(/(?<=\.)\s+/)) {
    current.push(sentence);
    words += sentence.split(/\s+/).length;
    if (words >= 110) {
      out.push(current.join(" "));
      current = [];
      words = 0;
    }
  }
  if (words >= 45) out.push(current.join(" "));
  return out.map((p) => `<p>${escape(p.trim())}</p>`).join("\n");
}

const doc = (title, body, head = "") =>
  `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>${title}</title>${head}</head>
<body>
${body}
</body></html>`;

/** The original page: one AI-authored article, one human-authored, nav and footer decoys. */
function articlePage(flagged, clean) {
  return doc(
    "e2e",
    `  <nav><p>Navigation text that must never be scored, however long it is made.</p></nav>
  <main>
    <article id="flagged">${paras(flagged.text)}</article>
    <hr>
    <article id="clean">${paras(clean.text)}</article>
  </main>
  <footer><p>Footer text that must never be scored, however long it is made.</p></footer>`,
  );
}

/**
 * scope.md 7.1 in full: every candidate element, and every exclusion, carrying identical
 * flagged text.
 *
 * Note the markup is written without whitespace between adjacent block tags in `#nested`.
 * That is deliberate: pretty-printed HTML has a whitespace text node at every boundary and
 * hides the case where two paragraphs would otherwise be concatenated into one word.
 */
function structuredPage(flagged) {
  const t = escape(flagged.text);
  return doc(
    "e2e structured",
    `  <main>
    <h1>Candidates</h1>

    <ul id="in-li"><li>${t}</li></ul>

    <blockquote id="in-blockquote">${t}</blockquote>

    <dl id="in-dd"><dt>A term</dt><dd>${t}</dd></dl>

    <table id="in-td">
      <thead><tr><th>Column</th></tr></thead>
      <tbody><tr><td>${t}</td></tr></tbody>
    </table>

    <h1>Nesting</h1>
    <ul id="nested"><li><p>${t}</p></li></ul>

    <h1>Exclusions, all carrying the same text</h1>

    <pre id="x-pre"><code>${t}</code></pre>
    <figure id="x-figure"><figcaption>${t}</figcaption></figure>
    <div id="disqus_thread"><p>${t}</p></div>
    <section class="comments" id="x-comments"><p>${t}</p></section>
    <div id="x-editable" contenteditable="true"><p>${t}</p></div>
    <aside id="x-aside"><p>${t}</p></aside>
    <form id="x-form"><p>${t}</p></form>
    <div id="x-role" role="complementary"><p>${t}</p></div>
  </main>`,
  );
}

/**
 * A dark ground and a tinted callout. scope.md 9 mixes against the *effective* background,
 * so what has to hold is that the two blocks resolve to different, non-white colours.
 *
 * Both live inside one <article> because that is what `contentRoot()` selects first: a
 * callout sitting outside it would never be extracted, and the test would be asserting
 * against a block the extension was right to ignore.
 */
function darkPage(flagged) {
  return doc(
    "e2e dark",
    `  <main>
    <article>
      <div id="dark-body">${paras(flagged.text)}</div>
      <div id="callout"><p>${escape(flagged.text)}</p></div>
    </article>
  </main>`,
    `<style>
  html, body { background: rgb(17, 17, 17); color: rgb(238, 238, 238); }
  #callout { background: rgb(40, 30, 60); padding: 1em; }
</style>`,
  );
}

/** An empty shell the harness fills at runtime, to drive the SPA MutationObserver path. */
function spaPage() {
  return doc("e2e spa", `  <main id="root"><p>Placeholder, replaced by the harness.</p></main>`);
}

/**
 * The saved real-page corpus, served at /real/<name>. All four are works of the United
 * States Government and all four are human-written, so the expected outcome is no
 * highlights at all -- the false-positive direction, which is the one that matters for a
 * high-precision detector and the one generated markup cannot test.
 */
function realPages() {
  const dir = join(here, "real");
  const manifestPath = join(dir, "manifest.json");
  if (!existsSync(manifestPath)) return { routes: {}, pages: [] };

  const manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));
  const routes = {};
  const pages = [];
  for (const page of manifest.pages) {
    const file = join(dir, `${page.name}.html`);
    if (!existsSync(file)) continue;
    routes[`/real/${page.name}`] = readFileSync(file, "utf-8");
    pages.push(page);
  }
  return { routes, pages };
}

export function buildPages() {
  const fx = JSON.parse(readFileSync(join(repo, "fixtures", "documents.json"), "utf-8"));
  const flagged = fx.documents.find((d) => d.expected.runs.some((r) => r.flagged));
  const clean = fx.documents.find((d) => !d.expected.runs.some((r) => r.flagged));
  if (!flagged || !clean) throw new Error("documents.json needs one flagged and one clean case");

  const real = realPages();

  return {
    flaggedName: flagged.name,
    cleanName: clean.name,
    /** Provenance for the saved corpus: name, url, licence note, sha256. */
    realPages: real.pages,
    /** The raw flagged text, for pages the harness assembles in the browser. */
    flaggedText: flagged.text,
    /** Paragraph markup for that text, ready to inject. */
    flaggedParagraphs: paras(flagged.text),
    routes: {
      "/": articlePage(flagged, clean),
      "/structured": structuredPage(flagged),
      "/dark": darkPage(flagged),
      "/spa": spaPage(),
      ...real.routes,
    },
  };
}
