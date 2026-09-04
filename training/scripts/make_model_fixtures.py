"""Emit the two model-dependent fixtures: fixtures/tokenize.json and fixtures/logits.json.

Unlike the three fixtures from make_fixtures.py, these are tied to a specific model bundle
and are regenerated whenever the bundle version changes. Both record that version, and the
TypeScript suite skips them when the local bundle does not match rather than failing.

  uv run --extra export python scripts/make_model_fixtures.py \
      --bundle ../artifacts/bundles/mb-base-0.2.0-dev

tokenize.json closes the gap scope.md 5.6 names and nothing asserted: that browser
tokenization matches training, including the [CLS]/[SEP] post-processor. verify_graph only
ever checked model_max_length.

logits.json is the oracle the browser must reproduce. Without it the extension could ship a
session that loads, runs, and returns quietly wrong numbers -- which is precisely how the
r1 export shipped a model that had lost twenty points of AUROC.

Source text is prose pulled from the repo's own markdown, so the fixture is reproducible
from a checkout and needs no corpus access. It is scored, never labelled: these are parity
oracles, not evaluation data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "training" / "src"))

from slopmarker.data.chunking import chunk_block  # noqa: E402
from slopmarker.data.normalize import collapse_whitespace, count_words  # noqa: E402
from slopmarker.eval.aggregate import Chunk, aggregate  # noqa: E402
from slopmarker.eval.calibration import Calibration  # noqa: E402

MAX_LENGTH = 512

# Typography that must survive to the model (scope.md 7.4: collapse_whitespace is the model
# input, so curly quotes and em dashes are still there when it tokenizes).
EDGE_CASES = [
    "The report\u2019s findings \u2014 released Tuesday \u2014 said the \u201Cunprecedented\u201D "
    "growth would continue. Analysts weren\u2019t convinced\u2026 and said so.",
    "Se\u00F1or Garc\u00EDa\u2019s caf\u00E9 r\u00E9sum\u00E9 na\u00EFve fa\u00E7ade \u2014 "
    "accented text should tokenize identically on both sides of this fence.",
    "Prices fell 3.14% in Q2. Dr. Smith of St. Louis said e.g. the U.S. market vs. the U.K. "
    "market showed no Ph.D.-level divergence. He was right.",
]


# Public-domain human prose, added because the repo's own markdown is partly LLM-written
# and every document drawn from it flags. A fixture with no negative case would pass just as
# happily against an extension that flagged everything.
PUBLIC_DOMAIN = {
    "austen-pride-and-prejudice.txt": (
        "It is a truth universally acknowledged, that a single man in possession of a good "
        "fortune, must be in want of a wife. However little known the feelings or views of "
        "such a man may be on his first entering a neighbourhood, this truth is so well "
        "fixed in the minds of the surrounding families, that he is considered as the "
        "rightful property of some one or other of their daughters. My dear Mr. Bennet, "
        "said his lady to him one day, have you heard that Netherfield Park is let at last? "
        "Mr. Bennet replied that he had not. But it is, returned she; for Mrs. Long has just "
        "been here, and she told me all about it. Mr. Bennet made no answer. Do not you want "
        "to know who has taken it? cried his wife impatiently. You want to tell me, and I "
        "have no objection to hearing it. This was invitation enough. Why, my dear, you must "
        "know, Mrs. Long says that Netherfield is taken by a young man of large fortune from "
        "the north of England; that he came down on Monday in a chaise and four to see the "
        "place, and was so much delighted with it that he agreed with Mr. Morris "
        "immediately; that he is to take possession before Michaelmas, and some of his "
        "servants are to be in the house by the end of next week. What is his name? Bingley. "
        "Is he married or single? Oh! single, my dear, to be sure! A single man of large "
        "fortune; four or five thousand a year. What a fine thing for our girls!"
    ),
    "melville-moby-dick.txt": (
        "Call me Ishmael. Some years ago, never mind how long precisely, having little or no "
        "money in my purse, and nothing particular to interest me on shore, I thought I "
        "would sail about a little and see the watery part of the world. It is a way I have "
        "of driving off the spleen and regulating the circulation. Whenever I find myself "
        "growing grim about the mouth; whenever it is a damp, drizzly November in my soul; "
        "whenever I find myself involuntarily pausing before coffin warehouses, and bringing "
        "up the rear of every funeral I meet; and especially whenever my hypos get such an "
        "upper hand of me, that it requires a strong moral principle to prevent me from "
        "deliberately stepping into the street, and methodically knocking people's hats off "
        "then, I account it high time to get to sea as soon as I can. This is my substitute "
        "for pistol and ball. With a philosophical flourish Cato throws himself upon his "
        "sword; I quietly take to the ship. There is nothing surprising in this. If they but "
        "knew it, almost all men in their degree, some time or other, cherish very nearly "
        "the same feelings towards the ocean with me."
    ),
}


def prose_blocks(paths: list[Path], min_words: int) -> list[str]:
    """Continuous prose paragraphs from markdown: no headings, tables, code or lists."""
    out: list[str] = []
    for path in sorted(paths):
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            stripped = block.strip()
            if not stripped:
                continue
            first = stripped.splitlines()[0].lstrip()
            if first.startswith(("#", "|", "-", "*", ">", "`", "    ")):
                continue
            if "```" in stripped or "|" in stripped:
                continue
            text = collapse_whitespace(stripped)
            if count_words(text) >= min_words:
                out.append(text)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "fixtures")
    ap.add_argument("--max-texts", type=int, default=60)
    args = ap.parse_args()

    bundle: Path = args.bundle.resolve()
    calibration_text = (bundle / "calibration.json").read_text(encoding="utf-8")
    calibration = json.loads(calibration_text)
    version = calibration["version"]

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(bundle / "tokenizer.json"))

    # scope.md 5.6: confirm the post-processor is present. Without it the browser would
    # feed the model bare token ids and no [CLS], and the pooled logit would be garbage.
    spec = json.loads((bundle / "tokenizer.json").read_text(encoding="utf-8"))
    post = spec.get("post_processor")
    if post is None:
        raise SystemExit("tokenizer.json has no post_processor; [CLS]/[SEP] would be missing")
    probe = tok.encode("hello world")
    cls_id, sep_id = probe.ids[0], probe.ids[-1]
    decoded_cls = tok.id_to_token(cls_id)
    decoded_sep = tok.id_to_token(sep_id)
    print(f"post_processor {post.get('type')}: {decoded_cls!r} .. {decoded_sep!r}")

    tok.enable_truncation(max_length=MAX_LENGTH)

    docs = sorted((REPO / "docs").rglob("*.md")) + [REPO / "README.md"]
    texts: list[str] = []
    for block in prose_blocks(docs, min_words=40):
        texts.extend(chunk_block(block))
    # Deterministic, deduplicated, and spread across the corpus rather than front-loaded.
    seen: set[str] = set()
    unique = [t for t in texts if not (t in seen or seen.add(t))]
    step = max(1, len(unique) // args.max_texts)
    chosen = unique[::step][: args.max_texts] + EDGE_CASES

    # Cases that exceed max_length, because truncation is where the two tokenizers disagree
    # and it is NOT an edge case: chunk_block caps at 400 words and scope.md 4.3 puts a
    # 400-word window at ~500-540 tokens. Without these the fixture passed against a browser
    # that silently dropped the closing [SEP] from every long chunk.
    long_text = " ".join(unique[:12])
    for target_words in (400, 600):
        words = collapse_whitespace(long_text).split(" ")[:target_words]
        if len(words) == target_words:
            chosen.append(" ".join(words))

    encodings = [tok.encode(collapse_whitespace(t)) for t in chosen]

    n_truncated = sum(1 for e in encodings if len(e.ids) >= MAX_LENGTH)
    if n_truncated == 0:
        raise SystemExit("no case reaches max_length; the truncation path would go unpinned")

    tokenize_payload = {
        "version": 1,
        "model_version": version,
        "n_truncated": n_truncated,
        "note": (
            "text -> token ids, asserting the browser tokenizer matches training including "
            "the [CLS]/[SEP] post-processor (scope.md 5.6). Input is collapse_whitespace(text), "
            f"truncation at max_length={MAX_LENGTH}."
        ),
        "max_length": MAX_LENGTH,
        "cls_token_id": cls_id,
        "sep_token_id": sep_id,
        "cases": [
            {"text": t, "input_ids": e.ids, "attention_mask": e.attention_mask}
            for t, e in zip(chosen, encodings, strict=True)
        ],
    }
    (args.out / "tokenize.json").write_text(
        json.dumps(tokenize_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"tokenize.json: {len(chosen)} cases")

    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(bundle / "model.onnx"), providers=["CPUExecutionProvider"])
    logits: list[float] = []
    for e in encodings:
        out = sess.run(
            None,
            {
                "input_ids": np.asarray([e.ids], dtype=np.int64),
                "attention_mask": np.asarray([e.attention_mask], dtype=np.int64),
            },
        )
        logits.append(float(np.asarray(out[0]).reshape(-1)[0]))

    logits_payload = {
        "version": 1,
        "model_version": version,
        "note": (
            "The oracle the browser must reproduce. Logits from the shipped int8 ONNX under "
            "Python onnxruntime, batch 1, no padding. A WASM session that loads and runs but "
            "returns different numbers is the failure this pins down."
        ),
        "max_length": MAX_LENGTH,
        "tolerance": 1e-3,
        "cases": [
            {"text": t, "n_tokens": len(e.ids), "logit": lg}
            for t, e, lg in zip(chosen, encodings, logits, strict=True)
        ],
    }
    (args.out / "logits.json").write_text(
        json.dumps(logits_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lo, hi = min(logits), max(logits)
    ntok = [len(e.ids) for e in encodings]
    print(f"logits.json:   {len(logits)} cases, logit range [{lo:.3f}, {hi:.3f}]")
    print(f"               tokens min={min(ntok)} median={sorted(ntok)[len(ntok) // 2]} max={max(ntok)}")

    # ---- documents.json: the whole scope.md 8 path, end to end ----------------------
    #
    # The other two fixtures pin one stage each. This one pins the composition, which is
    # what scope.md 1 actually promises ("identical detection behaviour on both browsers":
    # same model, same constants, same scoring logic). It is the only fixture that would
    # catch the extension chunking correctly, scoring correctly, and then aggregating over
    # the wrong sequence.
    cal = Calibration.from_json(calibration_text)

    def score_chunks(texts_in: list[str]) -> list[float]:
        out: list[float] = []
        for chunk_text in texts_in:
            enc = tok.encode(collapse_whitespace(chunk_text))
            logit_out = sess.run(
                None,
                {
                    "input_ids": np.asarray([enc.ids], dtype=np.int64),
                    "attention_mask": np.asarray([enc.attention_mask], dtype=np.int64),
                },
            )
            out.append(float(np.asarray(logit_out[0]).reshape(-1)[0]))
        return out

    documents: list[dict[str, object]] = []
    # Whole markdown files, so a document spans many paragraphs and produces several chunks
    # -- which is what exercises run pooling, the document prior and the 150-word minimum.
    doc_sources = [
        *sorted((REPO / "docs").rglob("*.md")),
        REPO / "README.md",
        REPO / "training" / "README.md",
        REPO / "fixtures" / "README.md",
    ]
    named_texts: list[tuple[str, str]] = [
        (path.name, " ".join(prose_blocks([path], min_words=20))) for path in doc_sources
    ]
    named_texts.extend(PUBLIC_DOMAIN.items())

    for name, raw_text in named_texts:
        doc_text = collapse_whitespace(raw_text)
        chunk_texts = chunk_block(doc_text)
        # A single-chunk document is kept deliberately: it is the case where the document
        # prior fires (one chunk cannot be 20% of itself and still be below it) and where
        # the 150-word run minimum decides the outcome on its own.
        if not chunk_texts:
            continue
        chunk_logits = score_chunks(chunk_texts)
        scored = [
            Chunk(logit=lg, words=count_words(c))
            for lg, c in zip(chunk_logits, chunk_texts, strict=True)
        ]
        result = aggregate(scored, cal)
        documents.append(
            {
                "name": name,
                "text": doc_text,
                "chunks": [{"text": c, "words": count_words(c), "logit": lg}
                           for c, lg in zip(chunk_texts, chunk_logits, strict=True)],
                "expected": {
                    "t_on_effective": result.t_on_effective,
                    "chunk_p": result.chunk_p,
                    "chunk_p_penalized": result.chunk_p_penalized,
                    "runs": [
                        {"start": r.start, "end": r.end, "words": r.words,
                         "score": r.score, "flagged": r.flagged}
                        for r in result.runs
                    ],
                },
            }
        )

    (args.out / "documents.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_version": version,
                "note": (
                    "Whole documents through chunk_block -> int8 ONNX -> aggregate, the same "
                    "path eval/documents.py::score_document runs. Pins the composition, not "
                    "the stages: this is what scope.md 1's 'identical detection behaviour' "
                    "means in practice."
                ),
                "tolerance": 1e-3,
                "documents": documents,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    n_flagged = sum(1 for d in documents
                    if any(r["flagged"] for r in d["expected"]["runs"]))  # type: ignore[index]
    total_chunks = sum(len(d["chunks"]) for d in documents)  # type: ignore[arg-type]
    print(f"documents.json: {len(documents)} documents, {total_chunks} chunks, "
          f"{n_flagged} with a flagged run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
