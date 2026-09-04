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
    calibration = json.loads((bundle / "calibration.json").read_text(encoding="utf-8"))
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

    encodings = [tok.encode(collapse_whitespace(t)) for t in chosen]

    tokenize_payload = {
        "version": 1,
        "model_version": version,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
