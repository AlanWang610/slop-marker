# fixtures/

Cross-language parity fixtures. Read by `training/tests/` (Python) and `extension/tests/`
(TypeScript) so the two implementations can't drift.

Model-independent, emitted by `training/scripts/make_fixtures.py`:

- `normalize.json` — input text → normalized form + sha256 (§7.4)
- `windows.json` — paragraph text → sentence spans, chunk boundaries and word counts
  (§4.3, §7.3)
- `aggregate.json` — chunk scores + calibration constants → expected run flags, covering
  the length penalty, run pooling, document prior and hysteresis (§8)

Small, hand-checkable, and versioned with the code — not generated at test time. The
calibration constants inside `aggregate.json` are synthetic on purpose, so recalibrating
the model never invalidates the extension's test suite.

Model-dependent, emitted by `training/scripts/make_model_fixtures.py` against a local
bundle. Both record `model_version`, and the TypeScript suite **skips** them when the local
bundle is absent or a different version, since `artifacts/` never enters git:

- `tokenize.json` — text → token ids, asserting the browser tokenizer matches training
  including the `[CLS]`/`[SEP]` post-processor (§5.6). Nothing else asserts this;
  `verify_graph` only ever checked `model_max_length`.
- `logits.json` — text → the logit the shipped int8 ONNX produces under Python
  onnxruntime. The oracle ONNX Runtime Web must reproduce. A browser session that loads,
  runs, and returns quietly wrong numbers is exactly how r1 shipped a model that had lost
  twenty points of AUROC.
- `documents.json` — whole documents through `chunk_block` → int8 ONNX → `aggregate`, the
  path `eval/documents.py::score_document` runs. Pins the *composition* rather than the
  stages, which is what §1's "identical detection behaviour" means in practice. It carries
  public-domain human prose alongside the repo's own markdown, because the markdown all
  flags and a fixture with no negative case would pass against an extension that flagged
  everything.

Corpus-derived, emitted by `training/scripts/make_langgate_fixture.py` from a processed
windows shard:

- `langgate.json` — §7.2's language-gate assertion. Not that the two gates agree block for
  block (the gate is not a coupling point) but that they agree on *acceptance*, and that
  where they disagree the disagreement is not concentrated on non-native English. Its
  English cases are real corpus windows — `hard_negative_kind == non_native_forum` and
  human windows with none — because the question is whether a detector trips on real
  second-language prose, and constructed examples answer an easier one. Non-English is
  public-domain prose, since the corpus is English-gated and contains none. A privacy
  filter drops any window carrying a link, handle, email or phone number.

Regenerate the model-dependent pair after a model refresh:

```bash
uv run --extra modal python tools/pull_bundle.py --version <new-version>
cd training && uv run --extra export python scripts/make_model_fixtures.py \
    --bundle ../artifacts/bundles/<new-version>
```

Source text for those two is prose pulled from the repo's own markdown, so they are
reproducible from a checkout with no corpus access. It is scored, never labelled — these
are parity oracles, not evaluation data.
