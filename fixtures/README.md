# fixtures/

Cross-language parity fixtures. Read by `training/tests/` (Python) and
`extension/tests/` (TypeScript) so the two implementations can't drift.

Planned:

- `normalize.json` — input text → normalized form + sha256 (§7.4)
- `windows.json` — paragraph text → chunk boundaries and word counts (§4.3, §7.3)
- `aggregate.json` — chunk scores + calibration constants → expected run flags, covering
  the length penalty, run pooling, document prior and hysteresis (§8)
- `tokenize.json` — text → token ids, asserting the browser tokenizer matches training,
  including the `[CLS]`/`[SEP]` post-processor (§5.6)

Small, hand-checkable, and versioned with the code — not generated at test time.
