# data/

Corpora. Everything here is gitignored.

Suggested layout (created by the corpus scripts, not checked in):

```
raw/         pre-2022 crawl slices, Wikipedia, Reddit, news archives, as fetched
generated/   AI side: per-model × sampling × prompt-style outputs, plus attack variants
interim/     cleaned, deduplicated, genre-tagged
processed/   train / val / test splits with AI-fraction and genre labels
```

Rules:

- Splits are by **document and by source**, never by window — windows from one document
  must not straddle train and val (§4.3).
- Held-out calibration set is separate from the test set (§4.5).
- Provenance (source, generator model, prompt style, sampling params, attack applied) is
  recorded per row; it's what makes per-genre FPR (§4.4) and refresh (§4.7) possible.
- Anything expensive to regenerate — the AI side especially — gets backed up outside git.
