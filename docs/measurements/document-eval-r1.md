# Document-level evaluation -- run r1, bundle `mb-base-0.3.0-dev`

Everything in `calibration-*.md` is chunk-level, and a chunk is not what anyone sees. A
reader sees a highlighted **run**, produced only after the length penalty, run pooling,
the document prior and the 150-word minimum. These numbers come from the whole scope.md
8 path -- `chunk_block` -> int8 ONNX -> `aggregate` -- on the artifact that ships.

Three sources, at increasing distance from the training distribution.

| source | what it is | documents | doc FPR | CP 95% upper |
|---|---|---|---|---|
| `test_human` | held-out human documents from our own corpus | 1,500 | 0.33% | 0.70% |
| `unseen_host` | held-out human documents from domains absent from train | 1,000 | 0.20% | 0.63% |
| `raid` | external benchmark, 8 domains, human side never harvested | 1,196 human | 0.17% | 0.53% |

The plan's gate is **doc-level FPR <= 3%**. All three pass with roughly a factor of four
in hand on the upper bound, and `release_gate` now refuses a bundle that cannot show
this measurement (see `export-r1.md`).

## Where the false positives are

`test_human`, 5 flagged of 1,500:

| genre | n | flagged | FPR |
|---|---|---|---|
| academic_formal | 411 | 3 | 0.0073 |
| blog_personal | 495 | 2 | 0.0040 |
| product_marketing | 255 | 0 | 0 |
| news | 119 | 0 | 0 |
| forum_comment | 115 | 0 | 0 |
| press_release | 38 | 0 | 0 |
| technical_docs | 34 | 0 | 0 |
| encyclopedia | 33 | 0 | 0 |

`unseen_host`, 2 flagged of 1,000, both `blog_personal` (2 of 523).

**The mined hard negatives stayed clean in both runs.** These categories are the reason
the corpus was built the way it was -- scope.md 2 names templated marketing copy, press
releases and non-native English as the dominant false-positive risks:

| hard negative | test_human | unseen_host |
|---|---|---|
| product_template | 0 / 174 | 0 / 244 |
| non_native_forum | 0 / 39 | 0 / 47 |
| press_release | 0 / 36 | 0 / 28 |
| academic_formal | 3 / 400 | -- |

So the residual false-positive risk is not where the design expected it. It is citation-
dense academic prose and ordinary personal blogs, not marketing or press-release
boilerplate.

## RAID

The first external test this model has faced, and the one that says how much of the
0.9939 on our own test split is corpus-specific.

| | value |
|---|---|
| documents | 2,385 (1,196 human, 1,189 AI) |
| AUROC, ranked by strongest chunk | **0.8819** |
| AUROC, ranked by strongest run score | 0.6345 |
| documents producing no run at all | 86.2% |
| median words / chunks per document | 259 / 1 |

False positives transfer. The operating point does not:

| threshold | FPR | recall |
|---|---|---|
| our shipped `t_on` = 0.8397 | 0.17% | 26.1% |
| RAID's own 1% FPR (t = 0.4393) | 1% | 38.9% |
| RAID's own 5% FPR (t = 0.1280) | 5% | **53.0%** |

`t_on` was placed where our AI windows sit near p = 0.96. RAID's 2023-era generators
score lower, so the same threshold buys a false-positive rate of 0.17% at a recall
nobody wants. That is the intended failure direction for this product -- out of
distribution it misses AI rather than flagging humans -- but it is a real limit, and
the 53% at RAID's own operating point is the number to quote for the model's actual
discriminative power on unseen generators.

Recall by domain localizes the gap:

| domain | human | FPR | recall |
|---|---|---|---|
| wiki | 150 | 0.0067 | 0.5068 |
| reviews | 146 | 0.0000 | 0.4514 |
| news | 150 | 0.0000 | 0.4467 |
| books | 150 | 0.0000 | 0.3533 |
| abstracts | 150 | 0.0000 | 0.1600 |
| reddit | 150 | 0.0067 | 0.1409 |
| poetry | 150 | 0.0000 | 0.0203 |
| recipes | 150 | 0.0000 | 0.0133 |

The top four are continuous prose, which is what this corpus is made of. The bottom two
are structurally different text -- recipes are ingredient lists and imperative steps,
poetry is verse -- and the model is at chance on both. Neither appears anywhere in
training. That is a legible boundary rather than a mystery, and it is also a warning:
the corpus covers eight web genres and generalizes within that shape, not outside it.

RAID's attacks were not informative here. Only `none` and `whitespace` survived the
per-cell sampling in useful numbers, and whitespace insertion did not reduce recall:

| attack | n | human FPR | recall |
|---|---|---|---|
| none | 1,190 | 0.0033 | 0.2416 |
| whitespace | 1,195 | 0.0000 | 0.2797 |

Whitespace scoring *higher* than unattacked is about two standard errors at this size.
It persisted from a smaller run rather than washing out, and the plausible mechanism is
that whitespace insertion fragments tokenization into sequences that read as un-humanlike.
Treat it as unexplained, not as evidence of robustness.

## Three defects found in the evaluation itself

Recorded because each one produced a number that looked publishable.

1. **A stream prefix is not a sample.** RAID's train split is ordered by domain, so
   reading the first 60k rows returned 1,493 documents all from `abstracts`, with three
   attacks of twelve, and reported an AUROC for it. Now stratified by (domain, class,
   attacked) cell over the whole file, printing which domains it reached.
2. **AUROC over ties describes the ties.** Documents were ranked by `max_run_score`,
   which defaults to 0.0 when no run clears `t_on`. 86% of RAID documents produce no
   run, so both classes collapsed onto one value and AUROC sat near chance by
   construction -- the 0.563 first reported. Ranking now uses the strongest penalized
   chunk, which is always defined.
3. **Genre came from the wrong place.** `build()` relabels documents still marked
   `other` from their text and writes the result into the windows without updating
   `interim/`, so reading `doc.genre` reported the harvester's guess. It put 42% of the
   sample in a bucket the corpus manifest does not contain and made `blog_personal`
   vanish. The headline FPR never depended on genre; the per-genre attribution did, and
   it was wrong -- an earlier reading blamed `press_release` for a false positive that
   was actually a personal blog.

A fourth is worth noting as a reproducibility fix rather than a defect in a published
number: the document pool was built by iterating a `set` of document ids, and Python
randomizes string hashing per process, so the seeded shuffle drew a different sample on
every run. `test_human` measured 3/1,500 before the fix and 5/1,500 after -- two draws
from the same population, overlapping intervals, no change to any conclusion, but only
the second is reproducible.

## What these numbers do not cover

- **Mixed authorship.** 0.75% of the test split has an AI fraction strictly between 0
  and 0.7, so span-level attribution is untested. See `training-r1.md`.
- **Adversarial robustness.** `attack` is `None` for all 55,715 rows of our test split;
  the plan's paraphrase, humanizer, shuffling and typo program never ran. RAID's two
  surviving attacks are the only adversarial evidence in this document.
- **The live web.** `unseen_host` is unseen *hosts*, not unseen *era* -- pre-2022
  FineWeb from domains absent from training. A genuinely current-web human set cannot be
  built with trustworthy labels, which is why the whole corpus is pre-2022.
