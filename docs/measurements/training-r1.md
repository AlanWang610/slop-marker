# Training run r1 -- ModernBERT-base on corpus v3

| setting | value |
|---|---|
| backbone | `answerdotai/ModernBERT-base` |
| corpus | v3, 746,798 windows from 247,460 documents |
| epochs | 2 |
| effective batch | 64 |
| learning rate | 3e-5, layerwise decay 0.9 over 22 layers |
| label smoothing | 0.05, symmetric target clamp |
| auxiliary genre head | weight 0.2, dropped at export |
| sampling | AI rate balanced within each genre, genre marginals preserved |
| GPU | one H100 |
| wall clock | **56.5 minutes**, 18,622 steps |

## Results

Scored with the PyTorch checkpoint over the **complete** split, not a subsample.
FPR counts only `ai_fraction == 0.0` windows; recall counts only `>= 0.7`; the
mid-band is excluded from both (264 windows in val, 418 in test).

| split | windows | human | ai | AUROC | pAUC@2%FPR | FPR@recall90 |
|---|---|---|---|---|---|---|
| val | 36,825 | 29,720 | 6,841 | 0.9924 | 0.9635 | 0.0021 |
| test | 55,715 | 44,770 | 10,527 | **0.9939** | **0.9677** | **0.0015** |

Per genre on test:

| genre | windows | human | ai | AUROC |
|---|---|---|---|---|
| blog_personal | 18,368 | 13,858 | 4,351 | 0.9923 |
| academic_formal | 17,795 | 14,890 | 2,803 | 0.9969 |
| product_marketing | 6,853 | 6,040 | 774 | 0.9934 |
| news | 4,395 | 3,195 | 1,149 | 0.9888 |
| forum_comment | 4,037 | 3,236 | 767 | 0.9969 |
| press_release | 1,678 | 1,384 | 284 | 0.9919 |
| encyclopedia | 1,495 | 1,193 | 285 | 0.9943 |
| technical_docs | 1,094 | 974 | 114 | 0.9937 |

Test scores slightly *above* val, which is the expected direction only because
nothing distinguishes the two: `held_out_domains_per_genre` is configured at 10 and
enforced nowhere, so no domains are reserved test-only and the test split does not
currently measure host-template memorization. Treat test here as a second val.

## Correction to the previously recorded figures

An earlier version of this file reported AUROC 0.9928 / pAUC 0.9632 / FPR@90 0.0014
under the heading "Validation results". Those came from `evaluate()`, which defaulted
to 40 batches against an unshuffled loader -- the same ~1,280 windows in `doc_id`
order on every call, 3.5% of val, selected by filename. The checkpoint was selected on
that slice too.

The slice turned out to be close to unbiased: its genre mix tracks the split's, and
its AUROC of 0.9946 sits near the full split's 0.9924. So the defect cost accuracy of
reporting rather than accuracy of the model, and the headline numbers above are within
0.001 of what was originally claimed. It was still a measurement of the wrong object,
and it is fixed: `cfg.eval_windows` now draws a fixed random subsample.

## Excess BCE

0.072 against a floor of zero. This is the number to read rather than raw loss --
soft-target BCE has a floor equal to the mean target entropy, so with mixed-authorship
rows the raw loss plateaus well above zero and never approaches it, which looks
identical to a broken run.

## Genre AI rates entering training

Measured before the sampler corrects them:

| genre | AI rate |
|---|---|
| blog_personal | 0.248 |
| news | 0.220 |
| encyclopedia | 0.198 |
| press_release | 0.200 |
| forum_comment | 0.172 |
| academic_formal | 0.155 |
| technical_docs | 0.127 |
| product_marketing | 0.105 |

All well below half. The balanced sampler equalises the rate inside each genre to 0.50
while leaving genre marginals untouched, so the model does not see these rates. What
the sampler cannot fix is a genre with too few distinct minority-class windows to
reweight, which is what `check_split_integrity` now asserts.

## How to read these numbers

They are optimistic, for two reasons that are documented rather than argued away.

`corpus-gate.md` records three surface probes above their a-priori bands, part of which
is a residual specificity confound: human documents carry proper nouns and parenthetical
asides that AI text on the same topic does not match. A test FPR of 0.15% at 90% recall
will not survive contact with the open web unchanged.

The corpus is also not the one the config describes. `genre_targets` asks for 75k-120k
windows per genre and is read by nothing, so academic prose reached 237k against a
target of 90k while technical_docs reached 14k against 75k; the genre-share cap was
computed against the pre-cap total and so did not bind at the configured 0.25. Both are
fixed in code but not in this corpus -- these numbers come from v3 as built.

**These figures describe the PyTorch checkpoint, which is not what ships.** The int8
artifact scores far worse; see `export-r1.md`.
