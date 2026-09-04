# Calibration -- bundle `mb-base-0.2.0-dev`

Thresholds selected on the **quantized artifact that ships**, not on the PyTorch
checkpoint (scope.md 5.5). Artifact: weight-only int8 encoder with an int4 embedding
table, 136.7 MB; bundle 140.3 MB across six files.

Selection ran on the calibration split, verification on test. Both were sampled
genre-stratified: 21,485 calibration and 21,767 test windows, so the smallest genre
still carries ~900 human windows rather than the 57 that set the threshold in r1.

## Operating point

| quantity | value |
|---|---|
| temperature | 0.7310 |
| `t_on` | 0.8397 |
| `t_off` | 0.7420 |
| binding genre | `press_release` |

## The oracle gap

The question scope.md 4.4 left open: what does shipping a single global threshold cost,
against an oracle that knows each window's genre and uses that genre's own threshold?

| | recall |
|---|---|
| single global threshold `t_on` | 0.9369 |
| per-genre oracle thresholds | 0.9552 |
| **gap** | **0.0183** |

**1.8 recall points.** The plan set ~20 points as the level that would force
reconsidering the design. At 1.8, dropping the genre head at export is vindicated:
the extension does not need to know what it is reading, and the model does not need to
ship a genre classifier to hit its false-positive bound.

The reason the gap is small is visible in the per-genre thresholds below. They span
0.179 to 0.840, which looks like a lot -- but the classes are separated well enough in
every genre (AUC 0.987 to 0.998) that moving the threshold across that range costs
little recall. A single threshold is cheap precisely because the model is good.

## Per-genre, calibration split

Threshold is the per-genre 2%-FPR point; `fpr_upper` is the Clopper-Pearson 95% upper
bound. `t_on` is the maximum over this column, which is why `press_release` binds.

| genre | human | ai | threshold | FPR | FPR upper | recall | AUC |
|---|---|---|---|---|---|---|---|
| academic_formal | 2,953 | 529 | 0.3873 | 0.0054 | 0.0082 | 0.9584 | 0.9956 |
| blog_personal | 2,571 | 903 | 0.4651 | 0.0066 | 0.0099 | 0.9502 | 0.9974 |
| encyclopedia | 1,079 | 281 | 0.4715 | 0.0093 | 0.0157 | 0.8861 | 0.9889 |
| forum_comment | 2,907 | 574 | 0.1790 | 0.0038 | 0.0063 | 0.9477 | 0.9961 |
| news | 2,970 | 511 | 0.7056 | 0.0077 | 0.0110 | 0.9276 | 0.9902 |
| **press_release** | 1,254 | 262 | **0.8397** | 0.0144 | 0.0212 | 0.9580 | 0.9952 |
| product_marketing | 3,143 | 337 | 0.4240 | 0.0057 | 0.0085 | 0.8813 | 0.9867 |
| technical_docs | 927 | 151 | 0.4677 | 0.0054 | 0.0113 | 0.9536 | 0.9980 |

## Verification on test, at the shipped `t_on`

Selection used a margin (thresholds chosen at a fraction of target on calibration) so
that the bound is met on held-out data rather than half the time. It is:

| quantity | value | gate |
|---|---|---|
| overall FPR | 0.00633 | <= 0.01 ✓ |
| worst per-genre FPR upper bound | 0.0184 | <= 0.02 ✓ |
| recall | 0.9385 | -- |
| AUROC | 0.9939 | -- |

| genre | human | FPR | FPR upper | recall |
|---|---|---|---|---|
| academic_formal | 2,917 | 0.0051 | 0.0079 | 0.9666 |
| blog_personal | 2,622 | 0.0072 | 0.0106 | 0.9377 |
| encyclopedia | 1,193 | 0.0025 | 0.0065 | 0.9544 |
| forum_comment | 2,825 | 0.0050 | 0.0077 | 0.9551 |
| news | 2,553 | 0.0090 | 0.0127 | 0.9074 |
| press_release | 1,384 | 0.0123 | 0.0184 | 0.9472 |
| product_marketing | 3,069 | 0.0062 | 0.0091 | 0.9459 |
| technical_docs | 974 | 0.0010 | 0.0049 | 0.8684 |

Test AUROC of 0.9939 on the quantized artifact equals the PyTorch checkpoint's 0.9939
on the same split. Quantization cost nothing measurable end to end.

## Quantization delta, on identical rows

Measured inside the export, on 4,000 calibration windows, and enforced as a gate:

| quantity | fp32 | int8 | drop | gate |
|---|---|---|---|---|
| AUROC | 0.9905 | 0.9896 | 0.0009 | <= 0.01 ✓ |
| pAUC@2%FPR | 0.9554 | 0.9538 | 0.0016 | <= 0.01 ✓ |
| Spearman | -- | 0.9819 | -- | >= 0.95 ✓ |
| decision flip rate | -- | 0.00725 | -- | <= 0.01 ✓ |
| p99 abs logit diff | -- | 0.800 | -- | -- |
| max abs logit diff | -- | 1.603 | -- | -- |

0.7% of windows land on opposite sides of the threshold between the two artifacts.
That is the honest cost of quantization for a reader: roughly one highlighted passage
in 140 would differ had the extension shipped fp32 at 599 MB.

## What is not measured

- **Document-level FPR.** scope.md 4.5's bound is chunk-level; what a reader sees is a
  highlighted *run*, after the length penalty, run pooling, the document prior and the
  150-word minimum. The plan requires doc-level FPR <= 3% through the full section 8
  pipeline. Not run.
- **RAID.** `raid_eval` is implemented and has not been run.
- **In-the-wild eval.** The 500-1,000 held-out hard-negative documents the plan names
  as the number that should actually govern shipping do not exist yet.
- These figures inherit corpus v3's defects, recorded in `corpus-gate.md` and
  `training-r1.md`: three surface probes above their bands, a residual specificity
  confound, and a genre mix that does not match the config because `genre_targets` is
  enforced nowhere. **Expect the open web to be worse than 0.63% FPR.**
