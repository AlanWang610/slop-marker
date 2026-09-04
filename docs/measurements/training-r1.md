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

## Validation results

| metric | value |
|---|---|
| AUROC | 0.9928 |
| pAUC at 2% FPR | 0.9632 |
| FPR at 90% recall | 0.0014 |
| excess BCE | 0.072 |

Excess BCE is the number to read rather than raw loss. Soft-target BCE has a floor
equal to the mean target entropy, so with mixed-authorship rows in the corpus the raw
loss plateaus well above zero and never approaches it -- which looks exactly like a
broken run. 0.072 against a floor of zero means the model fits the fractional targets
closely.

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

All well below half, which is why the balanced sampler exists: at these rates the model
could read the label off the genre. The sampler equalises the AI rate inside each genre
to 0.50 while leaving genre marginals untouched.

## How to read these numbers

They are optimistic, and the reason is documented in `corpus-gate.md`. Three surface
probes exceed their a-priori bands, and feature analysis attributes part of that to a
residual specificity confound -- human documents carry proper nouns and parenthetical
asides that AI text on the same topic does not match. A validation FPR of 0.14% at 90%
recall will not survive contact with the open web unchanged.

The numbers that matter for shipping are the per-genre false-positive rates on the
held-out calibration and test splits, measured on the quantized artifact, in
`calibration-*.md`.
