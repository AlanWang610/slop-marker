# Corpus gate across three builds

The gate runs deliberately weak baselines and every probe has a *band*, not a ceiling.
Bands were set a priori from design reasoning, before any corpus existed. Three builds:

| probe | v1 | v2 | v3 | band | v3 verdict |
|---|---|---|---|---|---|
| P1 surface features | 0.888 | 0.887 | 0.867 | 0.55-0.80 | FAIL |
| P2 punctuation and spacing only | 0.856 | 0.848 | 0.794 | <=0.70 | FAIL |
| P3 bag of words | 0.932 | 0.921 | 0.927 | 0.70-0.93 | pass |
| P5 topic only (content words) | 0.876 | 0.860 | 0.873 | <=0.78 | FAIL |
| P6 length only | 0.527 | 0.529 | 0.540 | <=0.65 | pass |
| P9 held-out generator transfer | 0.004 | 0.007 | **0.036** | >=0.03 | pass |

## What each build changed

**v1 -> v2: markdown syntax removed from both classes.** AI text carried literal `**`
and `#`; human web text has none, because trafilatura strips markup during extraction.
Measured at 8.3x more asterisks in AI text. Removing it is justified by production
behaviour rather than taste: the extension reads rendered DOM text, so those characters
cannot reach the model at inference. Also capped genre share, since academic prose was
45% of windows and its human side is dense with citations the AI side did not reproduce.

**v2 -> v3: the seed's own figures passed into every prompt.** Breaking the digit gap
down by prompt style located the cause exactly -- `rewrite`, the one style that reads
the seed document, reached 0.77x the human digit density, while `topic_prompt`, handed
only a subject line, managed 0.13x. A model given a topic and no specifics writes prose
with almost no numbers, while real writing on the same subjects is full of dates and
quantities. That was not a property of AI writing; it was a property of the prompt.

P2 fell 0.062 and the held-out transfer gap rose five-fold and now passes.

## What the residual failures actually are

Feature weights and single-feature AUCs from the P1 probe on v3:

| feature | solo AUC | reading |
|---|---|---|
| type-token ratio | 0.716 | genuine AI style |
| capitalised words | 0.326 (human higher) | residual artifact |
| parentheses | 0.321 (human higher) | residual artifact |
| mean word length | 0.647 | genuine AI style |
| em dash | 0.622 | genuine AI style |
| comma rate | 0.618 | genuine AI style |

The strongest single driver is lexical diversity, which is a property of AI writing and
exactly what a detector should key on. So is em-dash rate, word length and comma
density -- these were deliberately left alone, since removing them would make the corpus
easier than the web.

The residual artifact is the same family as the digit problem, one level down: human
documents carry proper nouns, named sources and parenthetical asides that AI text about
the same topic does not reproduce as densely. Passing figures into the prompts reduced
this; passing entities did not fully close it.

## Honest position

Two of the three remaining failures are substantially explained by legitimate style
signal, and P9 passing is evidence the corpus is not merely encoding our harness -- the
bag-of-words probe now loses measurable ground on generators it has never seen.

But this is not proof, and the bands were not met. The residual specificity confound is
real and will inflate measured performance relative to production, where AI text is
often edited and human text is not uniformly citation-dense. The per-genre
false-positive numbers should be read with that in mind.

A fourth corpus iteration targeting entity density is the obvious next step. It was not
run because each cycle is roughly ninety minutes and the returns were diminishing;
that is a scheduling decision, not a claim that the corpus is clean.
