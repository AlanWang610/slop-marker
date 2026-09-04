# Corpus gate, first real corpus (v1) -- FAILED

964,319 windows from 253,199 documents. Five of six probes failed. Recorded because
the diagnosis drove two corpus changes, and because a gate that never fails is not a
gate.

| probe | AUC | band | verdict |
|---|---|---|---|
| P1 surface features | 0.888 | 0.55-0.80 | FAIL |
| P2 punctuation and spacing only | 0.856 | <=0.70 | FAIL |
| P3 bag of words | 0.932 | 0.70-0.93 | FAIL |
| P5 topic only (content words) | 0.876 | <=0.78 | FAIL |
| P6 length only | 0.527 | <=0.65 | pass |
| P9 held-out generator transfer | 0.004 gap | >=0.03 | FAIL |

P2 is the one that gives the game away: erase every letter, keep only punctuation and
spacing, and the classes are still separable at 0.856. P9 says the same thing from the
other side -- in-distribution 0.932 against held-out 0.927 means the probe transfers
perfectly to generators it has never seen, so it learned the harness rather than AI
writing.

P6 passing is the one piece of good news: seed length mirroring worked, so document
length carries almost no class information.

## Surface statistics that explain it

Measured over 8,000 sampled training windows, as rate per word.

| feature | human | AI | ratio |
|---|---|---|---|
| asterisk | 0.00073 | 0.00607 | **8.32** |
| em dash | 0.00043 | 0.00312 | 7.26 |
| hash | 0.00021 | 0.00069 | 3.29 |
| digit | 0.0873 | 0.0290 | **0.33** |
| parenthesis | 0.0116 | 0.0026 | **0.22** |
| semicolon | 0.00208 | 0.00095 | 0.46 |
| double quote | 0.0038 | 0.0021 | 0.55 |

Two distinct causes.

**Markdown syntax.** Asterisks and hashes are markup a model emits verbatim and that
trafilatura strips out of extracted web text, so the human side has essentially none.
Removed from both classes -- justified because at inference the extension reads
rendered DOM text, where a heading is the words inside an `<h2>` and bold is the words
inside a `<strong>`. Those characters cannot reach the model in production.

**Composition.** Academic prose was 45% of windows, and its human side carries
citations, page numbers and reference markers that the AI side does not reproduce.
That is what drives the digit and parenthesis ratios, and it makes punctuation a class
signal rather than a genre one. Genre share is now capped at 25%.

The em-dash ratio is deliberately left alone. That is AI writing style, it appears on
the real web, and removing it would make the corpus easier than reality.
