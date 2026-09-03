"""The corpus gate: deliberately weak baselines, run before any training starts.

The point is not to build a detector. It is to find out whether the corpus is separable
for reasons that have nothing to do with AI writing -- markdown, length, topic, or an
artifact of our own generation harness.

Every probe has a *band*, not a ceiling, and that is the part people get wrong. A
bag-of-words model at 0.99 means the corpus is trivially separable and the real number
will not survive contact with the web. But the same model at 0.55 also means something
is broken, because AI text genuinely does carry lexical signatures -- getting none
implies the two classes were preprocessed differently, or over-scrubbed.

The sharpest single check is the last one. The generation harness is shared across every
model; a model's writing style is not. If a bag-of-words probe transfers to unseen
generators as well as it does in-distribution, it has learned our pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.normalize import collapse_whitespace, count_words
from .metrics import auroc

# Codepoints rather than literals, matching data/normalize.py.
EM_DASH = chr(0x2014)
CURLY = "".join(map(chr, (0x2018, 0x2019, 0x201C, 0x201D)))
SLOP_MARKERS = ("in conclusion", "moreover", "delve", "tapestry", "furthermore", "landscape")
CONTRACTIONS = re.compile(r"\b\w+'(s|t|re|ve|ll|d|m)\b", re.I)
STOPWORDS = frozenset(
    """
    the a an and or but of in on at to for with from by as is are was were be been being
    this that these those it its his her their our your my we you they he she i not no
    all any some more most other such only own same so than too very can will just should
    now about into over after before between during under above then there here what which
    """.split()  # noqa: SIM905 - a word list reads better than a literal
)


@dataclass
class ProbeResult:
    name: str
    auc: float
    low: float
    high: float
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.low <= self.auc <= self.high

    def __str__(self) -> str:
        verdict = "pass" if self.passed else "FAIL"
        return f"{self.name:34} auc={self.auc:.3f} band=[{self.low:.2f},{self.high:.2f}] {verdict}"


def surface_features(text: str) -> list[float]:
    """24 hand-built features. If these alone separate the classes, the corpus is broken."""
    collapsed = collapse_whitespace(text)
    words = collapsed.split(" ")
    n_words = max(1, len(words))
    lines = text.splitlines() or [text]
    sentence_lengths = [len(s.split()) for s in re.split(r"[.!?]+", collapsed) if s.strip()]
    lower = collapsed.lower()
    return [
        n_words,
        len(collapsed),
        float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
        float(np.var(sentence_lengths)) if sentence_lengths else 0.0,
        collapsed.count(EM_DASH) / n_words,
        sum(collapsed.count(c) for c in CURLY) / n_words,
        sum(1 for ln in lines if ln.lstrip().startswith("#")) / len(lines),
        sum(1 for ln in lines if re.match(r"^\s*([-*•]|\d+[.)])\s", ln)) / len(lines),
        collapsed.count("**") / n_words,
        len(CONTRACTIONS.findall(collapsed)) / n_words,
        collapsed.count(",") / n_words,
        collapsed.count(";") / n_words,
        collapsed.count(":") / n_words,
        collapsed.count("!") / n_words,
        collapsed.count("?") / n_words,
        collapsed.count('"') / n_words,
        collapsed.count("(") / n_words,
        len({w.lower() for w in words}) / n_words,  # type-token ratio
        sum(1 for w in words if w.lower() in STOPWORDS) / n_words,
        sum(1 for w in words if any(c.isdigit() for c in w)) / n_words,
        sum(1 for w in words if w[:1].isupper()) / n_words,
        float(np.mean([len(w) for w in words])),
        sum(lower.count(m) for m in SLOP_MARKERS) / n_words,
        len(lines) / n_words,
    ]


def _punctuation_shape(text: str) -> str:
    """Letters collapsed to 'a', so only punctuation and spacing survive."""
    return re.sub(r"[^\W\d_]", "a", collapse_whitespace(text))


def _content_words(text: str) -> str:
    """Function words removed, so only topic remains."""
    return " ".join(w for w in collapse_whitespace(text).split(" ") if w.lower() not in STOPWORDS)


def _fit_auc(train_x: Any, train_y: Any, test_x: Any, test_y: Any) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if len(set(train_y)) < 2 or len(set(test_y)) < 2:
        return float("nan")
    scaler = StandardScaler(with_mean=not hasattr(train_x, "tocsr"))
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(train_x, train_y)
    scores = model.decision_function(test_x)
    test_y = np.asarray(test_y)
    return auroc(scores[test_y == 0], scores[test_y == 1])


def _tfidf_auc(train_texts, train_y, test_texts, test_y, **kwargs) -> float:  # type: ignore[no-untyped-def]
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=50_000, **kwargs)
    return _fit_auc(
        vectorizer.fit_transform(train_texts), train_y, vectorizer.transform(test_texts), test_y
    )


def run_probes(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    *,
    heldout_texts: list[str] | None = None,
    heldout_labels: list[int] | None = None,
) -> list[ProbeResult]:
    results: list[ProbeResult] = []

    train_surface = np.array([surface_features(t) for t in train_texts])
    test_surface = np.array([surface_features(t) for t in test_texts])
    results.append(
        ProbeResult(
            "P1 surface features",
            _fit_auc(train_surface, train_labels, test_surface, test_labels),
            0.55,
            0.80,
            "24 hand features: length, punctuation, markdown, slop markers",
        )
    )

    results.append(
        ProbeResult(
            "P2 punctuation and spacing only",
            _tfidf_auc(
                [_punctuation_shape(t) for t in train_texts],
                train_labels,
                [_punctuation_shape(t) for t in test_texts],
                test_labels,
                analyzer="char",
                ngram_range=(2, 5),
            ),
            0.0,
            0.70,
            "letters erased; only shape survives",
        )
    )

    bow = _tfidf_auc(train_texts, train_labels, test_texts, test_labels, ngram_range=(1, 2))
    results.append(
        ProbeResult(
            "P3 bag of words",
            bow,
            0.70,
            0.93,
            "too high means trivially separable; too low means over-scrubbed",
        )
    )

    results.append(
        ProbeResult(
            "P5 topic only (content words)",
            _tfidf_auc(
                [_content_words(t) for t in train_texts],
                train_labels,
                [_content_words(t) for t in test_texts],
                test_labels,
            ),
            0.0,
            0.78,
            "high means the classes differ by subject, not by style",
        )
    )

    length_train = np.array([[count_words(t)] for t in train_texts])
    length_test = np.array([[count_words(t)] for t in test_texts])
    results.append(
        ProbeResult(
            "P6 length only",
            _fit_auc(length_train, train_labels, length_test, test_labels),
            0.0,
            0.65,
            "high means seed length mirroring failed",
        )
    )

    if heldout_texts and heldout_labels and len(set(heldout_labels)) > 1:
        heldout = _tfidf_auc(
            train_texts, train_labels, heldout_texts, heldout_labels, ngram_range=(1, 2)
        )
        gap = bow - heldout
        results.append(
            ProbeResult(
                "P9 held-out generator transfer",
                gap,
                0.03,
                1.0,
                f"bow in-dist {bow:.3f} vs held-out {heldout:.3f}; "
                "a small gap means the probe learned our harness, not AI writing",
            )
        )

    return results


def gate(results: list[ProbeResult]) -> tuple[bool, list[str]]:
    failures = [str(r) for r in results if not r.passed and not np.isnan(r.auc)]
    return not failures, failures
