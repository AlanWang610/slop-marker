"""The calibration.json contract (scope.md 5), shared with the extension.

Everything the extension needs to turn a logit into a highlight lives here, so that a
model refresh is a new bundle and not an extension release (scope.md 4.7). That is why
the scope.md 8 constants -- the length-penalty pivot, the run-word minimum and the
document prior -- are fields rather than literals in TypeScript.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class AggregateParams:
    """Constants for scope.md 8. Defaults are the values scope.md states."""

    length_penalty_words: int = 80
    run_min_words: int = 150
    doc_prior_min_fraction: float = 0.20
    # scope.md 8 says "raise t_on by +0.05" in probability space. At a <=1% FPR operating
    # point t_on lands near 0.95-0.99, so +0.05 yields an unreachable threshold above 1.0
    # and every sparse document silently stops being scoreable. The bump is applied in
    # log-odds space instead, where it is scale-free and cannot leave the unit interval.
    doc_prior_bump_logodds: float = 0.25


@dataclass(frozen=True)
class Calibration:
    version: str
    temperature: float
    t_on: float
    t_off: float
    min_words: int = 40
    max_length: int = 512
    aggregate: AggregateParams = field(default_factory=AggregateParams)

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if not 0.0 < self.t_off < self.t_on < 1.0:
            raise ValueError(f"need 0 < t_off < t_on < 1, got t_off={self.t_off} t_on={self.t_on}")

    def to_json(self) -> str:
        payload = {"schema_version": SCHEMA_VERSION, **asdict(self)}
        return json.dumps(payload, indent=2, sort_keys=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Calibration:
        raw = json.loads(text)
        raw.pop("schema_version", None)
        agg = raw.pop("aggregate", {})
        return cls(**raw, aggregate=AggregateParams(**agg))

    @classmethod
    def load(cls, path: Path) -> Calibration:
        return cls.from_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8", newline="\n")


def t_off_from(t_on: float, delta_logodds: float) -> float:
    """Derive t_off from t_on in log-odds space.

    scope.md 8 specifies `t_off = t_on - 0.1` in probability space, but that band is
    wildly non-uniform: at t_on=0.97 it spans 3.48 -> 1.90 in log-odds, and at t_on=0.60
    it spans 0.41 -> 0.00. A fixed log-odds offset behaves the same wherever t_on lands.
    """
    return sigmoid(logit(t_on) - delta_logodds)
