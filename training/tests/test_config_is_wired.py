"""Every knob in corpus.yaml must be read by something.

Four separate defects in this pipeline had the same shape: a value was written into
the config, parsed into the config object, and never read. The corpus ran with
genre_targets ignored, which is why academic prose reached 237k windows against a
target of 90k while technical_docs reached 14k against 75k -- and the thin
technical_docs calibration set that followed is what set the shipped threshold.

A test cannot verify that a value is *used correctly*. It can verify the name appears
somewhere outside the parser, which is enough to catch a knob that is pure decoration.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src" / "slopmarker"
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corpus.yaml"

# Knobs that are documented but not yet enforced. Every entry is a known gap, not an
# exemption: the list is here so it is visible in a diff and shrinks over time.
NOT_YET_ENFORCED = {
    # Enforcing these means rebalancing the harvest, not filtering after the fact.
    # genre_targets is the consequential one: ignoring it is why academic prose reached
    # 237k windows against a target of 90k and technical_docs 14k against 75k.
    "genre_targets",
    "human_windows",
    "ai_windows",
    "min_natural_fraction",
    "max_source_fraction",
    "max_host_fraction_per_genre",
    # held_out_domains() is written and tested; nothing reserves the domains yet, so
    # the test split does not currently measure host-template memorization.
    "held_out_domains_per_genre",
    # Mined via the FineWeb language_score band at harvest time, not from this range.
    "l2_language_score_range",
    # Decontamination runs on n-gram overlap only; the MinHash layer is not built.
    "raid_jaccard_threshold",
    # decontam.DEFAULT_NGRAM is 13 and matches, but RaidIndex.build is called without
    # a size, so the two 13s agree by coincidence rather than by wiring.
    "raid_ngram",
    # datasketch derives the band count from num_perm and threshold, so this cannot be
    # set independently. It documents the intent rather than driving anything.
    "bands",
    # The 2022 cutoff IS applied -- every harvester takes cutoff="2022-01-01" -- but
    # from its own default, not from here. The guard works; this knob does not drive it.
    "max_date",
    # No spend counter exists. The cap is a number in a file and nothing consults it,
    # which means the run was never actually bounded by it.
    "budget_usd",
}


def config_keys() -> set[str]:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    keys: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                keys.add(str(key))
                walk(value)

    walk(raw)
    return keys


def source_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in SRC.rglob("*.py") if path.name != "config.py"
    )


def is_read(key: str, text: str) -> bool:
    """Whole-word match: `human_windows` must not be satisfied by a longer name."""
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])", text) is not None


@pytest.mark.parametrize("key", sorted(config_keys() - NOT_YET_ENFORCED))
def test_every_config_key_is_read_somewhere(key: str) -> None:
    assert is_read(key, source_text()), (
        f"{key} is set in corpus.yaml and read nowhere outside config.py. "
        "Either wire it up or add it to NOT_YET_ENFORCED with a reason."
    )


def test_the_unenforced_list_is_honest() -> None:
    """A knob that got wired up must leave the list, or the list stops meaning anything."""
    text = source_text()
    stale = {key for key in NOT_YET_ENFORCED if is_read(key, text)}
    assert not stale, f"now enforced, remove from NOT_YET_ENFORCED: {sorted(stale)}"


def test_the_unenforced_list_is_current() -> None:
    """Every name on the list must still exist in the config."""
    assert config_keys() >= NOT_YET_ENFORCED
