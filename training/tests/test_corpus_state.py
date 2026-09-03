"""Config hashing and the shard/resume protocol."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from slopmarker.corpus.config import DEFAULT_CONFIG, SplitConfig, load_config
from slopmarker.corpus.state import (
    clear_stage,
    is_done,
    mark_done,
    read_shard,
    read_summary,
    shard_path,
    write_shard,
)


class TestConfig:
    def test_loads_the_checked_in_config(self) -> None:
        cfg = load_config()
        assert cfg.corpus_seed
        assert cfg.human_windows == cfg.ai_windows, "classes must be balanced"
        assert cfg.max_date == "2022-01-01"
        assert len(cfg.short_hash) == 8

    def test_window_buckets_are_contiguous_and_normalized(self) -> None:
        cfg = load_config()
        buckets = cfg.window_buckets
        assert sum(w for _, _, w in buckets) == pytest.approx(1.0)
        for (_, hi, _), (lo, _, _) in pairwise(buckets):
            assert hi == lo, "buckets must tile the range without gaps or overlap"
        assert buckets[0][0] == 40
        assert buckets[-1][1] == 400

    def test_genre_targets_sum_to_the_human_quota(self) -> None:
        cfg = load_config()
        assert sum(cfg.genre_targets.values()) == cfg.human_windows

    def test_hash_changes_with_content(self, tmp_path: Path) -> None:
        original = DEFAULT_CONFIG.read_text(encoding="utf-8")
        edited = tmp_path / "corpus.yaml"
        edited.write_text(original.replace("corpus_seed: 20260903", "corpus_seed: 1"), "utf-8")
        assert load_config(edited).content_hash != load_config().content_hash

    def test_split_proportions_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            SplitConfig(train=0.9, val=0.05, calibration=0.05, test=0.05)


class TestShards:
    def test_roundtrip(self, tmp_path: Path) -> None:
        path = shard_path(tmp_path, "raw", "part-00000")
        rows = ['{"a": 1}', '{"b": "\u00e9\u4e2d"}', '{"c": null}']
        assert write_shard(path, rows) == 3
        assert list(read_shard(path)) == rows

    def test_empty_shard(self, tmp_path: Path) -> None:
        path = shard_path(tmp_path, "raw", "empty")
        assert write_shard(path, []) == 0
        assert list(read_shard(path)) == []

    def test_many_rows_survive_block_boundaries(self, tmp_path: Path) -> None:
        path = shard_path(tmp_path, "raw", "big")
        rows = [f'{{"i": {i}, "pad": "{"x" * 200}"}}' for i in range(20000)]
        assert write_shard(path, rows) == len(rows)
        assert list(read_shard(path)) == rows

    def test_write_is_atomic(self, tmp_path: Path) -> None:
        """A crash mid-write must not leave something a resume would trust."""
        path = shard_path(tmp_path, "raw", "part-00000")

        def explode() -> object:
            yield '{"a": 1}'
            raise RuntimeError("container died")

        with pytest.raises(RuntimeError):
            write_shard(path, explode())
        assert not path.exists()


class TestResume:
    def test_marker_lifecycle(self, tmp_path: Path) -> None:
        assert not is_done(tmp_path, "clean", "abc12345", "part-0")
        mark_done(tmp_path, "clean", "abc12345", "part-0", {"rows": 7})
        assert is_done(tmp_path, "clean", "abc12345", "part-0")
        assert read_summary(tmp_path, "clean", "abc12345", "part-0") == {"rows": 7}

    def test_config_change_invalidates_only_that_stage(self, tmp_path: Path) -> None:
        mark_done(tmp_path, "clean", "oldhash1", "part-0", {})
        mark_done(tmp_path, "harvest", "oldhash1", "part-0", {})
        # A new config hash means nothing is considered done...
        assert not is_done(tmp_path, "clean", "newhash2", "part-0")
        # ...but work under the old hash is still recorded.
        assert is_done(tmp_path, "harvest", "oldhash1", "part-0")

    def test_clear_stage_is_scoped(self, tmp_path: Path) -> None:
        mark_done(tmp_path, "clean", "h", "part-0", {})
        mark_done(tmp_path, "clean", "h", "part-1", {})
        mark_done(tmp_path, "harvest", "h", "part-0", {})
        assert clear_stage(tmp_path, "clean", "h") == 2
        assert not is_done(tmp_path, "clean", "h", "part-0")
        assert is_done(tmp_path, "harvest", "h", "part-0")
