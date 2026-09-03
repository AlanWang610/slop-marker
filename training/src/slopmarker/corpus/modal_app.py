"""Modal app for corpus construction.

Stages fan out over shards and write zstd JSONL to a Volume. Every mapped stage checks
its completion marker first, so a killed run resumes at shard granularity rather than
from the beginning.

Shards are large on purpose. Modal Volumes contend above roughly five concurrent
commits, so a wide fan-out of small writers is slower than a narrow fan-out of big
ones, and the resume markers make large units cheap to retry.
"""

from __future__ import annotations

import os
from typing import Any

import modal

APP_NAME = "slopmarker-corpus"
VOLUME_NAME = "slopmarker-data"
DATA_ROOT = "/data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name("slopmarker-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "datasets>=3",
        "datasketch>=1.6",
        "tldextract>=5",
        "py3langid>=0.4.0",
        "numpy>=2",
        "pyarrow>=17",
        "pydantic>=2.9",
        "pyyaml>=6",
        "tokenizers>=0.21",
        "zstandard>=0.23",
    )
    .env({"HF_HOME": "/hf-cache", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_python_source("slopmarker")
    .add_local_dir("configs", remote_path="/root/configs")
)

hf_secret = modal.Secret.from_name("huggingface")

VOLUMES = {DATA_ROOT: volume, "/hf-cache": hf_cache}


def _config() -> Any:
    """Load the config from the image copy of configs/."""
    from pathlib import Path

    from slopmarker.corpus.config import load_config

    return load_config(Path("/root/configs/corpus.yaml"))


@app.function(
    image=image,
    volumes=VOLUMES,
    secrets=[hf_secret],
    cpu=2.0,
    memory=8192,
    timeout=6 * 3600,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0),
)
def harvest_shard(
    source: str,
    shard_index: int,
    num_shards: int,
    limit: int | None = None,
    fineweb_sample: str | None = None,
    keep_natural_rate: float = 0.25,
) -> dict[str, Any]:
    """Harvest and clean one shard of one source."""
    from pathlib import Path

    from slopmarker.corpus.clean import clean_document
    from slopmarker.corpus.sources import HARVESTERS
    from slopmarker.corpus.state import is_done, mark_done, read_summary, shard_path, write_shard

    root = Path(DATA_ROOT)
    cfg = _config()
    name = f"{source}-{shard_index:05d}"
    if is_done(root, "interim", cfg.short_hash, name):
        return read_summary(root, "interim", cfg.short_hash, name)

    harvester = HARVESTERS[source]
    kwargs: dict[str, Any] = {"shard": (shard_index, num_shards)}
    if limit is not None:
        kwargs["limit" if source != "pile" else "limit_per_subset"] = limit
    if source == "fineweb":
        kwargs["keep_natural_rate"] = keep_natural_rate
        if fineweb_sample:
            kwargs["sample"] = fineweb_sample
    if source == "reddit":
        kwargs.pop("limit", None)
        if limit is not None:
            kwargs["limit_per_subreddit"] = limit

    # FineWeb has already run C4 and Gopher filters; rerunning them removes nothing.
    check_quality = source != "fineweb"
    kept, dropped = 0, {}
    rows = []
    for doc in harvester(**kwargs):
        result = clean_document(doc.text, cfg.cleaning, check_quality=check_quality)
        if not result.kept:
            dropped[result.reason] = dropped.get(result.reason, 0) + 1
            continue
        doc.text = result.text
        doc.n_words = result.n_words
        rows.append(doc.to_json())
        kept += 1

    written = write_shard(shard_path(root, "interim", name), rows)
    volume.commit()
    summary = {"source": source, "shard": shard_index, "kept": written, "dropped": dropped}
    mark_done(root, "interim", cfg.short_hash, name, summary)
    volume.commit()
    return summary


@app.function(image=image, volumes=VOLUMES, cpu=8.0, memory=65536, timeout=4 * 3600)
def build_corpus(version: str) -> dict[str, Any]:
    """Deduplicate, decontaminate, assign splits and cut windows.

    Single container: clustering and split assignment need a global view, and at this
    scale a whole-corpus MinHashLSH is a few hundred MB.
    """
    from pathlib import Path

    from slopmarker.corpus.build import build

    volume.reload()
    return build(Path(DATA_ROOT), _config(), version)


@app.function(
    image=image, volumes=VOLUMES, secrets=[hf_secret], cpu=4.0, memory=32768, timeout=3600
)
def build_raid_index(limit: int = 20000) -> dict[str, Any]:
    """Cache RAID's human documents so decontamination need not re-stream them."""
    import json
    from pathlib import Path

    from slopmarker.corpus.decontam import load_raid_human_texts

    root = Path(DATA_ROOT) / "external"
    root.mkdir(parents=True, exist_ok=True)
    texts = load_raid_human_texts(limit=limit)
    (root / "raid_human.json").write_text(json.dumps(texts), encoding="utf-8")
    volume.commit()
    return {"raid_human_documents": len(texts)}


@app.local_entrypoint()
def main(
    sources: str = "fineweb,cc_news,pile,pes2o,reddit",
    shards: int = 8,
    limit: int = 0,
    fineweb_sample: str = "",
    keep_natural_rate: float = 0.25,
) -> None:
    """Launch the harvest, then build the corpus.

    Example pilot run:
        modal run -m slopmarker.corpus.modal_app -- \
            --sources fineweb,cc_news --shards 2 --limit 2000
    """
    jobs = [
        (source, index, shards, limit or None, fineweb_sample or None, keep_natural_rate)
        for source in sources.split(",")
        for index in range(shards)
    ]
    total = 0
    for summary in harvest_shard.starmap(jobs, return_exceptions=True):
        if isinstance(summary, Exception):
            print(f"  shard failed: {summary}")
            continue
        total += summary["kept"]
        print(f"  {summary['source']}-{summary['shard']:05d}: kept {summary['kept']}")
    print(f"harvested {total} documents")


if os.environ.get("SLOPMARKER_DEBUG"):  # pragma: no cover
    print(f"modal app {APP_NAME} volume {VOLUME_NAME}")
