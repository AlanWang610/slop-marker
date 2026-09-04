"""Spawn AI-side generation against the deployed Modal app, then exit.

    uv run python scripts/launch_generation.py --shards 24 --per-shard 5000
    uv run python scripts/launch_generation.py --status
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

APP = "slopmarker-corpus"
STATE = Path(__file__).resolve().parent / ".generation_calls.json"


def spawn(shards: int, per_shard: int, workers: int) -> None:
    fn = modal.Function.from_name(APP, "generate_shard")
    calls = []
    for index in range(shards):
        call = fn.spawn(index, shards, per_shard, workers)
        calls.append({"id": call.object_id, "shard": index})
    STATE.write_text(json.dumps(calls, indent=2), encoding="utf-8")
    print(f"spawned {shards} generation shards x {per_shard} docs ({workers} workers each)")
    print(f"target: {shards * per_shard} AI documents")


def status() -> None:
    if not STATE.exists():
        print("no generation calls recorded")
        return
    calls = json.loads(STATE.read_text(encoding="utf-8"))
    done = running = failed = 0
    total = 0
    models: dict[str, int] = {}
    styles: dict[str, int] = {}
    for entry in calls:
        call = modal.FunctionCall.from_id(entry["id"])
        try:
            result = call.get(timeout=0)
            done += 1
            total += result.get("generated", 0)
            for name, n in (result.get("by_model") or {}).items():
                models[name] = models.get(name, 0) + n
            for name, n in (result.get("by_style") or {}).items():
                styles[name] = styles.get(name, 0) + n
        except TimeoutError:
            running += 1
        except Exception as exc:
            failed += 1
            entry["error"] = str(exc)[:160]
    print(f"done {done}  running {running}  failed {failed}")
    print(f"AI documents generated: {total}")
    if models:
        print("by model:", dict(sorted(models.items(), key=lambda kv: -kv[1])))
    if styles:
        print("by style:", dict(sorted(styles.items(), key=lambda kv: -kv[1])))
    for entry in calls:
        if "error" in entry:
            print(f"   shard {entry['shard']}: {entry['error']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=120)
    parser.add_argument("--per-shard", type=int, default=600)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        status()
    else:
        spawn(args.shards, args.per_shard, args.workers)


if __name__ == "__main__":
    main()
