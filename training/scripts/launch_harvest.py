"""Spawn harvest jobs against the deployed Modal app, then exit.

`modal run` keeps a client connection open for the life of the job, so a dropped
terminal kills the run. `spawn` hands the work to Modal and returns immediately, which
is what a long harvest needs. Progress is read back from the Volume, not from stdout.

    uv run python scripts/launch_harvest.py --plan full
    uv run python scripts/launch_harvest.py --status
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

APP = "slopmarker-corpus"

# (source, shards, limit per shard, keep_natural_rate)
PLANS: dict[str, list[tuple[str, int, int, float]]] = {
    "pilot": [
        ("fineweb", 4, 500, 0.10),
        ("cc_news", 2, 500, 0.0),
        ("pile", 2, 200, 0.0),
        ("pes2o", 2, 300, 0.0),
        ("reddit", 2, 400, 0.0),
    ],
    "full": [
        # FineWeb carries the genres no curated corpus supplies: press releases,
        # SEO copy, corporate blogs, templated product pages, non-native forums.
        ("fineweb", 16, 5000, 0.10),
        ("cc_news", 8, 8000, 0.0),
        ("pile", 6, 2500, 0.0),
        ("pes2o", 6, 4000, 0.0),
        ("reddit", 10, 6000, 0.0),
    ],
}

STATE = Path(__file__).resolve().parent / ".harvest_calls.json"


def spawn(plan: str) -> None:
    harvest = modal.Function.from_name(APP, "harvest_shard")
    calls = []
    for source, shards, limit, natural in PLANS[plan]:
        for index in range(shards):
            call = harvest.spawn(source, index, shards, limit, None, natural)
            calls.append({"id": call.object_id, "source": source, "shard": index})
    STATE.write_text(json.dumps(calls, indent=2), encoding="utf-8")
    print(f"spawned {len(calls)} shards for plan '{plan}'")
    for source, shards, limit, _ in PLANS[plan]:
        print(f"   {source:10} {shards} shards x {limit}")


def status() -> None:
    if not STATE.exists():
        print("no spawned calls recorded")
        return
    calls = json.loads(STATE.read_text(encoding="utf-8"))
    done = running = failed = 0
    for entry in calls:
        call = modal.FunctionCall.from_id(entry["id"])
        try:
            result = call.get(timeout=0)
            done += 1
            entry["kept"] = result.get("kept")
        except TimeoutError:
            running += 1
        except Exception as exc:
            failed += 1
            entry["error"] = str(exc)[:120]
    print(f"done {done}  running {running}  failed {failed}")
    kept = sum(e.get("kept") or 0 for e in calls)
    if kept:
        print(f"documents harvested so far: {kept}")
    for entry in calls:
        if "error" in entry:
            print(f"   {entry['source']}-{entry['shard']}: {entry['error']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", choices=sorted(PLANS))
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        status()
    elif args.plan:
        spawn(args.plan)
    else:
        parser.error("pass --plan or --status")


if __name__ == "__main__":
    main()
