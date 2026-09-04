# training/

Everything that produces a model bundle. Python 3.12 under `uv`; ruff and mypy strict.

## Running it

Stages run on Modal against one Volume. A thin CLI launches them; nothing long-running
depends on the local client staying connected.

```bash
# 1. Harvest the human side (spawns server-side, returns immediately)
uv run python scripts/launch_harvest.py --plan fineweb
uv run python scripts/launch_harvest.py --plan rest
uv run python scripts/launch_harvest.py --status

# 2. Generate the AI side from harvested seeds
uv run python scripts/launch_generation.py --shards 120 --per-shard 600
uv run python scripts/launch_generation.py --status

# 3. Build windows, then gate, then train
modal run -m slopmarker.corpus.modal_app   # or call build_corpus directly
uv run python scripts/run_pipeline.py --stage gate   --version v2
uv run python scripts/run_pipeline.py --stage train  --version v2 --run-id r1
uv run python scripts/run_pipeline.py --stage export --version v2 --run-id r1
```

Model IDs and provider constraints move, so they are discovered rather than assumed:

```bash
uv run modal run scripts/discover_models.py   # what each provider actually serves
uv run modal run scripts/probe_models.py      # which parameters each model accepts
```

## Test suites

Three, because the dependency stacks conflict and never share an environment.

```bash
uv run --extra corpus --extra dev --extra evaluate pytest tests --ignore=tests/test_model.py
uv run --extra train  --extra dev pytest tests/test_model.py
uv run python scripts/make_fixtures.py        # regenerate the cross-language fixtures
```

## Things worth knowing before changing anything here

**The corpus gate is not advisory.** `eval/shortcut_probe.py` fits deliberately weak
baselines and every one has a *band*, not a ceiling. Bag-of-words at 0.99 means the
corpus is trivially separable; at 0.55 it means the classes were preprocessed
differently. Both fail. It caught two real defects on the first real corpus.

**Markdown is stripped from both classes.** Not to remove model style — at inference
the extension reads rendered DOM text, so `**` and `#` cannot reach the model in
production, while a model emits them verbatim. Before this, a literal asterisk was
8x more frequent in AI text than human.

**Genre is assigned from text alone**, by the same function on both sides. The AI side
has no URL, so a URL-derived human label and a prompt-derived AI label would make the
labelling process itself carry the class.

**The AI fraction is always measured, never requested.** A model told to rewrite 30%
will not comply. Mixed documents are spliced and the resulting fraction is measured;
the target is provenance only.

**Windows carry span-derived labels.** A window cut from the human half of a half-AI
document is entirely human. Anything that edits document text must move the spans with
it — see `_strip_markdown_keeping_spans`.

**Calibration runs on the int8 artifact, not the PyTorch model** (scope.md 5.5), which
is why export, quantization and calibration are one Modal function rather than three.

## Layout

```
configs/     corpus.yaml (seeds, window PMF, quotas, caps), hosts/ (mining lists)
src/slopmarker/
  corpus/    harvest, mining, cleaning, dedup, decontamination, generation, build
  data/      normalize / sentences / chunking (cross-language coupling points),
             windows (span-aware sampler), schema, splits, dataset
  model/     modeling, losses, layerwise decay, training loop, Modal app
  export/    ONNX export, parity across shapes, int8 quantization, bundle
  eval/      metrics, calibration, aggregation (scope.md 8), corpus gate, reports
scripts/     thin launchers; nothing long-running lives in the local process
```
