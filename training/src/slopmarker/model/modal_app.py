"""Modal app for training, scoring, export and calibration.

Separate from the corpus app because the images share nothing: this one carries torch
and transformers, that one carries datasets and the provider SDKs.
"""

from __future__ import annotations

from typing import Any

import modal

APP_NAME = "slopmarker-train"
DATA_ROOT = "/data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("slopmarker-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("slopmarker-hf-cache", create_if_missing=True)
VOLUMES = {DATA_ROOT: volume, "/hf-cache": hf_cache}

# transformers is pinned by optimum-onnx (<4.58); 4.57.6 is the newest release that
# also satisfies ModernBERT's >=4.48 requirement. One image for train and export, so
# the checkpoint that trains is the checkpoint that exports.
base = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.8.*", extra_options="--index-url https://download.pytorch.org/whl/cu128"
    )
    .uv_pip_install(
        "transformers==4.57.6",
        "accelerate>=1",
        "numpy>=2",
        "scipy>=1.14",
        "scikit-learn>=1.5",
        "pyyaml>=6",
        "zstandard>=0.23",
        "tokenizers>=0.21",
        "pydantic>=2.9",
        "pyarrow>=17",
    )
    .env(
        {
            "HF_HOME": "/hf-cache",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

export_base = base.uv_pip_install(
    "optimum>=2.1",
    "optimum-onnx==0.1.0",
    "onnx>=1.17",
    "onnxruntime==1.29.0",
    # onnxruntime.quantization.matmul_nbits_quantizer imports it and does not declare it.
    "onnx_ir",
    # RAID is read as a plain HuggingFace dataset. The raid-bench package pins
    # numpy<1.27 and scikit-learn<1.4 and would drag the project onto a 2023 stack.
    "datasets>=3",
)


def _with_local(img: modal.Image) -> modal.Image:
    return img.add_local_python_source("slopmarker").add_local_dir(
        "configs", remote_path="/root/configs"
    )


# Unauthenticated Hub requests get rate-limited (observed during the harvest), and
# the backbone download would fail on a 429 rather than retry usefully.
hf_secret = modal.Secret.from_name("huggingface")

train_image = _with_local(base)
export_image = _with_local(export_base)


def _score_chunks(texts: list[str], session: Any, tokenizer: Any) -> list[float]:
    """Adapt an ONNX session to the ChunkScorer protocol in eval.documents."""
    import numpy as np

    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    logits = session.run(
        None,
        {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        },
    )[0]
    return [float(v) for v in np.asarray(logits).reshape(-1)]


def _held_out_human_documents(root: Any, version: str, limit: int) -> list[Any]:
    """Human documents whose windows landed in the test split, sampled deterministically."""
    import random

    from slopmarker.corpus.build import load_documents
    from slopmarker.data.dataset import load_windows

    by_id = {d.doc_id: d for d in load_documents(root, "interim")}
    doc_ids = sorted({w.doc_id for w in load_windows(root, version, "test")})
    human = [
        by_id[doc_id]
        for doc_id in doc_ids
        if doc_id in by_id and by_id[doc_id].doc_class == "human"
    ]
    random.Random(0).shuffle(human)
    return human[:limit]


@app.function(
    image=train_image,
    gpu="H100",
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=8 * 3600,
    retries=modal.Retries(max_retries=2),
)
def train_model(
    version: str, run_id: str, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    import json
    from pathlib import Path

    from slopmarker.data.dataset import load_windows
    from slopmarker.model.train import TrainConfig, train

    volume.reload()
    root = Path(DATA_ROOT)
    cfg = TrainConfig(**(overrides or {}))

    train_rows = load_windows(root, version, "train")
    val_rows = load_windows(root, version, "val")
    if not train_rows:
        raise RuntimeError(f"no training windows under processed/{version}")
    print(f"train={len(train_rows)} val={len(val_rows)}")

    out_dir = root / "artifacts" / "runs" / run_id
    result = train(train_rows, val_rows, cfg, out_dir)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    volume.commit()
    return result


@app.function(image=train_image, gpu="L40S", volumes=VOLUMES, secrets=[hf_secret], timeout=4 * 3600)
def score_split(run_id: str, version: str, split: str, checkpoint: str = "best") -> dict[str, Any]:
    """Score a split with the PyTorch checkpoint and write a scores file.

    Every downstream consumer -- per-genre FPR, temperature fitting, threshold
    selection, the int8 comparison -- reads a scores file rather than a model, which
    is what makes 'recalibrate on the quantized artifact' a one-word change.
    """
    import json
    from pathlib import Path

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from slopmarker.data.dataset import load_windows
    from slopmarker.data.normalize import collapse_whitespace

    volume.reload()
    root = Path(DATA_ROOT)
    path = root / "artifacts" / "runs" / run_id / checkpoint
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).cuda().eval()

    rows = load_windows(root, version, split)
    out: list[dict[str, Any]] = []
    batch = 64
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            chunk = rows[start : start + batch]
            encoded = tokenizer(
                [collapse_whitespace(r.text) for r in chunk],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**encoded).logits.squeeze(-1).float().cpu().numpy()
            for row, logit in zip(chunk, logits, strict=True):
                out.append(
                    {
                        "window_id": row.window_id,
                        "doc_id": row.doc_id,
                        "genre": row.genre,
                        "ai_fraction": row.ai_fraction,
                        "words": row.n_words,
                        "generator": row.generator,
                        "attack": row.attack,
                        "hard_negative_kind": row.hard_negative_kind,
                        "logit": float(logit),
                    }
                )

    dest = root / "scores" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{split}.{checkpoint}.json").write_text(json.dumps(out), encoding="utf-8")
    volume.commit()
    return {"split": split, "scored": len(out)}


@app.function(
    image=export_image,
    cpu=16.0,
    memory=65536,
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=4 * 3600,
)
def export_and_calibrate(
    run_id: str,
    version: str,
    bundle_version: str,
    checkpoint: str = "best",
    max_score: int = 30000,
) -> dict[str, Any]:
    """Export to ONNX, quantize, calibrate on the int8 artifact, assemble the bundle.

    The ordering is the point of this function existing as one unit: scope.md 5.5
    requires the threshold to be selected on the artifact that actually ships, not on
    the PyTorch model.
    """
    import json
    import random
    import time
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    from slopmarker.data.dataset import load_windows
    from slopmarker.data.normalize import collapse_whitespace, strip_markdown
    from slopmarker.eval.calibrate import calibrate, verify
    from slopmarker.eval.calibration import AggregateParams, Calibration, t_off_from
    from slopmarker.eval.documents import document_fpr
    from slopmarker.export.bundle import assemble
    from slopmarker.export.gates import compare_scores as _compare
    from slopmarker.export.gates import release_gate
    from slopmarker.export.onnx_export import (
        check_parity,
        export_fp32,
        quantize_mixed,
        verify_graph,
    )

    volume.reload()
    root = Path(DATA_ROOT)
    ckpt = root / "artifacts" / "runs" / run_id / checkpoint
    export_dir = root / "artifacts" / "export" / run_id
    report: dict[str, Any] = {"run_id": run_id, "version": bundle_version}

    fp32 = export_fp32(ckpt, export_dir)

    # Scoring runs on CPU through ONNX Runtime, so the splits are subsampled. Sampling
    # is stratified by genre rather than uniform: genre shares run from 25% down to 2%,
    # so a uniform sample starves the smallest genre -- and the smallest genre's tail is
    # what sets the global threshold, so it is the one that must not be thin.
    rng = random.Random(0)

    def sample(rows: list[Any], per_genre: int) -> list[Any]:
        by_genre: dict[str, list[Any]] = {}
        for row in rows:
            by_genre.setdefault(row.genre, []).append(row)
        out: list[Any] = []
        for genre_rows in by_genre.values():
            take = min(per_genre, len(genre_rows))
            out.extend(rng.sample(genre_rows, take))
        return out

    calib_rows = sample(load_windows(root, version, "calibration"), max_score)
    test_rows = sample(load_windows(root, version, "test"), max_score)
    print(f"scoring {len(calib_rows)} calibration and {len(test_rows)} test windows on int8")
    report["n_calibration"] = len(calib_rows)
    report["n_test"] = len(test_rows)
    sample_texts = [collapse_whitespace(r.text) for r in calib_rows[:200]]

    parity = check_parity(ckpt, fp32, sample_texts)
    report["parity"] = parity.to_dict()
    if not parity.passed:
        report["error"] = "fp32 parity failed across shapes"
        volume.commit()
        return report

    # Weight-only, block-wise, with the embedding table at a narrower width than the
    # encoder. Measured against fp32 on identical rows: pAUC 0.9563 against 0.9571, a
    # drop of 0.0008, at 136.7MB. Every dynamic-quantization recipe lost 4-6 points of
    # pAUC because it rescales activations per tensor as well as compressing weights.
    # See docs/measurements/export-r1.md.
    int8 = export_dir / "model.int8.onnx"
    report["quantization"] = quantize_mixed(fp32, int8, encoder_bits=8, embedding_bits=4)
    report["graph"] = verify_graph(fp32, ckpt)

    # Score the calibration and test splits with the *int8* artifact.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 16
    session = ort.InferenceSession(
        str(int8), sess_options=options, providers=["CPUExecutionProvider"]
    )
    fp32_session = ort.InferenceSession(
        str(fp32), sess_options=options, providers=["CPUExecutionProvider"]
    )
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    def score(rows: list[Any], label: str, engine: Any = None) -> list[dict[str, Any]]:
        out = []
        started = time.time()
        for start in range(0, len(rows), 64):
            if start and start % 3200 == 0:
                rate = start / max(1e-6, time.time() - started)
                remaining = (len(rows) - start) / max(1e-6, rate)
                print(
                    f"  {label}: {start}/{len(rows)} at {rate:.0f}/s,"
                    f" ~{remaining / 60:.1f} min left"
                )
            chunk = rows[start : start + 64]
            enc = tokenizer(
                [collapse_whitespace(r.text) for r in chunk],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            logits = (engine or session).run(
                None,
                {
                    "input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64),
                },
            )[0]
            for row, logit in zip(chunk, logits, strict=True):
                out.append(
                    {
                        "genre": row.genre,
                        "ai_fraction": row.ai_fraction,
                        "logit": float(np.asarray(logit).reshape(-1)[0]),
                    }
                )
        return out

    # Quantization is the one step that can degrade the model without tripping any
    # structural check, so both graphs score the same rows before anything downstream
    # trusts the int8 numbers. The subsample is small because the comparison is a
    # ranking test, not an estimate of a 2% tail.
    probe_rows = calib_rows[:4000]
    report["int8_vs_fp32"] = _compare(
        score(probe_rows, "probe-fp32", fp32_session), score(probe_rows, "probe-int8")
    )
    print(f"int8 vs fp32: {report['int8_vs_fp32']}")

    calib_scores = score(calib_rows, "calibration")
    result = calibrate(calib_scores)
    t_off = t_off_from(result.t_on, 0.60)
    checked = verify(score(test_rows, "test"), result.temperature, result.t_on)

    calibration = Calibration(
        version=bundle_version,
        temperature=round(result.temperature, 4),
        t_on=round(result.t_on, 4),
        t_off=round(t_off, 4),
        aggregate=AggregateParams(),
    )
    calibration_path = export_dir / "calibration.json"
    calibration.save(calibration_path)

    report["calibration"] = {
        "temperature": result.temperature,
        "t_on": result.t_on,
        "t_off": t_off,
        "binding_genre": result.binding_genre,
        "recall_global": result.recall,
        "recall_oracle": result.oracle_recall,
        "oracle_gap": result.oracle_gap,
        "per_genre": [vars(g) for g in result.per_genre],
    }
    report["test_verification"] = checked

    # Document-level FPR, through the full scope.md 8 path on the artifact that ships.
    # It has to run here rather than as a separate step, because it needs the
    # calibration this function just produced and it gates the bundle this function is
    # about to assemble. Held-out human documents only: the question is how often a
    # reader is shown a highlight on prose no model wrote.
    documents = [
        strip_markdown(doc.text) for doc in _held_out_human_documents(root, version, limit=1200)
    ]
    print(f"document-level FPR over {len(documents)} held-out human documents")
    report["document_fpr"] = document_fpr(
        documents,
        lambda texts: _score_chunks(texts, session, tokenizer),
        calibration,
    )
    print(f"  {report['document_fpr']}")

    report["gate"] = release_gate(report)

    if report["gate"]["passed"]:
        bundle_dir = root / "artifacts" / "bundles" / bundle_version
        report["bundle"] = assemble(
            int8_model=int8,
            tokenizer_dir=ckpt,
            calibration_path=calibration_path,
            out_dir=bundle_dir,
            provenance=report,
            gate=report["gate"],
            read_only=False,
        )
    else:
        # Everything measured is still written out. A refused bundle is a result, and
        # the numbers behind the refusal are the ones worth reading.
        print("release gate FAILED; no bundle assembled")
        for failure in report["gate"]["failures"]:
            print(f"  - {failure}")

    (export_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), "utf-8")
    volume.commit()
    return report


@app.function(
    image=train_image,
    gpu="L40S",
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=4 * 3600,
)
def raid_eval(run_id: str, checkpoint: str = "best", limit: int = 20000) -> dict[str, Any]:
    """External evaluation on RAID (scope.md 4.6).

    RAID rows are whole documents while the model scores 40-400 word chunks, so the
    whole pipeline runs: chunk, score, then the scope.md 8 aggregation to a document
    score. That is the honest comparison, and it is also the only exercise of the
    aggregation code against text we did not generate.

    Read as a plain HuggingFace dataset rather than through the raid-bench package,
    which pins numpy<1.27 and scikit-learn<1.4 and would drag the project onto a 2023
    stack for no benefit.
    """
    import json
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from slopmarker.data.chunking import chunk_block
    from slopmarker.data.normalize import collapse_whitespace
    from slopmarker.eval.aggregate import Chunk, aggregate
    from slopmarker.eval.calibration import Calibration
    from slopmarker.eval.metrics import auroc, fpr_at_threshold, recall_at_threshold

    volume.reload()
    root = Path(DATA_ROOT)
    ckpt = root / "artifacts" / "runs" / run_id / checkpoint
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).cuda().eval()

    calibration_path = root / "artifacts" / "export" / run_id / "calibration.json"
    cal = (
        Calibration.load(calibration_path)
        if calibration_path.exists()
        else Calibration(version="uncalibrated", temperature=1.0, t_on=0.9, t_off=0.8)
    )

    rows = []
    dataset = load_dataset("liamdugan/raid", split="train", streaming=True)
    for index, row in enumerate(dataset):
        if index >= limit:
            break
        rows.append(
            {
                "text": row["generation"],
                "is_ai": row["model"] != "human",
                "domain": row.get("domain", "unknown"),
                "attack": row.get("attack", "none"),
            }
        )

    def document_score(text: str) -> float | None:
        chunks = chunk_block(collapse_whitespace(text))
        if not chunks:
            return None
        scored = []
        for start in range(0, len(chunks), 32):
            batch = chunks[start : start + 32]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to("cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            scored.extend(
                Chunk(logit=float(v), words=len(c.split()))
                for v, c in zip(np.atleast_1d(logits), batch, strict=True)
            )
        result = aggregate(scored, cal)
        # Document score is the strongest run, which is what a reader would see.
        return max((r.score for r in result.runs), default=min(result.chunk_p, default=0.0))

    human, ai = [], []
    by_domain: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    by_attack: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for row in rows:
        score = document_score(row["text"])
        if score is None:
            continue
        (ai if row["is_ai"] else human).append(score)
        by_domain[row["domain"]].append((score, row["is_ai"]))
        by_attack[row["attack"]].append((score, row["is_ai"]))

    human_arr, ai_arr = np.array(human), np.array(ai)
    report: dict[str, Any] = {
        "run_id": run_id,
        "documents": len(rows),
        "n_human": len(human),
        "n_ai": len(ai),
        "auroc": auroc(human_arr, ai_arr) if human and ai else None,
        "fpr_at_shipped_t_on": fpr_at_threshold(human_arr, cal.t_on),
        "recall_at_shipped_t_on": recall_at_threshold(ai_arr, cal.t_on),
        "calibration_version": cal.version,
        "note": (
            "RAID's generators skew to 2023-era models and its domains contain no press "
            "releases, marketing copy or product pages -- none of the genres scope.md 2 "
            "names as the dominant false-positive risk. It reads better than production."
        ),
    }
    for label, grouped in (("by_domain", by_domain), ("by_attack", by_attack)):
        report[label] = {
            key: {
                "n": len(values),
                "auroc": auroc(
                    np.array([s for s, is_ai in values if not is_ai]),
                    np.array([s for s, is_ai in values if is_ai]),
                ),
            }
            for key, values in sorted(grouped.items())
            if any(is_ai for _, is_ai in values) and any(not is_ai for _, is_ai in values)
        }

    dest = root / "artifacts" / "runs" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "raid_eval.json").write_text(json.dumps(report, indent=2, default=str), "utf-8")
    volume.commit()
    return report


@app.function(
    image=export_image,
    cpu=16.0,
    memory=65536,
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=4 * 3600,
)
def compare_artifacts(
    run_id: str, version: str, split: str = "test", limit: int = 4000, checkpoint: str = "best"
) -> dict[str, Any]:
    """Score identical rows through the fp32 and int8 graphs and compare.

    This is the gate that should have run before the bundle was assembled. Every other
    check in the export path is structural -- node counts, file size, opsets -- and none
    of them can see a quantization that ran correctly and still destroyed the model.
    Only scoring both artifacts on the same rows can, and running it on one split holds
    the split fixed so the only variable left is the arithmetic.
    """
    import json
    import random
    import time
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    from slopmarker.data.dataset import load_windows
    from slopmarker.data.normalize import collapse_whitespace
    from slopmarker.export.gates import compare_scores

    volume.reload()
    root = Path(DATA_ROOT)
    ckpt = root / "artifacts" / "runs" / run_id / checkpoint
    export_dir = root / "artifacts" / "export" / run_id

    rows = load_windows(root, version, split)
    rng = random.Random(0)
    if len(rows) > limit:
        rows = rng.sample(rows, limit)
    texts = [collapse_whitespace(r.text) for r in rows]
    print(f"comparing on {len(rows)} {split} windows")

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 16

    def run(path: Path, label: str) -> np.ndarray:
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        out: list[float] = []
        started = time.time()
        for start in range(0, len(texts), 64):
            if start and start % 1600 == 0:
                rate = start / max(1e-6, time.time() - started)
                print(f"  {label}: {start}/{len(texts)} at {rate:.0f}/s")
            enc = tokenizer(
                texts[start : start + 64],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            logits = session.run(
                None,
                {
                    "input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64),
                },
            )[0]
            out.extend(float(v) for v in np.asarray(logits).reshape(-1))
        return np.array(out, dtype=np.float64)

    scores = {
        "fp32": run(export_dir / "model.onnx", "fp32"),
        "int8": run(export_dir / "model.int8.onnx", "int8"),
    }

    def as_rows(values: np.ndarray) -> list[dict[str, Any]]:
        return [
            {"genre": row.genre, "ai_fraction": row.ai_fraction, "logit": float(value)}
            for row, value in zip(rows, values, strict=True)
        ]

    report = {
        "run_id": run_id,
        "split": split,
        **compare_scores(as_rows(scores["fp32"]), as_rows(scores["int8"])),
    }
    (export_dir / f"compare.{split}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    volume.commit()
    return report


@app.function(
    image=export_image,
    cpu=16.0,
    memory=65536,
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=4 * 3600,
)
def quantization_sweep(
    run_id: str, version: str, split: str = "calibration", limit: int = 1500
) -> dict[str, Any]:
    """Score every quantization recipe against fp32 on identical rows.

    scope.md 5.4 picks its op list to hit a file size. Size is the easy half: the
    question that decides the recipe is how much separation each variant costs, and
    that can only be measured. The variants differ along the two axes that matter --
    whether the 50k-row embedding table is quantized at all, and whether weights get
    one scale per tensor or one per channel.
    """
    import json
    import random
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    from slopmarker.data.dataset import load_windows
    from slopmarker.data.normalize import collapse_whitespace
    from slopmarker.export.gates import compare_scores
    from slopmarker.export.onnx_export import (
        quantize_int8,
        quantize_mixed,
        quantize_weight_only,
    )

    volume.reload()
    root = Path(DATA_ROOT)
    ckpt = root / "artifacts" / "runs" / run_id / "best"
    export_dir = root / "artifacts" / "export" / run_id
    fp32 = export_dir / "model.onnx"

    rows = load_windows(root, version, split)
    rng = random.Random(0)
    if len(rows) > limit:
        rows = rng.sample(rows, limit)
    texts = [collapse_whitespace(r.text) for r in rows]
    print(f"sweeping on {len(rows)} {split} windows")

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 16

    def run(path: Path) -> list[dict[str, Any]]:
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        out: list[dict[str, Any]] = []
        for start in range(0, len(texts), 64):
            enc = tokenizer(
                texts[start : start + 64],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            logits = session.run(
                None,
                {
                    "input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64),
                },
            )[0]
            for row, value in zip(
                rows[start : start + 64], np.asarray(logits).reshape(-1), strict=True
            ):
                out.append(
                    {"genre": row.genre, "ai_fraction": row.ai_fraction, "logit": float(value)}
                )
        return out

    baseline = run(fp32)

    # Two families, and the difference between them is the experiment.
    #
    # quantize_dynamic compresses weights AND rescales activations per tensor at
    # runtime. Every recipe built that way lost 4-6 points of pAUC with individual
    # logits moving by up to 7, which weight rounding does not do.
    #
    # MatMulNBits quantizes only the constant weight operands, block-wise, and leaves
    # activations in float. If pAUC returns, the activations were the cause -- and the
    # artifact is the one worth shipping anyway, since ORT Web's WASM backend supports
    # the op and block scales handle outlier weights better than one scale per tensor.
    dynamic: dict[str, dict[str, Any]] = {
        "dyn_weights_gather": {"quantize_embeddings": True, "const_b_only": True},
        "dyn_weights_gather_per_channel": {
            "quantize_embeddings": True,
            "per_channel": True,
            "const_b_only": True,
        },
        # reduce_range holds weights to 7 bits, leaving accumulator headroom. Normally
        # an old-hardware workaround, but it also stops products saturating when the
        # activations carry outliers, so it is worth a row.
        "dyn_weights_gather_reduce_range": {
            "quantize_embeddings": True,
            "per_channel": True,
            "const_b_only": True,
            "reduce_range": True,
        },
        # The recipe the first bundle shipped, kept so the comparison stays on record.
        "dyn_all_matmul_gather": {"quantize_embeddings": True, "const_b_only": False},
    }
    weight_only: dict[str, dict[str, Any]] = {
        "wo_int8": {"bits": 8, "quantize_embeddings": False},
        "wo_int4": {"bits": 4, "quantize_embeddings": True},
    }
    # Encoder and embedding table at different widths. int8 throughout the encoder
    # measured lossless but leaves the table in fp32 at 270MB, because MatMulNBits
    # quantizes Gather only at 4 bits. These rows ask what the table costs.
    mixed: dict[str, dict[str, Any]] = {
        "mix_e8_emb4": {"encoder_bits": 8, "embedding_bits": 4},
        "mix_e8_emb8": {"encoder_bits": 8, "embedding_bits": 8},
    }

    results: dict[str, Any] = {}

    def record(name: str, build: Any) -> None:
        """One failing variant must not cost the whole sweep, as fp16 did."""
        path = export_dir / f"model.{name}.onnx"
        try:
            quantization = build(path)
            comparison = compare_scores(baseline, run(path))
        except Exception as error:  # a variant may be unsupported by this ORT build
            results[name] = {"error": f"{type(error).__name__}: {error}"}
            print(f"{name}: FAILED {type(error).__name__}: {error}")
            return
        results[name] = {"quantization": quantization, "comparison": comparison}
        print(
            f"{name}: {quantization['int8_mb']:.0f}MB"
            f" nodes {quantization['matmul_integer_nodes']}"
            f" auroc {comparison['fp32']['auroc']:.4f} -> {comparison['int8']['auroc']:.4f}"
            f" pauc {comparison['fp32']['pauc_2pct']:.4f} -> {comparison['int8']['pauc_2pct']:.4f}"
            f" spearman {comparison['spearman']:.4f}"
        )
        path.unlink(missing_ok=True)  # a 150-600MB file per variant fills the volume

    for name, kwargs in weight_only.items():
        record(name, lambda p, k=kwargs: quantize_weight_only(fp32, p, **k))
    for name, kwargs in mixed.items():
        record(name, lambda p, k=kwargs: quantize_mixed(fp32, p, **k))
    for name, kwargs in dynamic.items():
        record(name, lambda p, k=kwargs: quantize_int8(fp32, p, **k))

    report = {"run_id": run_id, "split": split, "n": len(rows), "variants": results}
    (export_dir / "quantization_sweep.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    volume.commit()
    return report


@app.function(
    image=export_image,
    cpu=16.0,
    memory=65536,
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=2 * 3600,
)
def determinism_check(run_id: str, version: str, limit: int = 400) -> dict[str, Any]:
    """Score the same rows twice per artifact, and report the CPU it ran on.

    Two sweeps over identical inputs returned AUROC 0.978 and 0.550 for the same
    dynamic-quantization recipe, from files identical in size and node counts. Either
    the int8 kernels are nondeterministic or they differ by host instruction set. For a
    model that ships to thousands of browsers, an artifact whose scores depend on the
    machine is unusable whatever its pAUC, so this measures it rather than assuming.

    Run it more than once to compare across containers; within a container, two passes
    over the same rows must agree bitwise.
    """
    import hashlib
    import json
    import platform
    import subprocess
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    from slopmarker.data.dataset import load_windows
    from slopmarker.data.normalize import collapse_whitespace
    from slopmarker.export.onnx_export import quantize_int8, quantize_mixed

    volume.reload()
    root = Path(DATA_ROOT)
    ckpt = root / "artifacts" / "runs" / run_id / "best"
    export_dir = root / "artifacts" / "export" / run_id
    fp32 = export_dir / "model.onnx"

    rows = load_windows(root, version, "test")[:limit]
    texts = [collapse_whitespace(r.text) for r in rows]
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    try:
        flags = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=30, check=False
        ).stdout
        isa = sorted(
            f for f in ("avx512f", "avx512_vnni", "avx2", "avx_vnni") if f in flags.lower()
        )
        model_name = next(
            (ln.split(":", 1)[1].strip() for ln in flags.splitlines() if "Model name" in ln), "?"
        )
    except Exception:  # lscpu is not guaranteed present
        isa, model_name = [], "?"

    def score(path: Path) -> np.ndarray:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        out: list[float] = []
        for start in range(0, len(texts), 32):
            enc = tokenizer(
                texts[start : start + 32],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            logits = session.run(
                None,
                {
                    "input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64),
                },
            )[0]
            out.extend(float(v) for v in np.asarray(logits).reshape(-1))
        return np.array(out, dtype=np.float64)

    recipes = {
        "mix_e8_emb4": lambda p: quantize_mixed(fp32, p, encoder_bits=8, embedding_bits=4),
        "dyn_weights_gather_per_channel": lambda p: quantize_int8(
            fp32, p, quantize_embeddings=True, per_channel=True, const_b_only=True
        ),
    }
    results: dict[str, Any] = {"cpu": model_name, "isa": isa, "node": platform.node()}
    for name, build in recipes.items():
        path = export_dir / f"determinism.{name}.onnx"
        build(path)
        first, second = score(path), score(path)
        results[name] = {
            "within_container_max_diff": float(np.max(np.abs(first - second))),
            "logits_sha256": hashlib.sha256(first.tobytes()).hexdigest()[:16],
            "mean_logit": float(first.mean()),
            "first_five": [round(v, 6) for v in first[:5]],
        }
        print(f"{name}: {json.dumps(results[name])}")
        path.unlink(missing_ok=True)
    print(f"cpu={model_name} isa={isa}")
    return results


@app.function(
    image=export_image,
    cpu=16.0,
    memory=65536,
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=6 * 3600,
)
def document_eval(
    run_id: str,
    source: str = "test_human",
    limit: int = 1500,
    version: str = "v3",
) -> dict[str, Any]:
    """Score whole documents through the full scope.md 8 pipeline on the shipped artifact.

    Every number reported so far is chunk-level, but a reader never sees a chunk. They
    see a highlighted *run*, produced after the length penalty, run pooling, the
    document prior and the 150-word minimum. A document is a false positive when a run
    survives all of that, which is a stricter question than whether some 200-word
    window crossed a threshold.

    Three sources, in increasing distance from the training distribution:

      test_human   held-out human documents from our own corpus. In-distribution, so
                   this is a floor rather than an estimate of production.
      unseen_host  held-out human documents whose registered domain appears nowhere in
                   train. held_out_domains_per_genre is configured and enforced
                   nowhere, so this reconstructs the reservation after the fact; it is
                   the only measurement of host-template memorisation available.
      raid         the external benchmark. Its human side comes from eight sources this
                   corpus deliberately never harvested, so it is genuinely
                   out-of-distribution human text, and it carries adversarial attacks
                   that our corpus contains none of.
    """
    import json
    import random
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    from slopmarker.corpus.build import load_documents
    from slopmarker.data.dataset import load_windows
    from slopmarker.data.normalize import strip_markdown
    from slopmarker.eval.calibration import Calibration
    from slopmarker.eval.documents import score_document
    from slopmarker.eval.metrics import auroc, clopper_pearson_upper

    volume.reload()
    root = Path(DATA_ROOT)
    ckpt = root / "artifacts" / "runs" / run_id / "best"
    export_dir = root / "artifacts" / "export" / run_id
    cal = Calibration.load(export_dir / "calibration.json")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 16
    session = ort.InferenceSession(
        str(export_dir / "model.int8.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    rng = random.Random(0)

    def score(text: str) -> dict[str, Any] | None:
        return score_document(text, lambda t: _score_chunks(t, session, tokenizer), cal)

    documents: list[dict[str, Any]] = []
    if source == "raid":
        from datasets import load_dataset

        # The train split is labelled and public; test is leaderboard-held-out.
        #
        # Rows are ordered by domain, so taking a prefix of the stream samples one
        # domain and calls it RAID. The first run of this eval did exactly that and
        # returned 1,493 rows all from `abstracts`. Quotas per (domain, class, attacked)
        # cell instead, with shard shuffling so the scan does not start at the same
        # domain every time, and the scan cap is reported rather than assumed adequate.
        stream = load_dataset("liamdugan/raid", split="train", streaming=True)
        stream = stream.shuffle(seed=0, buffer_size=20_000)
        per_cell = max(8, limit // 32)
        cells: dict[tuple[str, bool, bool], list[dict[str, Any]]] = defaultdict(list)
        scanned = 0
        for row in stream:
            scanned += 1
            if scanned > limit * 400:
                break
            is_ai = row["model"] != "human"
            attack = row.get("attack") or "none"
            key = (row.get("domain", "unknown"), is_ai, attack != "none")
            if len(cells[key]) >= per_cell:
                continue
            cells[key].append(
                {
                    "text": row["generation"],
                    "is_ai": is_ai,
                    "domain": row.get("domain", "unknown"),
                    "attack": attack,
                }
            )
            if len(cells) >= 32 and all(len(v) >= per_cell for v in cells.values()):
                break
        pool = [row for rows_ in cells.values() for row in rows_]
        rng.shuffle(pool)
        documents = pool[:limit]
        print(
            f"RAID scan: {scanned} rows, {len(cells)} cells, "
            f"domains={sorted({d for d, _, _ in cells})}"
        )
    else:
        by_id = {d.doc_id: d for d in load_documents(root, "interim")}
        train_hosts = set()
        for window in load_windows(root, version, "train"):
            doc = by_id.get(window.doc_id)
            if doc is not None and doc.host:
                train_hosts.add(doc.host)

        pool = []
        for doc_id in {w.doc_id for w in load_windows(root, version, "test")}:
            doc = by_id.get(doc_id)
            if doc is None or doc.doc_class != "human":
                continue
            if source == "unseen_host" and (not doc.host or doc.host in train_hosts):
                continue
            pool.append(
                {
                    # build() strips markdown from both classes before windowing, and
                    # production reads rendered DOM text, so match that here.
                    "text": strip_markdown(doc.text),
                    "is_ai": False,
                    "domain": doc.genre,
                    "attack": doc.hard_negative_kind or "none",
                }
            )
        rng.shuffle(pool)
        documents = pool[:limit]

    print(f"scoring {len(documents)} documents from {source}")
    results: list[dict[str, Any]] = []
    for index, row in enumerate(documents):
        if index and index % 200 == 0:
            print(f"  {index}/{len(documents)}")
        scored = score(row["text"])
        if scored is None:
            continue
        results.append(
            {**scored, "is_ai": row["is_ai"], "domain": row["domain"], "attack": row["attack"]}
        )

    human_rows = [r for r in results if not r["is_ai"]]
    ai_rows = [r for r in results if r["is_ai"]]

    def rate(rows: list[dict[str, Any]]) -> float:
        return sum(r["flagged"] for r in rows) / len(rows) if rows else float("nan")

    report: dict[str, Any] = {
        "run_id": run_id,
        "source": source,
        "calibration_version": cal.version,
        "t_on": cal.t_on,
        "documents_scored": len(results),
        "documents_skipped_too_short": len(documents) - len(results),
        "n_human": len(human_rows),
        "n_ai": len(ai_rows),
        "doc_level_fpr": rate(human_rows),
        "doc_level_recall": rate(ai_rows),
    }
    if human_rows:
        flagged = sum(r["flagged"] for r in human_rows)
        report["doc_level_fpr_upper"] = clopper_pearson_upper(flagged, len(human_rows))
    if human_rows and ai_rows:
        report["doc_auroc"] = auroc(
            np.array([r["max_chunk_p"] for r in human_rows]),
            np.array([r["max_chunk_p"] for r in ai_rows]),
        )
        report["doc_auroc_run_score"] = auroc(
            np.array([r["max_run_score"] for r in human_rows]),
            np.array([r["max_run_score"] for r in ai_rows]),
        )
        report["fraction_with_no_run"] = sum(1 for r in results if r["max_run_score"] == 0.0) / len(
            results
        )
        report["median_words"] = float(np.median([r["words"] for r in results]))
        report["median_chunks"] = float(np.median([r["n_chunks"] for r in results]))

    for label, key in (("by_domain", "domain"), ("by_attack", "attack")):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in results:
            grouped[str(row[key])].append(row)
        report[label] = {
            name: {
                "n": len(rows),
                "n_human": sum(1 for r in rows if not r["is_ai"]),
                "fpr": rate([r for r in rows if not r["is_ai"]]),
                "recall": rate([r for r in rows if r["is_ai"]]),
            }
            for name, rows in sorted(grouped.items())
        }

    (export_dir / f"doc_eval.{source}.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    volume.commit()
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("by_")}, indent=2))
    return report
