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
    "optimum>=2.1", "optimum-onnx==0.1.0", "onnx>=1.17", "onnxruntime==1.29.0"
)


def _with_local(img: modal.Image) -> modal.Image:
    return img.add_local_python_source("slopmarker").add_local_dir(
        "configs", remote_path="/root/configs"
    )


train_image = _with_local(base)
export_image = _with_local(export_base)


@app.function(
    image=train_image,
    gpu="H100",
    volumes=VOLUMES,
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


@app.function(image=train_image, gpu="L40S", volumes=VOLUMES, timeout=4 * 3600)
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


@app.function(image=export_image, cpu=16.0, memory=65536, volumes=VOLUMES, timeout=4 * 3600)
def export_and_calibrate(
    run_id: str, version: str, bundle_version: str, checkpoint: str = "best"
) -> dict[str, Any]:
    """Export to ONNX, quantize, calibrate on the int8 artifact, assemble the bundle.

    The ordering is the point of this function existing as one unit: scope.md 5.5
    requires the threshold to be selected on the artifact that actually ships, not on
    the PyTorch model.
    """
    import json
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    from slopmarker.data.dataset import load_windows
    from slopmarker.data.normalize import collapse_whitespace
    from slopmarker.eval.calibrate import calibrate, verify
    from slopmarker.eval.calibration import AggregateParams, Calibration, t_off_from
    from slopmarker.export.bundle import assemble
    from slopmarker.export.onnx_export import (
        check_parity,
        export_fp32,
        quantize_int8,
        verify_graph,
    )

    volume.reload()
    root = Path(DATA_ROOT)
    ckpt = root / "artifacts" / "runs" / run_id / checkpoint
    export_dir = root / "artifacts" / "export" / run_id
    report: dict[str, Any] = {"run_id": run_id, "version": bundle_version}

    fp32 = export_fp32(ckpt, export_dir)
    calib_rows = load_windows(root, version, "calibration")
    test_rows = load_windows(root, version, "test")
    sample_texts = [collapse_whitespace(r.text) for r in calib_rows[:200]]

    parity = check_parity(ckpt, fp32, sample_texts)
    report["parity"] = parity.to_dict()
    if not parity.passed:
        report["error"] = "fp32 parity failed across shapes"
        volume.commit()
        return report

    int8 = export_dir / "model.int8.onnx"
    report["quantization"] = quantize_int8(fp32, int8)
    report["graph"] = verify_graph(fp32, ckpt)

    # Score the calibration and test splits with the *int8* artifact.
    session = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    def score(rows: list[Any]) -> list[dict[str, Any]]:
        out = []
        for start in range(0, len(rows), 32):
            chunk = rows[start : start + 32]
            enc = tokenizer(
                [collapse_whitespace(r.text) for r in chunk],
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
            for row, logit in zip(chunk, logits, strict=True):
                out.append(
                    {
                        "genre": row.genre,
                        "ai_fraction": row.ai_fraction,
                        "logit": float(np.asarray(logit).reshape(-1)[0]),
                    }
                )
        return out

    calib_scores = score(calib_rows)
    result = calibrate(calib_scores)
    t_off = t_off_from(result.t_on, 0.60)
    checked = verify(score(test_rows), result.temperature, result.t_on)

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

    bundle_dir = root / "artifacts" / "bundles" / bundle_version
    report["bundle"] = assemble(
        int8_model=int8,
        tokenizer_dir=ckpt,
        calibration_path=calibration_path,
        out_dir=bundle_dir,
        provenance=report,
        read_only=False,
    )
    (export_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), "utf-8")
    volume.commit()
    return report


@app.function(
    image=train_image,
    gpu="L40S",
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
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
