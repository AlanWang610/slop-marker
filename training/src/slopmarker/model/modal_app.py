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
