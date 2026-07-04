"""Pull serving models FROM the W&B Model Registry (not hardcoded in the repo).

Two collections (both registered from HW2/HW3, versioned in the registry):
  * distilbert-aug-onnx : ONNX-INT8, the default /predict backend (fast)
  * distilbert-aug      : PyTorch checkpoint, used by /explain (and torch fallback)

All names/aliases are env-configurable so the same image can serve base/aug or a new
version without code changes. Requires WANDB_API_KEY (see start.sh).
"""
from __future__ import annotations

import os
from pathlib import Path

import wandb

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "phishing-scorer")

ONNX_COLLECTION = os.environ.get("ONNX_COLLECTION", "distilbert-aug-onnx")
ONNX_ALIAS = os.environ.get("ONNX_ALIAS", "int8")
ONNX_DIR = os.environ.get("ONNX_DIR", "models/onnx")

MODEL_COLLECTION = os.environ.get("MODEL_COLLECTION", "distilbert-aug")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "aug")
MODEL_DIR = os.environ.get("MODEL_DIR", "models/serve")


def _pull(collection: str, alias: str, target: str) -> str:
    ref = f"wandb-registry-model/{collection}:{alias}"
    print(f"downloading {ref} -> {target}")
    run = wandb.init(project=WANDB_PROJECT, job_type="serve",
                     settings=wandb.Settings(silent=True))
    art = run.use_artifact(ref, type="model")
    path = art.download(root=target)
    run.finish()
    Path(target).mkdir(parents=True, exist_ok=True)
    print(f"ready at: {path}")
    return path


def download_onnx(target: str = ONNX_DIR) -> str:
    return _pull(ONNX_COLLECTION, ONNX_ALIAS, target)


def download_torch(target: str = MODEL_DIR) -> str:
    return _pull(MODEL_COLLECTION, MODEL_ALIAS, target)


if __name__ == "__main__":
    # default: fetch the ONNX-INT8 serving model
    download_onnx()
