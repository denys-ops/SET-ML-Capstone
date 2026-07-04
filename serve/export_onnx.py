"""Offline: export the registry PyTorch model to ONNX-INT8 and register it back.

Flow (variant b — the optimized artifact is versioned in the registry too):
  1. pull  distilbert-aug (PyTorch) from the W&B registry
  2. export FP32 ONNX (torch.onnx.export)
  3. quantize INT8 (onnxruntime dynamic quantization)
  4. parity-check INT8 vs PyTorch on sample emails (max |Δ p_phish|)
  5. log a new artifact + link to registry collection `distilbert-aug-onnx:int8`

Runs anywhere with torch + onnxruntime + wandb. No local setup needed — run it inside
the serving image (which already has all deps):

  docker build -t phishing-scorer-serve .
  docker run --rm --env-file ../.env --entrypoint python phishing-scorer-serve export_onnx.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from download_model import ONNX_COLLECTION, download_torch
from scoring import MAX_LENGTH, format_results, to_texts

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "phishing-scorer")
EXPORT_DIR = Path(os.environ.get("EXPORT_DIR", "models/onnx-export"))
OPSET = 14
PARITY_TOL = 0.05  # max acceptable |Δ p_phish| between PyTorch and INT8


class _Wrap(torch.nn.Module):
    """Thin wrapper so ONNX sees a clean (input_ids, attention_mask) -> logits graph."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def _load_samples() -> list[dict]:
    p = Path(__file__).with_name("sample_emails.json")
    return json.loads(p.read_text())


def export() -> str:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    torch_dir = download_torch()  # pull PyTorch aug from registry
    tok = AutoTokenizer.from_pretrained(torch_dir)
    model = AutoModelForSequenceClassification.from_pretrained(torch_dir).eval()

    fp32_path = EXPORT_DIR / "model.onnx"
    int8_path = EXPORT_DIR / "model.int8.onnx"

    dummy = tok("hello world", return_tensors="pt", padding="max_length",
                max_length=MAX_LENGTH, truncation=True)
    print("exporting FP32 ONNX ...")
    torch.onnx.export(
        _Wrap(model), (dummy["input_ids"], dummy["attention_mask"]), str(fp32_path),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=OPSET, do_constant_folding=True)

    print("quantizing INT8 ...")
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    tok.save_pretrained(EXPORT_DIR)  # tokenizer travels with the ONNX artifact

    # ---- parity check: PyTorch vs INT8 on sample emails ----
    import onnxruntime as ort

    samples = _load_samples()
    texts = to_texts(samples)
    enc = tok(texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
    with torch.no_grad():
        torch_logits = model(**enc).logits.numpy()
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(None, {"input_ids": enc["input_ids"].numpy().astype(np.int64),
                                  "attention_mask": enc["attention_mask"].numpy().astype(np.int64)})[0]
    p_torch = np.array([r["p_phish"] for r in format_results(torch_logits)])
    p_onnx = np.array([r["p_phish"] for r in format_results(onnx_logits)])
    max_delta = float(np.abs(p_torch - p_onnx).max())
    print(f"parity: max |Δ p_phish| = {max_delta:.4f} (tol {PARITY_TOL})")
    for s, pt, po in zip(samples, p_torch, p_onnx):
        print(f"  torch={pt:.3f}  onnx={po:.3f}  | {s.get('subject','')!r}")
    if max_delta > PARITY_TOL:
        raise SystemExit(f"parity check FAILED: {max_delta:.4f} > {PARITY_TOL}")

    fp32_mb = fp32_path.stat().st_size / 1e6
    int8_mb = int8_path.stat().st_size / 1e6
    print(f"size: FP32 {fp32_mb:.1f} MB -> INT8 {int8_mb:.1f} MB (x{fp32_mb/int8_mb:.1f} smaller)")

    # ---- register INT8 artifact in the W&B registry ----
    run = wandb.init(project=WANDB_PROJECT, job_type="export")
    art = wandb.Artifact(ONNX_COLLECTION, type="model",
                         metadata={"source": "distilbert-aug", "quant": "int8", "opset": OPSET,
                                   "max_length": MAX_LENGTH, "parity_max_delta": max_delta,
                                   "size_mb_int8": round(int8_mb, 1)})
    art.add_file(str(int8_path), name="model.int8.onnx")
    for f in EXPORT_DIR.glob("*"):
        if f.name not in ("model.onnx", "model.int8.onnx"):
            art.add_file(str(f))  # tokenizer files
    run.log_artifact(art, aliases=["int8", "latest"])
    run.link_artifact(art, target_path=f"wandb-registry-model/{ONNX_COLLECTION}",
                      aliases=["int8", "latest"])
    run.finish()
    print(f"registered -> wandb-registry-model/{ONNX_COLLECTION}:int8")
    return str(int8_path)


if __name__ == "__main__":
    export()
