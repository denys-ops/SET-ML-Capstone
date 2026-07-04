"""ONNX-INT8 scoring backend — the default, low-latency path for /predict.

Runs the quantized DistilBERT via ONNX Runtime (no PyTorch at inference time), which
is ~3x faster / ~4x smaller than the FP32 PyTorch model — this is what makes reactive
per-keystroke scoring in the Whalen editor feasible.
"""
from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from scoring import MAX_LENGTH, format_results, to_texts


class OnnxScorer:
    def __init__(self, model_dir: str, onnx_file: str = "model.int8.onnx"):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        so = ort.SessionOptions()
        threads = int(os.environ.get("ORT_THREADS", "0"))
        if threads > 0:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            os.path.join(model_dir, onnx_file), sess_options=so,
            providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def _logits(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(texts, truncation=True, max_length=MAX_LENGTH,
                             padding=True, return_tensors="np")
        feed = {"input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64)}
        feed = {k: v for k, v in feed.items() if k in self.input_names}
        return self.session.run(None, feed)[0]

    def score(self, html: str, subject: str = "") -> dict:
        return self.score_batch([{"html": html, "subject": subject}])[0]

    def score_batch(self, emails: list[dict]) -> list[dict]:
        return format_results(self._logits(to_texts(emails)))


if __name__ == "__main__":
    import json
    import sys

    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ONNX_DIR", "models/onnx")
    s = OnnxScorer(model_dir)
    demo = '<p>Your account is locked. Verify now at http://acme-login.example.com/verify</p>'
    print(json.dumps(s.score(demo, subject="Urgent: account suspended"), indent=2))
