"""PyTorch scoring backend (fallback for /predict, and reused by /explain).

The default serving path is ONNX-INT8 (see onnx_infer.py); this backend is used when
BACKEND=torch, or as the model behind /explain (Integrated Gradients needs gradients,
which ONNX can't provide).
"""
from __future__ import annotations

import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scoring import MAX_LENGTH, format_results, to_texts


class TorchScorer:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def _logits(self, texts: list[str]):
        enc = self.tokenizer(texts, truncation=True, max_length=MAX_LENGTH,
                             padding=True, return_tensors="pt").to(self.device)
        return self.model(**enc).logits.cpu().numpy()

    def score(self, html: str, subject: str = "") -> dict:
        return self.score_batch([{"html": html, "subject": subject}])[0]

    def score_batch(self, emails: list[dict]) -> list[dict]:
        return format_results(self._logits(to_texts(emails)))


if __name__ == "__main__":
    import json
    import sys

    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_DIR", "models/serve")
    s = TorchScorer(model_dir)
    demo = '<p>Your account is locked. Verify now at http://acme-login.example.com/verify</p>'
    print(json.dumps(s.score(demo, subject="Urgent: account suspended"), indent=2))
