"""Shared scoring post-processing — identical for the ONNX and PyTorch backends.

Both backends produce raw logits (N, 2); everything after that (preprocess contract,
temperature calibration, obviousness scale, label) lives here so the two paths can
never drift apart.
"""
from __future__ import annotations

import os

import numpy as np

from preprocess import html_to_scoring_text

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "256"))
CALIB_T = float(os.environ.get("CALIB_T", "1.3301801681518555"))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))


def to_texts(emails: list[dict]) -> list[str]:
    """Raw {html, subject} -> normalised scoring text (the train/serve contract)."""
    return [html_to_scoring_text(e.get("html", ""), e.get("subject", "")) for e in emails]


def _softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def format_results(logits) -> list[dict]:
    """logits (N, 2) -> list of {label, p_phish (calibrated), raw_p, obviousness_1_10}."""
    logits = np.asarray(logits, dtype=np.float64)
    raw = _softmax(logits)[:, 1]
    cal = _softmax(logits / CALIB_T)[:, 1]
    out = []
    for i in range(len(logits)):
        p = float(cal[i])
        out.append({
            "label": "phishing" if p >= THRESHOLD else "safe",
            "p_phish": round(p, 4),
            "raw_p": round(float(raw[i]), 4),
            "obviousness_1_10": round(1 + 9 * p, 1),
        })
    return out
