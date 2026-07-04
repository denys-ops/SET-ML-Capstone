"""HW4 — append each prediction to a JSONL log for offline drift analysis.

The Evidently drift job (monitoring/evidently/) reads this file as the "current" dataset
and compares it against a reference built from the HW1 labelled set. We log a few cheap
input features (lengths, link count) plus the model output — enough for data- and
prediction-drift without storing raw email content.

Path: $LOG_DIR/predictions.jsonl  (LOG_DIR defaults to /app/logs, a mounted volume).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

LOG_DIR = Path(os.environ.get("LOG_DIR", "/app/logs"))
LOG_FILE = LOG_DIR / "predictions.jsonl"
_LOCK = threading.Lock()


def _n_links(html: str) -> int:
    """Cheap link-count feature (works for both HTML hrefs and bare URLs)."""
    low = html.lower()
    return low.count("http://") + low.count("https://")


def log_prediction(html: str, subject: str, result: dict,
                   endpoint: str, latency_ms: float) -> None:
    """Append one prediction record; never raise (logging must not break serving)."""
    rec = {
        "ts": time.time(),
        "backend": os.environ.get("BACKEND", "onnx").lower(),
        "endpoint": endpoint,
        "text_len": len(html or ""),
        "subject_len": len(subject or ""),
        "n_links": _n_links(html or ""),
        "p_phish": result.get("p_phish"),
        "raw_p": result.get("raw_p"),
        "obviousness": result.get("obviousness_1_10"),
        "label": result.get("label"),
        "latency_ms": round(latency_ms, 2),
    }
    try:
        with _LOCK:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with LOG_FILE.open("a") as f:
                f.write(json.dumps(rec) + "\n")
    except Exception:
        pass  # best-effort; a full disk must not take the API down
