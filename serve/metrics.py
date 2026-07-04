"""HW4 — Prometheus metrics for the phishing-obviousness scorer.

Ops metrics (HTTP request count + latency histograms per endpoint) come for free from
`prometheus-fastapi-instrumentator`, wired in main.py. Here we add the DOMAIN metrics that
make the dashboard meaningful for Whalen: how many predictions/min, split by verdict, and
the live distribution of the calibrated score / obviousness. Everything is exposed on
GET /metrics and scraped by Prometheus (see monitoring/prometheus/prometheus.yml).
"""
from __future__ import annotations

import os

from prometheus_client import Counter, Histogram

BACKEND = os.environ.get("BACKEND", "onnx").lower()

# predictions/min (the required "кількість передбачень за хвилину"), split by verdict
PREDICTIONS = Counter(
    "phishing_predictions_total",
    "Number of scored emails, by predicted verdict / backend / endpoint.",
    ["verdict", "backend", "endpoint"],
)

# live distribution of the calibrated P(phish) — prediction-drift signal at a glance
P_PHISH = Histogram(
    "phishing_p_phish",
    "Calibrated P(phish) of scored emails.",
    buckets=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

# 1-10 obviousness scale (product-health metric: are authored lures trending convincing?)
OBVIOUSNESS = Histogram(
    "phishing_obviousness",
    "Obviousness score (1=convincing .. 10=obvious) of scored emails.",
    buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
)

# model inference time (excludes HTTP/serialization overhead)
INFER_LATENCY = Histogram(
    "phishing_inference_latency_seconds",
    "Model inference latency per prediction.",
    ["endpoint", "backend"],
    buckets=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)


def record_prediction(result: dict, endpoint: str, latency_s: float) -> None:
    """Feed one scoring result into the domain metrics (call once per scored email)."""
    PREDICTIONS.labels(result.get("label", "?"), BACKEND, endpoint).inc()
    P_PHISH.observe(float(result.get("p_phish", 0.0)))
    OBVIOUSNESS.observe(float(result.get("obviousness_1_10", 1.0)))
    INFER_LATENCY.labels(endpoint, BACKEND).observe(max(latency_s, 0.0))
