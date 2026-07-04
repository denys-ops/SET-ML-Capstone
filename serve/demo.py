"""Video demo — narrated request -> response for the phishing-obviousness scorer.

For each sample email it prints, in plain language:
  * what was SENT (subject + body preview),
  * the VERDICT + obviousness score + how long /predict took (ONNX-INT8, ms),
  * the KEY ELEMENTS that drove the score (top tokens from /explain).

Run against the running service (docker compose up -d serve):
  python demo.py
  API_URL=http://localhost:8088 python demo.py        # default port
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

API_URL = os.environ.get("API_URL", "http://localhost:8088").rstrip("/")
SAMPLES = Path(__file__).with_name("sample_emails.json")

# ANSI colours (nice on video; harmless if the terminal ignores them)
R, G, Y, DIM, B, X = "\033[91m", "\033[92m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"
LINE = "─" * 66
DLINE = "═" * 66


def post(path: str, payload: dict, timeout: float = 120) -> tuple[dict, float]:
    req = urllib.request.Request(
        f"{API_URL}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body, (time.perf_counter() - t) * 1000


def preview(text: str, n: int = 90) -> str:
    text = " ".join(text.replace("<p>", " ").replace("</p>", " ").split())
    return text if len(text) <= n else text[:n] + "…"


_PUNCT = set(".,:;'\"!?-()[]{}<>/|`~")


def clean_drivers(tokens: list[dict], phish: bool, k: int = 5) -> str:
    """Top meaningful tokens pushing toward the predicted class (drop punctuation/subwords/dups)."""
    ranked = sorted(tokens, key=lambda t: t["attr"], reverse=phish)
    out, seen = [], set()
    for t in ranked:
        tok = t["token"]
        if tok.startswith("##") or tok in _PUNCT or (len(tok) < 2 and tok != "$"):
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) == k:
            break
    return ", ".join(out) if out else "—"


def main() -> None:
    emails = json.loads(SAMPLES.read_text())

    # backend + warm-up (so the timed numbers exclude first-call model loading)
    try:
        with urllib.request.urlopen(f"{API_URL}/health", timeout=10) as r:
            backend = json.loads(r.read()).get("backend", "?")
    except Exception:
        backend = "?"
    print(f"\n{DLINE}\n {B}ДЕМО — Phishing-Obviousness Scorer{X}   (бекенд: {B}{backend}{X})\n{DLINE}")
    print(f"{DIM} прогрів моделей…{X}", flush=True)
    for _ in range(6):  # warm ONNX session so timed numbers show steady-state latency
        post("/predict", {"html": "warmup", "subject": ""})
    post("/explain", {"html": "warmup", "subject": "", "top_k": 1})  # loads PyTorch once

    latencies, n_phish = [], 0
    for i, e in enumerate(emails, 1):
        subject, html = e.get("subject", ""), e.get("html", "")
        pred, ms = post("/predict", {"html": html, "subject": subject})
        expl, _ = post("/explain", {"html": html, "subject": subject, "top_k": 6})
        latencies.append(ms)

        phish = pred["label"] == "phishing"
        n_phish += phish
        mark = f"{R}🔴 ФІШИНГ{X}" if phish else f"{G}🟢 БЕЗПЕЧНО{X}"
        drivers = clean_drivers(expl.get("tokens", []), phish)

        print(f"\n{LINE}")
        print(f"{B}[{i}/{len(emails)}] 📧 Надіслано{X}")
        print(f"   Тема:  {subject}")
        print(f"   Тіло:  {DIM}{preview(html)}{X}")
        print(f"   {Y}⏱  відповідь /predict за {ms:.0f} ms{X}")
        print(f"   {mark}   очевидність {B}{pred['obviousness_1_10']}/10{X}   "
              f"(p_phish={pred['p_phish']:.3f})")
        arrow = "тягнуть → ФІШИНГ" if phish else "тягнуть → БЕЗПЕЧНО"
        print(f"   🔎 ключові елементи ({arrow}): {B}{drivers}{X}")

    print(f"\n{DLINE}")
    print(f" Підсумок: {len(emails)} листів → {R}{n_phish} фішинг{X} / "
          f"{G}{len(emails)-n_phish} безпечних{X}   |   "
          f"середня латентність /predict ≈ {B}{sum(latencies)/len(latencies):.0f} ms{X}")
    print(f"{DLINE}\n")


if __name__ == "__main__":
    main()
