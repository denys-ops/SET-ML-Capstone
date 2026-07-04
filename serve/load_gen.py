"""HW4 — load generator to drive the monitoring dashboard for the demo video.

Sends a steady stream of /predict requests (a mix of the sample emails plus light
random variations) so Grafana shows live predictions/min, latency and the verdict /
score distributions under load. stdlib-only — runs with the system python3.

  python load_gen.py                       # ~5 req/s for 120 s against :8088
  RPS=10 DURATION=300 python load_gen.py    # heavier / longer
  API_URL=http://localhost:8088 python load_gen.py
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from pathlib import Path

API_URL = os.environ.get("API_URL", "http://localhost:8088").rstrip("/")
RPS = float(os.environ.get("RPS", "5"))
DURATION = float(os.environ.get("DURATION", "120"))
SAMPLES = Path(__file__).with_name("sample_emails.json")

# extra phrases mixed in so the score/length distributions vary over the run (drift-ish)
_PHISH_BITS = [
    "Verify your account now at http://acme-login.example.com/verify",
    "Urgent: wire the payment before 5pm or the deal falls through.",
    "Your mailbox is full, click https://mail-quota.example.net to keep access.",
    "Final notice: unpaid invoice, pay immediately to avoid suspension.",
]
_SAFE_BITS = [
    "Here are the notes from today's standup, nothing urgent.",
    "Reminder: the team lunch is on Friday at noon.",
    "The Q3 roadmap doc is ready for your review when you have time.",
    "Thanks for the update, looks good on my end.",
]


def post(path: str, payload: dict, timeout: float = 30) -> int:
    req = urllib.request.Request(
        f"{API_URL}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def make_email(base: list[dict], i: int) -> dict:
    """Pick a base sample and splice in a random bit so traffic isn't identical."""
    e = random.choice(base)
    extra = random.choice(_PHISH_BITS if random.random() < 0.5 else _SAFE_BITS)
    return {"subject": e.get("subject", ""),
            "html": f"{e.get('html', '')} <p>{extra}</p>"}


def main() -> None:
    base = json.loads(SAMPLES.read_text())
    interval = 1.0 / RPS if RPS > 0 else 0.0
    print(f"→ load: {RPS} req/s for {DURATION:.0f}s against {API_URL}/predict "
          f"(open Grafana at http://localhost:3000)")
    sent, errors, t_end, i = 0, 0, time.time() + DURATION, 0
    while time.time() < t_end:
        try:
            post("/predict", make_email(base, i))
            sent += 1
        except Exception:
            errors += 1
        i += 1
        if sent % 25 == 0 and sent:
            print(f"  sent={sent} errors={errors}", flush=True)
        if interval:
            time.sleep(interval)
    print(f"✓ done: sent={sent} errors={errors}")


if __name__ == "__main__":
    main()
