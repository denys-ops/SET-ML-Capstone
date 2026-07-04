"""HW3 — demo client. Sends sample_emails.json to the running service and prints
the verdict + obviousness score for each, so the request->response flow is visible.

Usage (service running on :8080):
  python test_client.py                 # uses http://localhost:8080
  API_URL=http://localhost:8080 python test_client.py
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

API_URL = os.environ.get("API_URL", "http://localhost:8080").rstrip("/")
SAMPLES = Path(__file__).with_name("sample_emails.json")


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{API_URL}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def main() -> None:
    emails = json.loads(SAMPLES.read_text())
    print(f"POST {API_URL}/predict  ({len(emails)} sample emails)\n")
    for e in emails:
        out = post("/predict", {"html": e["html"], "subject": e.get("subject", "")})
        print(f"  subject : {e.get('subject','')!r}")
        print(f"  note    : {e.get('_note','')}")
        print(f"  -> {out['label'].upper():8s}  p_phish={out['p_phish']:.3f}  "
              f"obviousness={out['obviousness_1_10']}/10  (raw_p={out['raw_p']:.3f})\n")

    # on-demand explanation for the first (obvious phishing) example
    e = emails[0]
    ex = post("/explain", {"html": e["html"], "subject": e.get("subject", ""), "top_k": 8})
    top = ", ".join(f"{t['token']}({t['attr']:+.2f})" for t in ex.get("tokens", []))
    print(f"POST {API_URL}/explain  (top-8 phishing drivers for: {e.get('subject','')!r})")
    print(f"  {top}\n  (+ pushes toward phishing, - toward legit; {ex.get('latency_ms','?')} ms)")


if __name__ == "__main__":
    main()
