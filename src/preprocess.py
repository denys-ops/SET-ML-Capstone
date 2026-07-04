"""Shared preprocessing contract: `html_to_scoring_text`.

The SAME function must run on training data and on the serving input (HW3),
otherwise the model sees a different distribution at serve time (train/serve skew).
Extracted verbatim (logic-wise) from the capstone notebook so training (HW2) and
the FastAPI endpoint (HW3) import one source of truth.

Steps:
  1. strip <head>/<style>/<script>/<title>, comments; keep visible text;
  2. prepend subject;
  3. render Gophish placeholders ({{.FirstName}} -> John, {{.URL}} -> <URL>, ...);
  4. normalise URL / EMAIL / NUM into placeholders; squeeze whitespace.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup, Comment

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
NUM_RE = re.compile(r"\b\d[\d.,]*\b")
WS_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[a-zA-Z!/][^>]*>")

GOPHISH = {
    "{{.FirstName}}": "John", "{{.LastName}}": "Smith",
    "{{.Email}}": "<EMAIL>", "{{.From}}": "<EMAIL>",
    "{{.URL}}": "<URL>", "{{.TrackingURL}}": "<URL>",
}


def render_placeholders(s: str) -> str:
    for k, v in GOPHISH.items():
        s = s.replace(k, v)
    return re.sub(r"\{\{\.\w+\}\}", "<VAR>", s)  # any other {{.X}}


def html_to_text(html: str) -> str:
    if not html:
        return ""
    if not TAG_RE.search(html):  # already plain text — skip parsing
        return html
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["head", "style", "script", "title"]):
        tag.decompose()
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()
    return soup.get_text(separator=" ")


def html_to_scoring_text(html: str, subject: str = "") -> str:
    """HTML(+subject) -> normalised text the model sees. Identical on train/serve."""
    raw = (html or "")[:30000]  # clip abnormally long bodies
    text = (subject or "") + "\n" + html_to_text(raw)
    text = render_placeholders(text)
    text = URL_RE.sub("<URL>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = NUM_RE.sub("<NUM>", text)
    return WS_RE.sub(" ", text).strip()


if __name__ == "__main__":
    demo = (
        '<!doctype html><html><head><style>body{padding:0}</style></head>'
        '<body><p>Hi {{.FirstName}},</p>'
        '<p>Confirm your account at <a href="{{.URL}}">https://acme-login.example.com/verify</a>'
        ' before 5pm. Amount due: $4,200.</p></body></html>'
    )
    print(html_to_scoring_text(demo, subject="Urgent: Wire Transfer"))
