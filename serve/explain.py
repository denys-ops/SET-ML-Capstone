"""Explainability backend — per-token attributions for inline highlighting.

Layer Integrated Gradients (Captum) on the embedding layer, attributed toward the
phishing class. Positive attr = pushes the score toward *phishing*, negative = toward
*legit* (same convention as the capstone F4). This powers the Whalen editor's inline
highlighting and is **on-demand** (a click), not per-keystroke, so latency is not
critical — hence PyTorch + gradients here rather than ONNX.
"""
from __future__ import annotations

import torch
from captum.attr import LayerIntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocess import html_to_scoring_text
from scoring import MAX_LENGTH, format_results


class Explainer:
    def __init__(self, model_dir: str, n_steps: int = 50):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.device = "cpu"  # IG is more stable/deterministic on CPU
        self.model.to(self.device)
        self.n_steps = n_steps
        self.lig = LayerIntegratedGradients(self._forward, self.model.get_input_embeddings())

    def _forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def explain(self, html: str, subject: str = "", top_k: int | None = None) -> dict:
        text = html_to_scoring_text(html, subject)
        enc = self.tokenizer(text, truncation=True, max_length=MAX_LENGTH,
                             return_tensors="pt").to(self.device)
        input_ids, attn = enc["input_ids"], enc["attention_mask"]

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attn).logits
        result = format_results(logits.cpu().numpy())[0]

        # baseline = all-PAD sequence; attribute toward the phishing class (target=1)
        ref = torch.full_like(input_ids, self.tokenizer.pad_token_id)
        attributions = self.lig.attribute(
            inputs=input_ids, baselines=ref, additional_forward_args=(attn,),
            target=1, n_steps=self.n_steps)
        attr = attributions.sum(dim=-1).squeeze(0)          # per-token score
        attr = attr / (attr.norm() + 1e-12)                 # L2-normalise for comparability

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        special = set(self.tokenizer.all_special_tokens)
        spans = [{"token": t, "attr": round(float(a), 4)}
                 for t, a in zip(tokens, attr.tolist()) if t not in special]
        if top_k:
            spans = sorted(spans, key=lambda s: abs(s["attr"]), reverse=True)[:top_k]
        result["tokens"] = spans
        return result


if __name__ == "__main__":
    import json
    import os
    import sys

    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_DIR", "models/serve")
    ex = Explainer(model_dir)
    demo = '<p>Urgent: verify your password at http://acme-login.example.com now</p>'
    print(json.dumps(ex.explain(demo, subject="Account suspended", top_k=10), indent=2))
