"""Spearman-валідація шкали очевидності (ціль капстоуна §6.3).

Питання: чи модельне каліброване `p_phish` монотонно відповідає ЛЮДСЬКОМУ
відчуттю «наскільки очевидний це фішинг» (obviousness 0–10)?

Дані: data/labeled/labeled_v{1,2}.parquet — ручна розмітка (Label Studio),
колонки text, verdict (Phishing/Safe), obviousness (1–10). ~ кілька десятків
прикладів — замало для fine-tune, але достатньо для рангової кореляції.
⚠️ Ці дані НІКОЛИ не для тренування — лише eval.

Контракт: той самий `html_to_scoring_text`, що й у serving (src/preprocess.py),
+ temperature scaling T з data/calibration.json.  Запуск:
    PYTHONPATH=src python src/eval_scale.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocess import html_to_scoring_text

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model" / "distilbert_aug"   # дефолт для корпоративного стилю
CALIB = ROOT / "data" / "calibration.json"
LABELED = [ROOT / "data" / "labeled" / f"labeled_v{v}.parquet" for v in (1, 2)]
MAX_LEN = 256
OUT_PNG = ROOT / "docs" / "spearman_scale.png"


def load_labeled() -> pd.DataFrame:
    frames = [pd.read_parquet(f) for f in LABELED if f.exists()]
    if not frames:
        raise FileNotFoundError(
            f"не знайдено розмічених parquet: {[str(p) for p in LABELED]}")
    df = pd.concat(frames, ignore_index=True)
    # v1/v2 перетинаються — лишаємо останню розмітку кожного sample_id
    if "sample_id" in df.columns:
        df = df.drop_duplicates(subset="sample_id", keep="last").reset_index(drop=True)
    # рядки без тексту скорити неможливо (NaN зламав би html_to_scoring_text)
    n0 = len(df)
    df = df[df["text"].notna()].reset_index(drop=True)
    if len(df) < n0:
        print(f"  відкинуто {n0 - len(df)} рядків без тексту")
    return df


@torch.no_grad()
def score(texts: list[str], temperature: float) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).eval()
    probs = []
    for t in texts:
        st = html_to_scoring_text(t)                 # train/serve контракт
        enc = tok(st, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        logits = model(**enc).logits.squeeze(0)
        p = torch.softmax(logits / temperature, dim=-1)[1].item()  # index 1 = phish
        probs.append(p)
    return np.array(probs)


def main() -> None:
    T = json.loads(CALIB.read_text())["temperature"]
    df = load_labeled()
    df["p_phish"] = score(df["text"].tolist(), T)
    print(f"T = {T:.4f}  |  всього розмічено: {len(df)}")

    # Розмітники оцінювали obviousness ЛИШЕ для фішингу (Safe -> NaN).
    # PRIMARY: рангова кореляція на фішинг-підмножині (obviousness 1–10).
    phish = df.dropna(subset=["obviousness"])
    if len(phish) < 2:
        raise ValueError(
            f"замало фішинг-рядків з obviousness для Spearman: n={len(phish)}")
    rho_p, pv_p = spearmanr(phish["obviousness"], phish["p_phish"])
    print(f"[primary] phishing-only  n={len(phish):2d}  "
          f"Spearman rho = {rho_p:.3f}  (p={pv_p:.2e})")

    # SECONDARY: повний діапазон, Safe імпутовано як obviousness=0 (легіт = «не палево»).
    full = df.copy()
    full["obviousness"] = full["obviousness"].fillna(0.0)
    rho_f, pv_f = spearmanr(full["obviousness"], full["p_phish"])
    print(f"[secondary] Safe=0        n={len(full):2d}  "
          f"Spearman rho = {rho_f:.3f}  (p={pv_f:.2e})")

    # санітарна перевірка бінарного розділення
    for v in ("Phishing", "Safe"):
        sub = df[df["verdict"] == v]["p_phish"]
        if len(sub):
            print(f"  mean p_phish [{v:8s}] = {sub.mean():.3f}  (n={len(sub)})")

    _plot(full, rho_p, rho_f)
    print(f"scatter -> {OUT_PNG}")


def _plot(full: pd.DataFrame, rho_primary: float, rho_full: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    def jit(a, s):
        return a.to_numpy(float) + rng.uniform(-s, s, len(a))
    fig, ax = plt.subplots(figsize=(7.4, 5))
    for verd, c, lbl in [("Safe", "#2e86ab", "легіт"), ("Phishing", "#d1495b", "фішинг")]:
        m = full["verdict"] == verd
        ax.scatter(jit(full.loc[m, "obviousness"], .14), jit(full.loc[m, "p_phish"], .02),
                   c=c, s=60, alpha=.72, edgecolors="white", linewidth=.7, label=lbl, zorder=3)
    ax.plot([0, 10], [0, 1], "--", color="#aaa", lw=1.3, zorder=1, label="ідеальний збіг")
    ax.set_xlabel("Оцінка людини: наскільки очевидний фішинг  (0 = легіт, 10 = явний)")
    ax.set_ylabel("Бал моделі  (0-1)")
    ax.set_xlim(-.6, 10.6); ax.set_ylim(-.1, 1.12)
    ax.set_title(f"Бал моделі проти оцінки людини\n"
                 f"ρ(усе)={rho_full:.2f}   ·   ρ(лише фішинг)={rho_primary:.2f}", fontsize=12)
    ax.annotate("фішинг, який модель зловила  (бал ≈ 1)", (2.9, 1.0), (1.4, .78),
                fontsize=9.5, color="#7a2233", arrowprops=dict(arrowstyle="->", color="#b3536a", lw=1.1))
    ax.annotate("фішинг, який модель проґавила  (бал ≈ 0)", (5, .02), (3.3, .30),
                fontsize=9.5, color="#7a2233", arrowprops=dict(arrowstyle="->", color="#b3536a", lw=1.1))
    ax.annotate("легіт → низький бал", (0, .02), (.5, .18),
                fontsize=9.5, color="#1c5a75", arrowprops=dict(arrowstyle="->", color="#2e86ab", lw=1.1))
    ax.legend(loc="center right", fontsize=9, framealpha=.95)
    ax.grid(alpha=.22, zorder=0)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=135)


if __name__ == "__main__":
    main()
