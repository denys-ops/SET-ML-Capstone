"""HW2 — training + experiment tracking (W&B).

Tasks (each = one comparable W&B run):
  baseline                      TF-IDF + LogReg (honest reference point)
  distilbert --variant base     DistilBERT fine-tune on public data
  distilbert --variant aug      continue-train base on corporate-style synthetic mix (F9)

Device-agnostic (cuda -> mps -> cpu) so the SAME script runs locally and on Colab GPU.
Hyperparameters come from params.yaml. Metrics/params/artifacts are logged to W&B;
the saved model is uploaded as a W&B artifact (-> Model Registry, see README).

Examples:
  python src/train.py baseline
  python src/train.py distilbert --variant base
  python src/train.py distilbert --variant aug --base-model models/distilbert-base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import wandb
from data import get_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def load_params() -> dict:
    p = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text())
    # env overrides (handy on Colab: DATA_SOURCE=hf, WANDB_PROJECT=...)
    import os
    if os.environ.get("DATA_SOURCE"):
        p["data"]["source"] = os.environ["DATA_SOURCE"]
    if os.environ.get("WANDB_PROJECT"):
        p["wandb"]["project"] = os.environ["WANDB_PROJECT"]
    return p


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def metrics(y_true, proba, thr: float = 0.5) -> dict:
    pred = (np.asarray(proba) >= thr).astype(int)
    return {
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
    }


# --------------------------------------------------------------------------- #
# Baseline: TF-IDF + Logistic Regression
# --------------------------------------------------------------------------- #
def run_baseline(p: dict) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    splits = get_splits(p["data"]["source"])
    tr, te = splits["train"], splits["test"]

    cfg = {"model": "tfidf+logreg", "ngram": "1-2", "min_df": 5,
           "max_features": 50000, "C": 4.0, "seed": p["seed"]}
    wandb.init(project=p["wandb"]["project"], entity=p["wandb"]["entity"],
               name="baseline-tfidf-logreg", job_type="train", config=cfg)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000,
                            sublinear_tf=True, strip_accents="unicode")
    Xtr = tfidf.fit_transform(tr["text"])
    Xte = tfidf.transform(te["text"])
    clf = LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced")
    clf.fit(Xtr, tr["label"])

    proba = clf.predict_proba(Xte)[:, 1]
    m = metrics(te["label"], proba)
    print("baseline test:", m)
    wandb.log({f"test/{k}": v for k, v in m.items()})
    wandb.summary.update({f"test/{k}": v for k, v in m.items()})
    wandb.finish()


# --------------------------------------------------------------------------- #
# DistilBERT: base fine-tune / augmented continue-train
# --------------------------------------------------------------------------- #
def _build_aug_trainset(p: dict, train_df: pd.DataFrame) -> pd.DataFrame:
    """Mix public subset + oversampled corporate synthetic (capstone F9 recipe)."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from synth import gen
    from preprocess import html_to_scoring_text

    a = p["augment"]
    synth = pd.DataFrame(gen(a["synth_n_phish"], a["synth_n_legit"], seed=a["synth_seed"]),
                         columns=["raw", "label", "category"]).drop_duplicates("raw")
    synth["text"] = synth["raw"].map(html_to_scoring_text)
    pub = train_df.sample(min(a["n_public"], len(train_df)), random_state=p["seed"])[["text", "label"]]
    mix = pd.concat([pub, *[synth[["text", "label"]]] * a["synth_oversample"]],
                    ignore_index=True).sample(frac=1, random_state=p["seed"]).reset_index(drop=True)
    print(f"aug trainset: {len(pub)} public + {len(synth)}x{a['synth_oversample']} synth = {len(mix)}")
    return mix


def run_distilbert(p: dict, variant: str, base_model: str | None) -> None:
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

    set_seed(p["seed"])
    device = pick_device()
    name, max_len = p["model"]["name"], p["model"]["max_length"]
    splits = get_splits(p["data"]["source"])
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    tokenizer = AutoTokenizer.from_pretrained(name)

    def to_ds(df):
        return Dataset.from_pandas(df[["text", "label"]], preserve_index=False).map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=max_len), batched=True)

    if variant == "aug":
        init_from = base_model or str(MODELS_DIR / "distilbert-base")
        train_data = _build_aug_trainset(p, train_df)
        lr, epochs = p["augment"]["lr"], p["augment"]["epochs"]
        run_name, out_name = "distilbert-aug", "distilbert-aug"
    else:
        init_from = name
        train_data = train_df
        lr, epochs = p["train"]["lr"], p["train"]["epochs"]
        run_name, out_name = "distilbert-base", "distilbert-base"

    cfg = {"variant": variant, "model": name, "max_length": max_len, "lr": lr,
           "epochs": epochs, "batch_size": p["train"]["batch_size"],
           "weight_decay": p["train"]["weight_decay"], "device": device,
           "init_from": init_from, "seed": p["seed"]}
    wandb.init(project=p["wandb"]["project"], entity=p["wandb"]["entity"],
               name=run_name, job_type="train", config=cfg)

    def compute_metrics(ep):
        logits, labels = ep
        proba = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        return metrics(labels, proba)

    model = AutoModelForSequenceClassification.from_pretrained(init_from, num_labels=2)
    save_dir = MODELS_DIR / out_name
    args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "out" / out_name),
        learning_rate=lr, per_device_train_batch_size=p["train"]["batch_size"],
        per_device_eval_batch_size=p["train"]["batch_size"], num_train_epochs=epochs,
        weight_decay=p["train"]["weight_decay"],
        eval_strategy="epoch" if variant == "base" else "no",
        save_strategy="no", logging_steps=50,
        report_to="wandb", run_name=run_name,
        fp16=(device == "cuda"), seed=p["seed"])

    trainer = Trainer(
        model=model, args=args, train_dataset=to_ds(train_data),
        eval_dataset=to_ds(val_df) if variant == "base" else None,
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics)

    trainer.train()
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    # final test metrics
    test_logits = trainer.predict(to_ds(test_df)).predictions
    proba = torch.softmax(torch.tensor(test_logits), dim=1)[:, 1].numpy()
    m = metrics(test_df["label"], proba)
    print(f"{run_name} test:", m)
    wandb.log({f"test/{k}": v for k, v in m.items()})
    wandb.summary.update({f"test/{k}": v for k, v in m.items()})

    # log model as artifact -> link to Model Registry from the W&B UI / CLI
    art = wandb.Artifact(out_name, type="model", metadata={**cfg, **{f"test_{k}": v for k, v in m.items()}})
    art.add_dir(str(save_dir))
    wandb.log_artifact(art, aliases=["latest", variant])
    wandb.finish()
    print("saved model ->", save_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="task", required=True)
    sub.add_parser("baseline", help="TF-IDF + LogReg reference run")
    d = sub.add_parser("distilbert", help="DistilBERT fine-tune")
    d.add_argument("--variant", choices=["base", "aug"], default="base")
    d.add_argument("--base-model", default=None,
                   help="path to base model for --variant aug (default models/distilbert-base)")

    args = ap.parse_args()
    p = load_params()
    if args.task == "baseline":
        run_baseline(p)
    else:
        run_distilbert(p, args.variant, args.base_model)


if __name__ == "__main__":
    main()
