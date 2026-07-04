"""Дістати ваги моделей у репо (вони НЕ в git — завеликі: 256 MB × 2).

Два джерела:
  1. W&B Model Registry (project `phishing-scorer`) — за замовчуванням.
     Потребує WANDB_API_KEY (env). Колекції:
       * distilbert-aug        (PyTorch)  -> model/distilbert_aug/
       * distilbert-aug-onnx   (ONNX-INT8) -> onnx/model_int8.onnx
  2. Локальні копії з DL-ядра (CAP_A) — прапорець --local. Детерміновано,
     без мережі; корисно на машині автора, де ваги вже лежать.

Приклади:
  WANDB_API_KEY=... python scripts/fetch_models.py          # тягне з W&B
  python scripts/fetch_models.py --local                    # копіює з CAP_A

Базова модель (model/distilbert) не в registry — ноутбук F2 навчає її ідемпотентно,
або бери її з CAP_A через --local.

⚠️ W&B-ваги — це коротко-контекстний експеримент (max_length=20, див. docs/EXPERIMENTS.md),
тоді як числа ноутбука рахуються на max_length=256. Для відповідності ноутбуку бери --local
(канонічні 256) або дай ноутбуку натренувати з нуля.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAP_A = Path("/Users/almin/PyCharmMiscProject/deepML/capstone")

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "phishing-scorer")


def fetch_wandb() -> None:
    if not os.environ.get("WANDB_API_KEY"):
        sys.exit("ERROR: WANDB_API_KEY не заданий. Або задай env, або запусти з --local.")
    import wandb

    def pull(collection: str, alias: str, target: Path) -> None:
        ref = f"wandb-registry-model/{collection}:{alias}"
        target.mkdir(parents=True, exist_ok=True)
        print(f"W&B: {ref} -> {target}")
        # context manager: run.finish() гарантовано (і позначить 'failed' при винятку)
        with wandb.init(project=WANDB_PROJECT, job_type="fetch",
                        settings=wandb.Settings(silent=True)) as run:
            art = run.use_artifact(ref, type="model")
            path = art.download(root=str(target))
        print(f"  downloaded -> {path} (перевір імена файлів усередині)")

    pull("distilbert-aug", "aug", ROOT / "model" / "distilbert_aug")
    pull("distilbert-aug-onnx", "int8", ROOT / "onnx")


def fetch_local() -> None:
    if not CAP_A.exists():
        sys.exit(f"ERROR: CAP_A не знайдено: {CAP_A}")
    jobs = [
        (CAP_A / "model" / "distilbert",     ROOT / "model" / "distilbert"),
        (CAP_A / "model" / "distilbert_aug", ROOT / "model" / "distilbert_aug"),
        (CAP_A / "onnx" / "model_int8.onnx", ROOT / "onnx" / "model_int8.onnx"),
    ]
    for src, dst in jobs:
        if not src.exists():
            print(f"  SKIP (нема): {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        print(f"  local: {src} -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Дістати ваги моделей (W&B або локально).")
    ap.add_argument("--local", action="store_true",
                    help="Копіювати з CAP_A замість W&B registry.")
    args = ap.parse_args()
    (fetch_local if args.local else fetch_wandb)()
    print("готово.")


if __name__ == "__main__":
    main()
