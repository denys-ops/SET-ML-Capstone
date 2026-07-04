"""Guard: src/preprocess.py та serve/preprocess.py мусять лишатися ідентичними.

serve/ пакується в Docker окремо (Dockerfile копіює лише serve/), тому має власну
копію preprocess.py. Щоб уникнути train/serve skew, ці дві копії мусять збігатися
байт-у-байт. Запускати в CI / перед здачею:  python scripts/check_preprocess_sync.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "preprocess.py"
SERVE = ROOT / "serve" / "preprocess.py"


def main() -> int:
    if not SRC.exists() or not SERVE.exists():
        print(f"MISSING: {SRC if not SRC.exists() else SERVE}")
        return 2
    if SRC.read_bytes() == SERVE.read_bytes():
        print("OK: src/preprocess.py == serve/preprocess.py (train/serve контракт цілий)")
        return 0
    print("DRIFT: src/preprocess.py != serve/preprocess.py — синхронізуй копії!")
    return 1


if __name__ == "__main__":
    sys.exit(main())
