"""
Boru hattı / CI doğrulaması: manifestteki her dosya için manifest width×height boyutunda
rastgele gri PNG üretir. Klinik eğitimde GERÇEK BT kesitlerini kullanın.

Kullanım:
  python scripts/generate_placeholder_pngs.py --manifest artifacts/coco_split.json --out-dir artifacts/demo_images
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        m = json.load(f)
    seen: set[str] = set()
    for split in ("train", "val", "test"):
        for row in m[split]:
            fn = row["file_name"]
            if fn in seen:
                continue
            seen.add(fn)
            w, h = int(row["width"]), int(row["height"])
            arr = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(out / fn)

    print("Yazılan benzersiz görüntü:", len(seen), "->", out.resolve())


if __name__ == "__main__":
    main()
