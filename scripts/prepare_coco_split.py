"""
COCO JSON'dan stratify edilmiş train/val/test manifest üretir.
Kullanım:
  python scripts/prepare_coco_split.py --coco-json path/instances_default.json --out artifacts/coco_split.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neurocv.coco_io import parse_coco
from neurocv.splitting import save_split_manifest, split_manifest_dict, stratified_train_val_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-json", type=str, required=True, help="CVAT COCO instances JSON")
    ap.add_argument("--out", type=str, default="artifacts/coco_split.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images, _, _ = parse_coco(args.coco_json)
    train_ids, val_ids, test_ids = stratified_train_val_test(images, seed=args.seed)

    payload = split_manifest_dict(images, train_ids, val_ids, test_ids, coco_source=str(Path(args.coco_json).resolve()), seed=args.seed)
    save_split_manifest(args.out, payload)
    print("Yazıldı:", Path(args.out).resolve())
    print("İstatistik:", payload["stats"])


if __name__ == "__main__":
    main()
