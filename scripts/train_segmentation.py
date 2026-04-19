"""
COCO polygon birleşik maskesi ile U-Net segmentasyon eğitimi.

Örnek:
  python scripts/train_segmentation.py --manifest artifacts/coco_split.json --images-dir path/to/images --coco-json path/instances_default.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from neurocv.coco_io import parse_coco
from neurocv.coco_io import index_annotations_by_image
from neurocv.datasets import BrainSegDataset
from neurocv.logging_utils import CsvLogger
from neurocv.losses import segmentation_loss
from neurocv.metrics import dice_iou_from_logits
from neurocv.models.unet_small import UNetSmall
from neurocv.transforms_med import IMG_SIZE


def load_rows(manifest: Path, split: str) -> list:
    with manifest.open("r", encoding="utf-8") as f:
        m = json.load(f)
    return list(m[split])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="artifacts/coco_split.json")
    ap.add_argument("--coco-json", type=str, required=True, help="Polygon kaynağı COCO JSON")
    ap.add_argument("--images-dir", type=str, required=True)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="runs/seg_coco")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _, anns, _ = parse_coco(args.coco_json)
    ann_index = index_annotations_by_image(anns)

    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "best_unet.pt"
    hist = CsvLogger(out_dir / "segmentation_log.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows = load_rows(manifest, "train")
    val_rows = load_rows(manifest, "val")

    train_ds = BrainSegDataset(args.images_dir, train_rows, ann_index, train=True, img_size=args.img_size)
    val_ds = BrainSegDataset(args.images_dir, val_rows, ann_index, train=False, img_size=args.img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNetSmall(in_ch=3, base=32).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_epoch = -1
    patience_left = args.patience
    best_state = None

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for x, m in train_loader:
            x = x.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = segmentation_loss(logits, m)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            n_tr += x.size(0)
        scheduler.step()
        tr_loss /= max(n_tr, 1)

        model.eval()
        va_loss = 0.0
        n_va = 0
        dice_acc = 0.0
        iou_acc = 0.0
        n_batches = 0
        with torch.no_grad():
            for x, m in val_loader:
                x = x.to(device, non_blocking=True)
                m = m.to(device, non_blocking=True)
                logits = model(x)
                loss = segmentation_loss(logits, m)
                va_loss += loss.item() * x.size(0)
                n_va += x.size(0)
                d, j = dice_iou_from_logits(logits, m)
                dice_acc += d
                iou_acc += j
                n_batches += 1
        va_loss /= max(n_va, 1)
        dice_acc /= max(n_batches, 1)
        iou_acc /= max(n_batches, 1)

        hist.log(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "val_dice": dice_acc,
                "val_iou": iou_acc,
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )
        print(
            f"Epoch {epoch}/{args.epochs}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_dice={dice_acc:.4f}  val_iou={iou_acc:.4f}"
        )

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_epoch = epoch
            patience_left = args.patience
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "img_size": args.img_size,
                },
                weights_path,
            )
            print("  -> En iyi checkpoint:", weights_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Erken durdurma.")
                break

    elapsed = time.time() - start

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val),
        "seconds": float(elapsed),
        "weights": str(weights_path.resolve()),
    }
    with (out_dir / "segmentation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Segmentasyon özeti:", json.dumps(summary, indent=2))
    hist.close()


if __name__ == "__main__":
    main()
