"""
COCO manifest ile ikili sınıflandırma (ResNet-18) eğitimi.
Eski proje .pth dosyaları kullanılmaz; ImageNet ön eğitim ağırlıkları ile başlar.

Örnek:
  python scripts/train_classification.py --manifest artifacts/coco_split.json --images-dir path/to/images
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
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from neurocv.datasets import BrainClsDataset
from neurocv.logging_utils import CsvLogger
from neurocv.losses import FocalLoss
from neurocv.models.resnet_cls import build_resnet18_binary
from neurocv.transforms_med import IMG_SIZE


def load_rows(manifest: Path, split: str) -> list:
    with manifest.open("r", encoding="utf-8") as f:
        m = json.load(f)
    return list(m[split])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="artifacts/coco_split.json")
    ap.add_argument("--images-dir", type=str, required=True, help="Görüntülerin kökü (manifest file_name ile birleştirilir)")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-focal", action="store_true", help="CrossEntropyLoss (ağırlıklı) kullan")
    ap.add_argument("--out-dir", type=str, default="runs/cls_coco")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "best_cls.pt"
    hist = CsvLogger(out_dir / "classification_log.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = load_rows(manifest, "train")
    val_rows = load_rows(manifest, "val")

    train_ds = BrainClsDataset(args.images_dir, train_rows, train=True, img_size=args.img_size)
    val_ds = BrainClsDataset(args.images_dir, val_rows, train=False, img_size=args.img_size)

    train_y = np.array([int(r["label_hemorrhage"]) for r in train_rows], dtype=np.int64)
    n0 = int((train_y == 0).sum())
    n1 = int((train_y == 1).sum())
    w0 = len(train_y) / (2 * max(n0, 1))
    w1 = len(train_y) / (2 * max(n1, 1))
    class_w = torch.tensor([w0, w1], dtype=torch.float32, device=device)

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

    model = build_resnet18_binary(use_imagenet=True).to(device)
    if not args.no_focal:
        criterion = FocalLoss(gamma=2.0, weight=class_w)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_w)
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
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            n_tr += x.size(0)
        scheduler.step()
        tr_loss /= max(n_tr, 1)

        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                va_loss += loss.item() * x.size(0)
                n_va += x.size(0)
        va_loss /= max(n_va, 1)

        hist.log(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )
        print(f"Epoch {epoch}/{args.epochs}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")

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
                    "class_weights": [float(w0), float(w1)],
                },
                weights_path,
            )
            print("  -> En iyi checkpoint kaydedildi:", weights_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Erken durdurma.")
                break

    elapsed = time.time() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    # Val metrikleri (en iyi ağırlıklar)
    model.eval()
    ys: list[int] = []
    ps: list[int] = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(pred.tolist())

    acc = accuracy_score(ys, ps)
    prec = precision_score(ys, ps, pos_label=1, average="binary", zero_division=0)
    rec = recall_score(ys, ps, pos_label=1, average="binary", zero_division=0)
    f1 = f1_score(ys, ps, pos_label=1, average="binary", zero_division=0)
    cm = confusion_matrix(ys, ps, labels=[0, 1])

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val),
        "seconds": float(elapsed),
        "val_accuracy": float(acc),
        "val_precision_hemorrhage": float(prec),
        "val_recall_hemorrhage": float(rec),
        "val_f1_hemorrhage": float(f1),
        "confusion_matrix_val": cm.tolist(),
    }
    with (out_dir / "classification_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Özet:", json.dumps(summary, indent=2))
    print("Süre (s):", round(elapsed, 1))
    hist.close()


if __name__ == "__main__":
    main()
