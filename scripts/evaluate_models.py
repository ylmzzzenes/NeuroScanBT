"""
Test split üzerinde sınıflandırma + segmentasyon değerlendirmesi, isteğe bağlı eski ResNet karşılaştırması.

Örnek:
  python scripts/evaluate_models.py --manifest artifacts/coco_split.json --images-dir path/to/images \\
    --coco-json path/instances_default.json --cls-weights runs/cls_coco/best_cls.pt --seg-weights runs/seg_coco/best_unet.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import models

from neurocv.coco_io import index_annotations_by_image, parse_coco
from neurocv.datasets import BrainClsDataset, BrainSegDataset
from neurocv.metrics import dice_iou_from_logits
from neurocv.models.resnet_cls import build_resnet18_binary
from neurocv.models.unet_small import UNetSmall
from neurocv.transforms_med import IMG_SIZE


def load_rows(manifest: Path, split: str) -> list:
    with manifest.open("r", encoding="utf-8") as f:
        m = json.load(f)
    return list(m[split])


def evaluate_cls(model, loader, device) -> dict:
    model.eval()
    ys, ps, probs_h = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pr = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = logits.argmax(dim=1).cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(pred.tolist())
            probs_h.extend(pr.tolist())
    acc = accuracy_score(ys, ps)
    prec = precision_score(ys, ps, pos_label=1, average="binary", zero_division=0)
    rec = recall_score(ys, ps, pos_label=1, average="binary", zero_division=0)
    f1 = f1_score(ys, ps, pos_label=1, average="binary", zero_division=0)
    cm = confusion_matrix(ys, ps, labels=[0, 1])
    return {
        "accuracy": float(acc),
        "precision_hemorrhage": float(prec),
        "recall_hemorrhage": float(rec),
        "f1_hemorrhage": float(f1),
        "confusion_matrix": cm.tolist(),
        "y_true": ys,
        "y_pred": ps,
        "hemorrhage_prob": probs_h,
    }


def evaluate_seg(model, loader, device) -> dict:
    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for x, m in loader:
            x = x.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            logits = model(x)
            d, j = dice_iou_from_logits(logits, m)
            dices.append(d)
            ious.append(j)
    return {
        "mean_dice": float(np.mean(dices)),
        "mean_iou": float(np.mean(ious)),
    }


def load_legacy_resnet(path: Path, device):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    ck = torch.load(path, map_location=device)
    if isinstance(ck, dict) and "model_state" in ck:
        m.load_state_dict(ck["model_state"])
    else:
        m.load_state_dict(ck)
    m = m.to(device)
    m.eval()
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="artifacts/coco_split.json")
    ap.add_argument("--coco-json", type=str, required=True)
    ap.add_argument("--images-dir", type=str, required=True)
    ap.add_argument("--cls-weights", type=str, required=True)
    ap.add_argument("--seg-weights", type=str, required=True)
    ap.add_argument("--legacy-weights", type=str, default="", help="İsteğe bağlı eski best_resnet18.pth karşılaştırması")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", type=str, default="runs/eval_report.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = Path(args.manifest)

    _, anns, _ = parse_coco(args.coco_json)
    ann_index = index_annotations_by_image(anns)

    test_rows = load_rows(manifest, "test")
    cls_ds = BrainClsDataset(args.images_dir, test_rows, train=False, img_size=args.img_size)
    seg_ds = BrainSegDataset(args.images_dir, test_rows, ann_index, train=False, img_size=args.img_size)
    cls_loader = DataLoader(cls_ds, batch_size=args.batch_size, shuffle=False)
    seg_loader = DataLoader(seg_ds, batch_size=args.batch_size, shuffle=False)

    cls = build_resnet18_binary(use_imagenet=False).to(device)
    ck = torch.load(args.cls_weights, map_location=device)
    cls.load_state_dict(ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck)
    cls_metrics = evaluate_cls(cls, cls_loader, device)

    seg = UNetSmall(in_ch=3, base=32).to(device)
    sck = torch.load(args.seg_weights, map_location=device)
    seg.load_state_dict(sck["model_state"] if isinstance(sck, dict) and "model_state" in sck else sck)
    seg_metrics = evaluate_seg(seg, seg_loader, device)

    miscls = [
        {
            "file_name": test_rows[i]["file_name"],
            "y_true": cls_metrics["y_true"][i],
            "y_pred": cls_metrics["y_pred"][i],
            "p_hemorrhage": cls_metrics["hemorrhage_prob"][i],
        }
        for i in range(len(test_rows))
        if cls_metrics["y_true"][i] != cls_metrics["y_pred"][i]
    ]

    report = {
        "classification_new_resnet": {
            "accuracy": cls_metrics["accuracy"],
            "precision_hemorrhage": cls_metrics["precision_hemorrhage"],
            "recall_hemorrhage": cls_metrics["recall_hemorrhage"],
            "f1_hemorrhage": cls_metrics["f1_hemorrhage"],
            "confusion_matrix": cls_metrics["confusion_matrix"],
            "misclassified_count": len(miscls),
            "misclassified": miscls,
        },
        "segmentation_unet": seg_metrics,
    }

    if args.legacy_weights and Path(args.legacy_weights).is_file():
        leg = load_legacy_resnet(Path(args.legacy_weights), device)
        # Eski eğitim ImageFolder sırası: alfabetik hem=0, no=1 farklı olabilir; burada sadece doğrusal karşılaştırma
        leg_metrics = evaluate_cls(leg, cls_loader, device)
        report["legacy_resnet18_best_pth"] = {
            "accuracy": leg_metrics["accuracy"],
            "f1_hemorrhage": leg_metrics["f1_hemorrhage"],
            "note": "Eski ağırlık farklı veri/dönüşümle eğitildi; doğrudan adil karşılaştırma olmayabilir.",
        }
        report["delta_f1_new_minus_legacy"] = cls_metrics["f1_hemorrhage"] - leg_metrics["f1_hemorrhage"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
