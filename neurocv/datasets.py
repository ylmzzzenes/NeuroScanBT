"""COCO manifest tabanlı PyTorch Dataset (sınıflandırma ve segmentasyon)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from neurocv.coco_io import CocoAnnotationRecord
from neurocv.mask_ops import polygons_to_mask
from neurocv.transforms_med import eval_cls_transforms, preprocess_seg_pair_eval, preprocess_seg_pair_train, train_cls_transforms


class BrainClsDataset(Dataset):
    """label: 0=no hemorrhage, 1=hemorrhage (binary)."""

    def __init__(
        self,
        images_dir: str | Path,
        manifest_rows: list[dict[str, Any]],
        train: bool,
        img_size: int,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.rows = manifest_rows
        self.tf = train_cls_transforms(img_size) if train else eval_cls_transforms(img_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        path = self.images_dir / row["file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"Görüntü bulunamadı: {path}")
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        y = int(row["label_hemorrhage"])
        return x, torch.tensor(y, dtype=torch.long)


class BrainSegDataset(Dataset):
    """Pozitif örneklerde birleşik maske; negatiflerde sıfır maskesi."""

    def __init__(
        self,
        images_dir: str | Path,
        manifest_rows: list[dict[str, Any]],
        anns_index: dict[int, list[CocoAnnotationRecord]],
        train: bool,
        img_size: int,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.rows = manifest_rows
        self.anns_index = anns_index
        self.train = train
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        iid = int(row["image_id"])
        w, h = int(row["width"]), int(row["height"])
        path = self.images_dir / row["file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"Görüntü bulunamadı: {path}")
        img = Image.open(path).convert("RGB")

        anns = self.anns_index.get(iid, [])
        if anns:
            mask_arr = polygons_to_mask(w, h, anns)
            mask = Image.fromarray(mask_arr * 255, mode="L")
        else:
            mask = Image.new("L", (w, h), 0)

        if self.train:
            x, m = preprocess_seg_pair_train(img, mask, self.img_size)
        else:
            x, m = preprocess_seg_pair_eval(img, mask, self.img_size)
        return x, m


def load_manifest_rows(split: str, manifest_path: str | Path) -> list[dict[str, Any]]:
    import json

    with Path(manifest_path).open("r", encoding="utf-8") as f:
        m = json.load(f)
    return list(m[split])
