"""Stratify edilmiş train/val/test bölmesi (görüntü düzeyi ikili etiket)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from neurocv.coco_io import CocoImageRecord


def stratified_train_val_test(
    images: list[CocoImageRecord],
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[list[int], list[int], list[int]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")
    ids = [im.image_id for im in images]
    y = [im.label_hemorrhage for im in images]

    ids_arr = np.array(ids, dtype=np.int64)
    y_arr = np.array(y, dtype=np.int64)

    tv_ids, test_ids, tv_y, _ = train_test_split(
        ids_arr,
        y_arr,
        test_size=test_ratio,
        stratify=y_arr,
        random_state=seed,
        shuffle=True,
    )
    val_fraction_of_tv = val_ratio / (train_ratio + val_ratio)
    train_ids, val_ids, _, _ = train_test_split(
        tv_ids,
        tv_y,
        test_size=val_fraction_of_tv,
        stratify=tv_y,
        random_state=seed,
        shuffle=True,
    )

    train_ids = sorted([int(x) for x in train_ids])
    val_ids = sorted([int(x) for x in val_ids])
    test_ids = sorted([int(x) for x in test_ids])
    return train_ids, val_ids, test_ids


def split_manifest_dict(
    images: list[CocoImageRecord],
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
    coco_source: str,
    seed: int,
) -> dict[str, Any]:
    id_to_im = {im.image_id: im for im in images}

    def pack(ids: list[int]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for i in ids:
            im = id_to_im[i]
            rows.append(
                {
                    "image_id": im.image_id,
                    "file_name": im.file_name,
                    "width": im.width,
                    "height": im.height,
                    "label_hemorrhage": im.label_hemorrhage,
                }
            )
        return rows

    return {
        "coco_source": coco_source,
        "seed": seed,
        "train": pack(train_ids),
        "val": pack(val_ids),
        "test": pack(test_ids),
        "stats": {
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
            "pos_train": sum(id_to_im[i].label_hemorrhage for i in train_ids),
            "pos_val": sum(id_to_im[i].label_hemorrhage for i in val_ids),
            "pos_test": sum(id_to_im[i].label_hemorrhage for i in test_ids),
        },
    }


def save_split_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
