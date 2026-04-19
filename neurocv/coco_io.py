"""
COCO (CVAT dışa aktarımı) JSON okuma ve görüntü–annotation eşlemesi.

Kurallar (bu veri seti):
- categories içinde yalnızca "hemorrhage" (id=1) bulunur.
- Bir görüntüde en az bir annotation varsa pozitif (kanama var).
- Annotation yoksa negatif (kanama yok, boş maske).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CocoImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    """1 = kanama (en az bir polygon), 0 = kanama yok."""
    label_hemorrhage: int


@dataclass(frozen=True)
class CocoAnnotationRecord:
    ann_id: int
    image_id: int
    category_id: int
    segmentation: list[list[float]]
    area: float
    bbox: list[float]


def load_coco_dict(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_coco(
    coco_path: str | Path,
) -> tuple[list[CocoImageRecord], list[CocoAnnotationRecord], list[dict[str, Any]]]:
    data = load_coco_dict(coco_path)
    raw_imgs = data.get("images") or []
    raw_anns = data.get("annotations") or []
    categories = data.get("categories") or []

    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for a in raw_anns:
        iid = int(a["image_id"])
        anns_by_image.setdefault(iid, []).append(a)

    images: list[CocoImageRecord] = []
    for im in raw_imgs:
        iid = int(im["id"])
        has = 1 if iid in anns_by_image and len(anns_by_image[iid]) > 0 else 0
        images.append(
            CocoImageRecord(
                image_id=iid,
                file_name=str(im["file_name"]),
                width=int(im["width"]),
                height=int(im["height"]),
                label_hemorrhage=has,
            )
        )

    anns_out: list[CocoAnnotationRecord] = []
    for a in raw_anns:
        anns_out.append(
            CocoAnnotationRecord(
                ann_id=int(a["id"]),
                image_id=int(a["image_id"]),
                category_id=int(a["category_id"]),
                segmentation=a.get("segmentation") or [],
                area=float(a.get("area", 0.0)),
                bbox=[float(x) for x in a.get("bbox", [])],
            )
        )

    images.sort(key=lambda x: x.image_id)
    return images, anns_out, categories


def index_annotations_by_image(anns: list[CocoAnnotationRecord]) -> dict[int, list[CocoAnnotationRecord]]:
    out: dict[int, list[CocoAnnotationRecord]] = {}
    for a in anns:
        out.setdefault(a.image_id, []).append(a)
    return out
