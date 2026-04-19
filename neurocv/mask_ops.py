"""COCO polygon segmentasyonlarını ikili maskeye dönüştürme."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from neurocv.coco_io import CocoAnnotationRecord


def polygons_to_mask(
    width: int,
    height: int,
    annotations: list[CocoAnnotationRecord],
    merge: str = "union",
) -> np.ndarray:
    """
    Tüm polygonları tek kanallı uint8 maskeye çizer (0/1).
    merge='union': pikseller mantıksal OR ile birleşir.
    """
    if merge != "union":
        raise ValueError("merge must be 'union'")

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for ann in annotations:
        seg = ann.segmentation
        if not seg:
            continue
        for poly in seg:
            if len(poly) < 6:
                continue
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            draw.polygon(pts, outline=1, fill=1)

    return np.asarray(mask, dtype=np.uint8)
