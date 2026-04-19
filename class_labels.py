"""
İkili sınıflandırma: checkpoint label_map ve legacy ImageFolder indeksleri.

Eski manifest/COCO eğitiminde: 0 = no_hemorrhage, 1 = hemorrhage (label_map).
ImageFolder (alfabetik): 0 = hemorrhage, 1 = no_hemorrhage.
"""
from __future__ import annotations

from typing import Any

IDX_NO_HEMORRHAGE = 0
IDX_HEMORRHAGE = 1

# Eski .pt meta alanında bulunabilir
COCO_LABEL_MAP: dict[str, str] = {
    "0": "no_hemorrhage",
    "1": "hemorrhage",
}

LEGACY_IDX_HEMORRHAGE = 0
LEGACY_IDX_NO_HEMORRHAGE = 1

LEGACY_LABEL_MAP: dict[str, str] = {
    "0": "hemorrhage",
    "1": "no_hemorrhage",
}


def hemorrhage_class_index_from_checkpoint(ck: dict[str, Any]) -> int:
    lm = ck.get("label_map")
    if isinstance(lm, dict):
        for k, v in lm.items():
            if v == "hemorrhage":
                return int(k)
    if "model_state" in ck:
        return IDX_HEMORRHAGE
    return LEGACY_IDX_HEMORRHAGE


def load_state_dict_and_hemorrhage_index(blob: Any) -> tuple[Any, int]:
    if isinstance(blob, dict) and "model_state" in blob:
        return blob["model_state"], hemorrhage_class_index_from_checkpoint(blob)
    return blob, LEGACY_IDX_HEMORRHAGE


def class_name_from_index(pred_idx: int, hem_idx: int) -> str:
    return "hemorrhage" if pred_idx == hem_idx else "no_hemorrhage"
