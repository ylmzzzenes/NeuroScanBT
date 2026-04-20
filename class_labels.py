"""
Checkpoint label_map ve düz state_dict ile ImageFolder indeks uyumu.

Düz .pth (ImageFolder): 0 = hemorrhage, 1 = no_hemorrhage (alfabetik klasör).
Eski dict checkpoint (model_state + label_map): indeks label_map’ten okunur.
"""
from __future__ import annotations

from typing import Any

IDX_HEMORRHAGE = 1
LEGACY_IDX_HEMORRHAGE = 0


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
