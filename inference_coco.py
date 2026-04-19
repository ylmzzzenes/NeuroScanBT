"""
COCO manifest ile eğitilmiş sınıflandırma + U-Net segmentasyon çıkarımı.

Ağırlık yolları:
- Ortam değişkenleri: NEURO_CLS_WEIGHTS, NEURO_SEG_WEIGHTS
- Varsayılan: weights/best_cls.pt, weights/best_unet.pt
"""
from __future__ import annotations

import base64
import io
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from neurocv.models.resnet_cls import build_resnet18_binary
from neurocv.models.unet_small import UNetSmall
from neurocv.transforms_med import IMG_SIZE, inference_cls_tensor, inference_seg_tensors

# ImageFolder sırası ile uyumlu: sınıf 0 = hemorrhage
HEM_CLASS_IDX = 0

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLS = os.path.join(_SCRIPT_DIR, "weights", "best_cls.pt")
DEFAULT_SEG = os.path.join(_SCRIPT_DIR, "weights", "best_unet.pt")

CLS_PATH = os.environ.get("NEURO_CLS_WEIGHTS", DEFAULT_CLS)
SEG_PATH = os.environ.get("NEURO_SEG_WEIGHTS", DEFAULT_SEG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cls_model: nn.Module | None = None
seg_model: nn.Module | None = None
_cls_img_size: int = IMG_SIZE
_seg_img_size: int = IMG_SIZE


def load_coco_models(cls_path: str | None = None, seg_path: str | None = None) -> tuple[nn.Module, nn.Module]:
    global cls_model, seg_model, _cls_img_size, _seg_img_size
    cp = cls_path or CLS_PATH
    sp = seg_path or SEG_PATH

    c = build_resnet18_binary(use_imagenet=False).to(device)
    ck = torch.load(cp, map_location=device)
    c.load_state_dict(ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck)
    _cls_img_size = int(ck.get("img_size", IMG_SIZE)) if isinstance(ck, dict) else IMG_SIZE
    c.eval()

    s = UNetSmall(in_ch=3, base=32).to(device)
    sk = torch.load(sp, map_location=device)
    s.load_state_dict(sk["model_state"] if isinstance(sk, dict) and "model_state" in sk else sk)
    _seg_img_size = int(sk.get("img_size", IMG_SIZE)) if isinstance(sk, dict) else IMG_SIZE
    s.eval()

    cls_model, seg_model = c, s
    return c, s


def _blend_overlay(rgb: Image.Image, mask_01: np.ndarray, alpha: float = 0.42) -> Image.Image:
    """mask_01: HxW float/bool 0–1."""
    r = np.zeros((*mask_01.shape, 3), dtype=np.uint8)
    r[..., 0] = (mask_01 * 255).astype(np.uint8)
    ov = Image.fromarray(r, mode="RGB").resize(rgb.size, Image.Resampling.BILINEAR)
    return Image.blend(rgb, ov, alpha=alpha)


def predict_cvat_pil(pil_image: Image.Image) -> dict[str, Any]:
    global cls_model, seg_model
    if cls_model is None or seg_model is None:
        load_coco_models()

    assert cls_model is not None and seg_model is not None

    rgb = pil_image.convert("RGB")
    hidx = HEM_CLASS_IDX
    with torch.no_grad():
        xt = inference_cls_tensor(rgb, _cls_img_size).to(device)
        logits = cls_model(xt)
        probs = torch.softmax(logits, dim=1)[0]
        p_hem = float(probs[hidx].item())
        p_no = float(probs[1 - hidx].item())
        pred_cls = int(torch.argmax(probs).item())
        predicted = "hemorrhage" if pred_cls == hidx else "no_hemorrhage"
        confidence = float(probs[pred_cls].item() * 100.0)

        xs = inference_seg_tensors(rgb, _seg_img_size).to(device)
        seg_logits = seg_model(xs)
        sm = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
        mask_bin = sm >= 0.5
        coverage = float(mask_bin.mean())

    cls_seg_conflict = predicted == "hemorrhage" and coverage < 1e-4

    mask_u8 = (sm * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_u8, mode="L").resize(rgb.size, Image.Resampling.BILINEAR)
    mask_arr = np.asarray(mask_img, dtype=np.float32) / 255.0
    blended = _blend_overlay(rgb, mask_arr, alpha=0.45)
    buf = io.BytesIO()
    blended.save(buf, format="PNG", optimize=True)
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "predicted_class": predicted,
        "confidence_percent": round(confidence, 2),
        "hemorrhage_probability": round(p_hem, 6),
        "no_hemorrhage_probability": round(p_no, 6),
        "segmentation_mask_coverage": coverage,
        "cls_seg_conflict": cls_seg_conflict,
        "segmentation_overlay_png_base64": overlay_b64,
        "model_display_name": "ResNet18 + U-Net",
    }


def models_loaded() -> bool:
    return cls_model is not None and seg_model is not None
