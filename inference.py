"""
Çıkarım katmanı — ResNet18 (pretrained API) ve MyCNN (custom API).
Ağırlıklar: NEUROSCAN_RESNET_PATH / NEUROSCAN_MYCNN_PATH veya proje kökündeki .pth dosyaları.
"""
from __future__ import annotations

import base64
import io
import math
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

_ROOT = Path(__file__).resolve().parent

RESNET_PATH = Path(
    os.environ.get("NEUROSCAN_RESNET_PATH", str(_ROOT / "resnet18_brain.pth"))
)
MYCNN_PATH = Path(
    os.environ.get("NEUROSCAN_MYCNN_PATH", str(_ROOT / "best_mycnn_v2.pth"))
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES: tuple[str, ...] = ("hemorrhage", "no_hemorrhage")
IMG_SIZE = 224

ApiModelId = Literal["pretrained", "custom"]


def api_id_to_model_key(api_model: ApiModelId) -> Literal["ResNet18", "MyCNN"]:
    return "ResNet18" if api_model == "pretrained" else "MyCNN"


resnet_model: nn.Module | None = None
mycnn_model: nn.Module | None = None


class MyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    if not state:
        return state
    if any(k.startswith("module.") for k in state):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def _state_dict_from_blob(blob: Any) -> dict[str, Any]:
    if isinstance(blob, dict):
        if "model_state" in blob:
            return _strip_module_prefix(blob["model_state"])
        if "state_dict" in blob:
            return _strip_module_prefix(blob["state_dict"])
    if isinstance(blob, dict) and all(isinstance(k, str) for k in blob):
        return _strip_module_prefix(blob)  # type: ignore[arg-type]
    raise ValueError("Checkpoint formatı tanınmadı.")


def build_resnet18_binary() -> models.ResNet:
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def load_resnet() -> None:
    global resnet_model
    resnet_model = None
    if not RESNET_PATH.is_file():
        return
    try:
        m = build_resnet18_binary()
        blob = _load_checkpoint(RESNET_PATH)
        state = _state_dict_from_blob(blob)
        m.load_state_dict(state, strict=True)
        resnet_model = m.to(device).eval()
    except Exception:
        resnet_model = None


def load_mycnn() -> None:
    global mycnn_model
    mycnn_model = None
    if not MYCNN_PATH.is_file():
        return
    try:
        m = MyCNN()
        blob = _load_checkpoint(MYCNN_PATH)
        state = blob["model_state"] if isinstance(blob, dict) and "model_state" in blob else blob
        if not isinstance(state, dict):
            state = _state_dict_from_blob(blob)
        else:
            state = _strip_module_prefix(state)
        m.load_state_dict(state, strict=True)
        mycnn_model = m.to(device).eval()
    except Exception:
        mycnn_model = None


def model_for_key(key: Literal["ResNet18", "MyCNN"]) -> nn.Module | None:
    if key == "ResNet18":
        return resnet_model
    if key == "MyCNN":
        return mycnn_model
    return None


_resnet_eval = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_mycnn_eval = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)


def _transform_for_key(key: Literal["ResNet18", "MyCNN"]):
    return _resnet_eval if key == "ResNet18" else _mycnn_eval


def wilson_ci(p: float, n: int = 80, z: float = 1.96) -> tuple[float, float]:
    """Wilson score aralığı; tek tahmin olasılığına karşılık etkin örneklem n ile yumuşatılır."""
    p = max(0.0, min(1.0, float(p)))
    if n <= 0:
        return p, p
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    inner = (p * (1.0 - p) / n) + (z2 / (4.0 * n * n))
    margin = (z / denom) * math.sqrt(max(inner, 0.0))
    return max(0.0, center - margin), min(1.0, center + margin)


def risk_level_from_probs(hemorrhage_probability: float) -> Literal["low", "medium", "high"]:
    h = float(hemorrhage_probability)
    if h >= 0.65:
        return "high"
    if h >= 0.35:
        return "medium"
    return "low"


def mycnn_inference_debug() -> dict[str, Any]:
    return {
        "weights_path": str(MYCNN_PATH),
        "exists": MYCNN_PATH.is_file(),
        "device": str(device),
    }


def predict_image_pil(
    pil: Image.Image,
    key: Literal["ResNet18", "MyCNN"],
    *,
    debug_raw_logits: bool = False,
) -> dict[str, Any]:
    model = model_for_key(key)
    if model is None:
        raise RuntimeError(f"Model yüklü değil: {key}")

    tfm = _transform_for_key(key)
    x = tfm(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)

    hem_p = float(probs[0].item())
    no_p = float(probs[1].item())
    pred_class = CLASS_NAMES[int(pred_idx.item())]
    lo, hi = wilson_ci(hem_p)

    out: dict[str, Any] = {
        "predicted_class": pred_class,
        "confidence_percent": float(conf.item() * 100.0),
        "hemorrhage_probability": hem_p,
        "no_hemorrhage_probability": no_p,
        "confidence_interval_low": lo,
        "confidence_interval_high": hi,
        "risk_level": risk_level_from_probs(hem_p),
    }
    if debug_raw_logits and key == "MyCNN":
        out["mycnn_debug"] = {
            "logits": logits[0].detach().cpu().tolist(),
            "softmax": probs.detach().cpu().tolist(),
        }
    return out


def _overlay_heatmap_on_pil(pil_rgb: Image.Image, saliency_hw: np.ndarray) -> bytes:
    """saliency_hw: [H,W] float 0..1"""
    pil_rgb = pil_rgb.convert("RGB")
    w, h = pil_rgb.size
    smap = np.clip(saliency_hw, 0.0, 1.0)
    smap = (smap * 255.0).astype(np.uint8)
    smap_img = Image.fromarray(smap).resize((w, h), Image.Resampling.BILINEAR)
    sal = np.asarray(smap_img).astype(np.float32) / 255.0
    orig = np.asarray(pil_rgb).astype(np.float32) / 255.0
    heat = np.zeros_like(orig)
    heat[:, :, 0] = sal
    heat[:, :, 1] = sal * 0.2
    alpha = 0.42
    blend = orig * (1.0 - alpha * sal[..., None]) + heat * (alpha * sal[..., None])
    blend = (np.clip(blend, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blend).save(buf, format="PNG")
    return buf.getvalue()


def grad_saliency_overlay_png(pil: Image.Image, key: Literal["ResNet18", "MyCNN"]) -> str | None:
    model = model_for_key(key)
    if model is None:
        return None
    pil_rgb = pil.convert("RGB")
    tfm = _transform_for_key(key)
    x = tfm(pil_rgb).unsqueeze(0).to(device).clone().detach().requires_grad_(True)
    model.eval()
    logits = model(x)
    pred = int(logits.argmax(dim=1).item())
    score = logits[0, pred]
    model.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad.detach_()
        x.grad.zero_()
    score.backward()
    if x.grad is None:
        return None
    g = x.grad[0]
    saliency = g.abs().mean(dim=0).detach().cpu().numpy()
    smin, smax = float(saliency.min()), float(saliency.max())
    if smax - smin < 1e-8:
        saliency = np.zeros_like(saliency)
    else:
        saliency = (saliency - smin) / (smax - smin)
    png = _overlay_heatmap_on_pil(pil_rgb, saliency)
    return base64.b64encode(png).decode("ascii")
