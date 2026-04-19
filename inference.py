"""
Paylaşılan beyin BT kanama sınıflandırma çıkarımı — masaüstü ve API için.

Eski best_resnet18.pth / best_mycnn_v2.pth: ImageFolder eğitimi ile uyumlu
  ön işleme (Resize + ToTensor, normalize yok) ve checkpoint’ten veya
  varsayılan olarak legacy sınıf indeksi (hemorrhage = 0).

Checkpoint dict içinde label_map varsa hemorrhage indeksi oradan okunur.
"""
from __future__ import annotations

import base64
import io
import math
import os
from typing import Any, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from class_labels import (
    class_name_from_index,
    load_state_dict_and_hemorrhage_index,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESNET_PATH = os.path.join(_SCRIPT_DIR, "best_resnet18.pth")
MYCNN_PATH = os.path.join(_SCRIPT_DIR, "best_mycnn_v2.pth")

IMG_SIZE = 224

ModelKey = Literal["ResNet18", "MyCNN"]
ApiModelId = Literal["pretrained", "custom"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train_pretrained.py val_test_transform ile aynı: Resize + ToTensor (RGB, [0,1])
legacy_eval_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)

resnet_model: nn.Module | None = None
mycnn_model: nn.Module | None = None
RESNET_HEM_IDX: int = 0
MYCNN_HEM_IDX: int = 0


def _state_dict_from_checkpoint(blob: Any) -> Any:
    if isinstance(blob, dict) and "model_state" in blob:
        return blob["model_state"]
    return blob


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
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

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_resnet() -> nn.Module:
    global resnet_model, RESNET_HEM_IDX
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    blob = torch.load(RESNET_PATH, map_location=device)
    state, RESNET_HEM_IDX = load_state_dict_and_hemorrhage_index(blob)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    resnet_model = model
    return model


def load_mycnn() -> nn.Module:
    global mycnn_model, MYCNN_HEM_IDX
    model = MyCNN()
    blob = torch.load(MYCNN_PATH, map_location=device)
    state, MYCNN_HEM_IDX = load_state_dict_and_hemorrhage_index(blob)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    mycnn_model = model
    return model


def api_id_to_model_key(api_id: ApiModelId) -> ModelKey:
    return "ResNet18" if api_id == "pretrained" else "MyCNN"


def model_for_key(name: ModelKey) -> nn.Module | None:
    return resnet_model if name == "ResNet18" else mycnn_model


def hemorrhage_index_for_model(key: ModelKey) -> int:
    return RESNET_HEM_IDX if key == "ResNet18" else MYCNN_HEM_IDX


def wilson_ci(p: float, n: int = 40, z: float = 1.96) -> Tuple[float, float]:
    """Wilson skor güven aralığı (olasılık için yaklaşık bant)."""
    p = max(1e-6, min(1.0 - 1e-6, p))
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def risk_level_from_probs(
    predicted_class: str, hemorrhage_prob: float, confidence: float
) -> Literal["low", "medium", "high"]:
    if predicted_class == "hemorrhage":
        return "high" if confidence >= 70 else "medium"
    if hemorrhage_prob >= 0.4:
        return "medium"
    if hemorrhage_prob >= 0.2:
        return "low"
    return "low"


def predict_tensor(
    model: nn.Module,
    selected_model_name: ModelKey,
    input_tensor: torch.Tensor,
    hem_idx: int,
) -> Tuple[str, float, float, float, dict[str, Any]]:
    """Karar: softmax argmax (eğitim / val ile aynı mantık; sabit threshold yok)."""
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    p_list = [float(probs[i].item()) for i in range(probs.shape[0])]
    pred = int(torch.argmax(probs).item())
    predicted_class = class_name_from_index(pred, hem_idx)
    hemorrhage_prob = float(probs[hem_idx].item())
    no_idx = 1 - hem_idx if probs.shape[0] == 2 else (1 if hem_idx == 0 else 0)
    no_hemorrhage_prob = float(probs[no_idx].item()) if probs.shape[0] == 2 else float("nan")
    confidence = float(probs[pred].item() * 100.0)

    debug = {
        "raw_class_probabilities": p_list,
        "predicted_class_index": pred,
        "hemorrhage_class_index": hem_idx,
        "threshold_used": None,
        "decision_rule": "argmax",
        "final_decision": predicted_class,
    }
    return predicted_class, confidence, hemorrhage_prob, no_hemorrhage_prob, debug


def predict_image_pil(pil_image: Image.Image, selected_model_name: ModelKey) -> dict[str, Any]:
    model = model_for_key(selected_model_name)
    if model is None:
        path = RESNET_PATH if selected_model_name == "ResNet18" else MYCNN_PATH
        raise RuntimeError(
            f"{selected_model_name} yüklenemedi. Ağırlık dosyası bekleniyor: {path}"
        )

    hem_idx = hemorrhage_index_for_model(selected_model_name)
    image = pil_image.convert("RGB")
    input_tensor = legacy_eval_transform(image).unsqueeze(0).to(device)
    predicted_class, confidence, hemorrhage_prob, no_hemorrhage_prob, dbg = predict_tensor(
        model, selected_model_name, input_tensor, hem_idx
    )
    lo, hi = wilson_ci(hemorrhage_prob)
    risk = risk_level_from_probs(predicted_class, hemorrhage_prob, confidence)

    out: dict[str, Any] = {
        "predicted_class": predicted_class,
        "confidence_percent": round(confidence, 2),
        "hemorrhage_probability": round(hemorrhage_prob, 6),
        "no_hemorrhage_probability": round(no_hemorrhage_prob, 6),
        "confidence_interval_low": round(lo, 4),
        "confidence_interval_high": round(hi, 4),
        "risk_level": risk,
        "debug": dbg,
    }
    return out


def predict_image_path(image_path: str, selected_model_name: ModelKey) -> dict[str, Any]:
    with Image.open(image_path) as img:
        return predict_image_pil(img, selected_model_name)


def predict_image(image_path: str, selected_model_name: str) -> Tuple[str, float, float, float]:
    """Masaüstü arayüz uyumluluğu: (class, conf%, hem_p, no_hem_p)."""
    key: ModelKey = "ResNet18" if selected_model_name == "ResNet18" else "MyCNN"
    out = predict_image_path(image_path, key)
    return (
        out["predicted_class"],
        out["confidence_percent"],
        out["hemorrhage_probability"],
        out["no_hemorrhage_probability"],
    )


def grad_saliency_overlay_png(pil_image: Image.Image, selected_model_name: ModelKey) -> str | None:
    """Gradyan önem haritası — predict ile aynı ön işleme (legacy_eval_transform)."""
    model = model_for_key(selected_model_name)
    if model is None:
        return None

    image = pil_image.convert("RGB")
    t = legacy_eval_transform(image).unsqueeze(0).to(device)
    t.requires_grad_(True)
    model.eval()
    out = model(t)
    target_idx = int(torch.argmax(out, dim=1).item())
    score = out[0, target_idx]
    model.zero_grad()
    if t.grad is not None:
        t.grad.zero_()
    score.backward()
    if t.grad is None:
        return None
    sal = t.grad.abs().max(dim=1)[0].squeeze().detach().cpu().numpy()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    r = (sal * 255).astype(np.uint8)
    b = ((1.0 - sal) * 255).astype(np.uint8)
    g = (np.minimum(r, b) * 0.35).astype(np.uint8)
    heat = np.stack([r, g, b], axis=-1)
    heat_img = Image.fromarray(heat, mode="RGB").resize(image.size, Image.Resampling.BILINEAR)
    blended = Image.blend(image, heat_img, alpha=0.45)
    buf = io.BytesIO()
    blended.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")
