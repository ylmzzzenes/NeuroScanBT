"""
Paylaşılan beyin BT kanama sınıflandırma çıkarımı — masaüstü ve API için.
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESNET_PATH = os.path.join(_SCRIPT_DIR, "best_resnet18.pth")
MYCNN_PATH = os.path.join(_SCRIPT_DIR, "best_mycnn_v2.pth")

IMG_SIZE = 224
CLASS_NAMES = ["hemorrhage", "no_hemorrhage"]
HEMORRHAGE_THRESHOLD = 0.45

ModelKey = Literal["ResNet18", "MyCNN"]
ApiModelId = Literal["pretrained", "custom", "cvat"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)

resnet_model: nn.Module | None = None
mycnn_model: nn.Module | None = None

# ImageFolder (alfabetik): 0 = hemorrhage, 1 = no_hemorrhage
HEMORRHAGE_CLASS_IDX = 0


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
    global resnet_model
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    blob = torch.load(RESNET_PATH, map_location=device)
    model.load_state_dict(_state_dict_from_checkpoint(blob))
    model = model.to(device)
    model.eval()
    resnet_model = model
    return model


def load_mycnn() -> nn.Module:
    global mycnn_model
    model = MyCNN()
    blob = torch.load(MYCNN_PATH, map_location=device)
    model.load_state_dict(_state_dict_from_checkpoint(blob))
    model = model.to(device)
    model.eval()
    mycnn_model = model
    return model


def api_id_to_model_key(api_id: ApiModelId) -> ModelKey:
    return "ResNet18" if api_id == "pretrained" else "MyCNN"


def model_for_key(name: ModelKey) -> nn.Module | None:
    return resnet_model if name == "ResNet18" else mycnn_model


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
    model: nn.Module, selected_model_name: ModelKey, input_tensor: torch.Tensor
) -> Tuple[str, float, float, float]:
    hem_idx = HEMORRHAGE_CLASS_IDX
    no_idx = 1 - hem_idx

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    hemorrhage_prob = probs[hem_idx].item()
    no_hemorrhage_prob = probs[no_idx].item()

    if selected_model_name == "ResNet18":
        pred = int(torch.argmax(probs).item())
        predicted_class = "hemorrhage" if pred == hem_idx else "no_hemorrhage"
        confidence = probs[pred].item() * 100
    else:
        if hemorrhage_prob >= HEMORRHAGE_THRESHOLD:
            predicted_class = "hemorrhage"
            confidence = hemorrhage_prob * 100
        else:
            predicted_class = "no_hemorrhage"
            confidence = no_hemorrhage_prob * 100

    return predicted_class, confidence, hemorrhage_prob, no_hemorrhage_prob


def predict_image_pil(pil_image: Image.Image, selected_model_name: ModelKey) -> dict[str, Any]:
    model = model_for_key(selected_model_name)
    if model is None:
        path = RESNET_PATH if selected_model_name == "ResNet18" else MYCNN_PATH
        raise RuntimeError(
            f"{selected_model_name} yüklenemedi. Ağırlık dosyası bekleniyor: {path}"
        )

    image = pil_image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    predicted_class, confidence, hemorrhage_prob, no_hemorrhage_prob = predict_tensor(
        model, selected_model_name, input_tensor
    )
    lo, hi = wilson_ci(hemorrhage_prob)
    risk = risk_level_from_probs(predicted_class, hemorrhage_prob, confidence)

    return {
        "predicted_class": predicted_class,
        "confidence_percent": round(confidence, 2),
        "hemorrhage_probability": round(hemorrhage_prob, 6),
        "no_hemorrhage_probability": round(no_hemorrhage_prob, 6),
        "confidence_interval_low": round(lo, 4),
        "confidence_interval_high": round(hi, 4),
        "risk_level": risk,
    }


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
    """Basit gradyan önem haritası — orijinal üzerine renkli bindirme, base64 PNG."""
    model = model_for_key(selected_model_name)
    if model is None:
        return None

    image = pil_image.convert("RGB")
    small = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    t = transforms.ToTensor()(small).unsqueeze(0).to(device)
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
