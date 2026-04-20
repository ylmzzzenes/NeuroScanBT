"""
MyCNN — tek model (Tkinter). Ağırlık: NEUROSCAN_MYCNN_PATH veya ./best_mycnn_v2.pth
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
import tkinter as tk
from torchvision import transforms

_ROOT = Path(__file__).resolve().parent
_MYCNN = _ROOT / "best_mycnn_v2.pth"


def _resolve_weights() -> Path:
    env = os.environ.get("NEUROSCAN_MYCNN_PATH")
    if env:
        p = Path(os.path.normpath(os.path.expanduser(env.strip())))
        if p.is_file():
            return p
    if _MYCNN.is_file():
        return _MYCNN
    print(f"[HATA] Ağırlık yok: {_MYCNN} veya NEUROSCAN_MYCNN_PATH")
    sys.exit(1)


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


IMG_SIZE = 224
CLASS_NAMES = ["hemorrhage", "no_hemorrhage"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
)


def load_mycnn() -> nn.Module:
    wpath = _resolve_weights()
    model = MyCNN()
    try:
        blob = torch.load(str(wpath), map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(str(wpath), map_location=device)
    state = blob["model_state"] if isinstance(blob, dict) and "model_state" in blob else blob
    model.load_state_dict(state)
    return model.to(device).eval()


try:
    mycnn_model = load_mycnn()
    print("MyCNN yüklendi:", device)
except Exception as e:
    print("[HATA] Model:", e)
    sys.exit(1)

selected_image_path: str | None = None


def predict_image(image_path: str) -> tuple[str, float]:
    image = Image.open(image_path).convert("RGB")
    t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = mycnn_model(t)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
    return CLASS_NAMES[pred.item()], conf.item() * 100.0


def select_image() -> None:
    global selected_image_path
    file_path = filedialog.askopenfilename(
        title="Görüntü Seç",
        filetypes=[("Görüntü", "*.png *.jpg *.jpeg *.bmp")],
    )
    if not file_path:
        return
    selected_image_path = file_path
    img = Image.open(file_path).convert("RGB")
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_label.configure(image=img_tk)
    image_label.image = img_tk
    prediction_label.config(text="")
    confidence_label.config(text="")
    status_label.config(text="Model: MyCNN — Tahmin Et'e basın.")


def run_prediction() -> None:
    if not selected_image_path:
        messagebox.showwarning("Uyarı", "Önce görüntü seçin.")
        return
    try:
        status_label.config(text="Tahmin yapılıyor...")
        root.update()
        cls, conf = predict_image(selected_image_path)
        prediction_label.config(text=f"Tahmin: {cls}")
        confidence_label.config(text=f"Güven: %{conf:.2f}")
        status_label.config(text="MyCNN ile tamamlandı.")
    except Exception as e:
        messagebox.showerror("Hata", str(e))


root = tk.Tk()
root.title("Beyin BT — MyCNN")
root.geometry("560x640")
root.resizable(False, False)

tk.Label(root, text="Beyin BT — MyCNN", font=("Arial", 18, "bold")).pack(pady=12)
tk.Button(root, text="Görüntü Seç", command=select_image, width=22, height=2).pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

tk.Button(root, text="Tahmin Et", command=run_prediction, width=22, height=2).pack(pady=10)

prediction_label = tk.Label(root, text="", font=("Arial", 15, "bold"))
prediction_label.pack(pady=8)
confidence_label = tk.Label(root, text="", font=("Arial", 13))
confidence_label.pack(pady=6)
status_label = tk.Label(root, text="Görüntü seçilmedi.", wraplength=480, justify="center")
status_label.pack(pady=14)

root.mainloop()
