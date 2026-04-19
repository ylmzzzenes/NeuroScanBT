import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import inference
from inference import load_mycnn, load_resnet, predict_image

# =========================
# AYARLAR (arayüz için)
# =========================
RESNET_PATH = inference.RESNET_PATH
MYCNN_PATH = inference.MYCNN_PATH

selected_image_path = None

# =========================
# GÖRÜNTÜ SEÇ
# =========================
def select_image():
    global selected_image_path

    file_path = filedialog.askopenfilename(
        title="Görüntü Seç",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")],
    )

    if file_path:
        selected_image_path = file_path

        img = Image.open(file_path).convert("RGB")
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)

        image_label.config(image=img_tk)
        image_label.image = img_tk

        prediction_label.config(text="")
        confidence_label.config(text="")
        prob_label.config(text="")
        status_label.config(text="Görüntü seçildi. Model seçip Tahmin Et butonuna bas.")

# =========================
# TAHMİN ET
# =========================
def run_prediction():
    if not selected_image_path:
        messagebox.showwarning("Uyarı", "Önce bir görüntü seç.")
        return

    try:
        selected_model_name = model_var.get()

        status_label.config(text="Tahmin yapılıyor...")
        root.update()

        predicted_class, confidence, hemorrhage_prob, no_hemorrhage_prob = predict_image(
            selected_image_path, selected_model_name
        )

        prediction_label.config(text=f"Tahmin: {predicted_class}")
        confidence_label.config(text=f"Güven: %{confidence:.2f}")
        prob_label.config(
            text=f"Hemorrhage: %{hemorrhage_prob*100:.2f} | No Hemorrhage: %{no_hemorrhage_prob*100:.2f}"
        )
        status_label.config(text=f"Tahmin tamamlandı. Kullanılan model: {selected_model_name}")

        root.update()

    except Exception as e:
        messagebox.showerror("Hata", str(e))
        print("HATA:", e)

# =========================
# ARAYÜZ
# =========================
root = tk.Tk()
root.title("Beyin BT Görüntü Sınıflandırma")
root.geometry("650x760")
root.resizable(False, False)

title_label = tk.Label(
    root,
    text="Beyin BT Görüntü Sınıflandırma",
    font=("Arial", 22, "bold"),
)
title_label.pack(pady=15)

subtitle_label = tk.Label(
    root,
    text="ResNet18 ve MyCNN Karşılaştırmalı Arayüz",
    font=("Arial", 12),
)
subtitle_label.pack(pady=5)

model_var = tk.StringVar(value="ResNet18")

model_frame = tk.Frame(root)
model_frame.pack(pady=10)

model_text = tk.Label(model_frame, text="Model Seç:", font=("Arial", 14, "bold"))
model_text.pack(side=tk.LEFT, padx=5)

model_menu = tk.OptionMenu(model_frame, model_var, "ResNet18", "MyCNN")
model_menu.config(width=15, font=("Arial", 12))
model_menu.pack(side=tk.LEFT, padx=5)

btn_select = tk.Button(
    root,
    text="Görüntü Seç",
    command=select_image,
    width=18,
    height=2,
    font=("Arial", 12),
)
btn_select.pack(pady=15)

image_label = tk.Label(root)
image_label.pack(pady=10)

btn_predict = tk.Button(
    root,
    text="Tahmin Et",
    command=run_prediction,
    width=18,
    height=2,
    font=("Arial", 12),
)
btn_predict.pack(pady=15)

prediction_label = tk.Label(
    root,
    text="",
    font=("Arial", 18, "bold"),
)
prediction_label.pack(pady=8)

confidence_label = tk.Label(
    root,
    text="",
    font=("Arial", 14),
)
confidence_label.pack(pady=5)

prob_label = tk.Label(
    root,
    text="",
    font=("Arial", 12),
    wraplength=550,
    justify="center",
)
prob_label.pack(pady=5)

status_label = tk.Label(
    root,
    text="Modeller yükleniyor...",
    font=("Arial", 11),
    wraplength=550,
    justify="center",
)
status_label.pack(pady=20)

_load_errors = []
try:
    load_resnet()
    print("ResNet18 yüklendi.")
except Exception as e:
    inference.resnet_model = None
    _load_errors.append(f"ResNet18: {e}")
    print("MODEL YÜKLEME HATASI (ResNet18):", e)

try:
    load_mycnn()
    print("MyCNN yüklendi.")
except Exception as e:
    inference.mycnn_model = None
    _load_errors.append(f"MyCNN: {e}")
    print("MODEL YÜKLEME HATASI (MyCNN):", e)

if inference.resnet_model is not None and inference.mycnn_model is not None:
    status_label.config(text="Modeller başarıyla yüklendi. Görüntü seçebilirsin.")
elif inference.resnet_model is not None or inference.mycnn_model is not None:
    ok = "ResNet18" if inference.resnet_model is not None else "MyCNN"
    status_label.config(
        text=f"Sadece {ok} yüklendi. Diğer model için .pth dosyasını bu klasöre koyun."
    )
    messagebox.showwarning(
        "Kısmi model yükleme",
        "\n".join(_load_errors),
    )
else:
    status_label.config(
        text="Hiçbir model yüklenemedi. .pth dosyalarını script ile aynı klasöre koyun."
    )
    messagebox.showerror("Model Yükleme Hatası", "\n".join(_load_errors))

root.mainloop()
