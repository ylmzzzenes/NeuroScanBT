"""
ResNet-18 ImageFolder eğitimi. Çıktılar: training_outputs/ altında PNG + metrik txt.
Veri yolu: proje kökünde split_dataset/ veya ortam değişkeni TRAIN_DATA_DIR.
"""
import os
import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ==============================
# YOLLAR
# ==============================
_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = _ROOT / "training_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

data_dir = os.environ.get("TRAIN_DATA_DIR", str(_ROOT / "split_dataset"))
model_save_path = str(_ROOT / "best_resnet18.pth")

IMG_SIZE = 224
BATCH_SIZE = 16
LR = 0.0001
EPOCHS = 15
PATIENCE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", device)
print("Veri klasörü:", data_dir)
print("Çıktı klasörü:", OUT_DIR)

# ==============================
# DATA AUGMENTATION + TRANSFORMS
# ==============================
train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ]
)

val_test_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)

# ==============================
# DATASET
# ==============================
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_test_transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes
print("Sınıflar:", class_names)

# ==============================
# PRETRAINED MODEL
# ==============================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model = model.to(device)

# ==============================
# LOSS / OPTIMIZER
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==============================
# EĞİTİM
# ==============================
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_loss = float("inf")
best_model_wts = copy.deepcopy(model.state_dict())
early_stop_counter = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 30)

    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_train += labels.size(0)

    epoch_train_loss = running_loss / total_train
    epoch_train_acc = running_corrects.double().item() / total_train

    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_val += labels.size(0)

    epoch_val_loss = running_loss / total_val
    epoch_val_acc = running_corrects.double().item() / total_val

    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
    print(f"Val   Loss: {epoch_val_loss:.4f} | Val   Acc: {epoch_val_acc:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        early_stop_counter = 0
        torch.save(model.state_dict(), model_save_path)
        print("En iyi model kaydedildi.")
    else:
        early_stop_counter += 1
        print(f"Early stopping sayacı: {early_stop_counter}/{PATIENCE}")

    if early_stop_counter >= PATIENCE:
        print("Early stopping tetiklendi.")
        break

model.load_state_dict(best_model_wts)

# ==============================
# TEST
# ==============================
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
# ImageFolder alfabetik: genelde 0=hemorrhage — ikili metrikler için pozitif sınıf 0
hemorrhage_idx = 0
prec = precision_score(
    all_labels, all_preds, average="binary", pos_label=hemorrhage_idx, zero_division=0
)
rec = recall_score(
    all_labels, all_preds, average="binary", pos_label=hemorrhage_idx, zero_division=0
)
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

print("\nTEST SONUÇLARI")
print("Accuracy :", acc)
print("Precision (hemorrhage):", prec)
print("Recall (hemorrhage):", rec)
print("\nConfusion Matrix:")
print(cm)

# ==============================
# PNG: eğitim eğrileri
# ==============================
prefix = "resnet18"
epochs_x = range(1, len(train_losses) + 1)

plt.figure(figsize=(9, 5))
plt.plot(epochs_x, train_losses, label="Train Loss", marker="o", markersize=3)
plt.plot(epochs_x, val_losses, label="Validation Loss", marker="s", markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet-18 — Train / Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / f"{prefix}_train_val_loss.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(epochs_x, train_accuracies, label="Train Accuracy", marker="o", markersize=3)
plt.plot(epochs_x, val_accuracies, label="Validation Accuracy", marker="s", markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet-18 — Train / Validation Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / f"{prefix}_train_val_accuracy.png", dpi=150)
plt.close()

# ==============================
# PNG: confusion matrix
# ==============================
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_ylabel("True label")
ax.set_xlabel("Predicted label")
ax.set_title("ResNet-18 — Test Confusion Matrix")
thresh = cm.max() / 2.0 if cm.size else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )
plt.tight_layout()
plt.savefig(OUT_DIR / f"{prefix}_confusion_matrix.png", dpi=150)
plt.close()

# ==============================
# TXT: metrikler
# ==============================
metrics_path = OUT_DIR / f"{prefix}_test_metrics.txt"
with metrics_path.open("w", encoding="utf-8") as f:
    f.write("Model: ResNet-18 (train_pretrained.py)\n")
    f.write(f"Classes (index order): {class_names}\n")
    f.write(f"Positive class for Precision/Recall: {class_names[hemorrhage_idx]} (index {hemorrhage_idx})\n\n")
    f.write(f"Accuracy:  {acc:.6f}\n")
    f.write(f"Precision (hemorrhage): {prec:.6f}\n")
    f.write(f"Recall (hemorrhage):    {rec:.6f}\n\n")
    f.write("Confusion matrix [rows=true, cols=pred]:\n")
    f.write(np.array2string(cm))
    f.write("\n\n")
    f.write(classification_report(all_labels, all_preds, target_names=class_names))

print(f"\nKaydedildi: {OUT_DIR / (prefix + '_train_val_loss.png')}")
print(f"Kaydedildi: {OUT_DIR / (prefix + '_train_val_accuracy.png')}")
print(f"Kaydedildi: {OUT_DIR / (prefix + '_confusion_matrix.png')}")
print(f"Kaydedildi: {metrics_path}")
