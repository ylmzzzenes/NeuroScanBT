"""ResNet-18 ImageFolder eğitimi → kök: best_resnet18.pth, grafikler: ../training_outputs/"""
import os
import copy
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

_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = _ROOT / "training_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
data_dir = os.environ.get("TRAIN_DATA_DIR", str(_ROOT / "split_dataset"))
model_save_path = str(_ROOT / "best_resnet18.pth")

IMG_SIZE, BATCH_SIZE, LR, EPOCHS, PATIENCE = 224, 16, 0.0001, 15, 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, data_dir, OUT_DIR)

train_tf = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ]
)
val_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tf)
test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
class_names = train_ds.classes
print("Sınıflar:", class_names)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val, best_wts = float("inf"), copy.deepcopy(model.state_dict())
early = 0

for epoch in range(EPOCHS):
    model.train()
    tl, tc, tt = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        pred = out.argmax(1)
        tl += loss.item() * x.size(0)
        tc += (pred == y).sum().item()
        tt += y.size(0)
    train_losses.append(tl / tt)
    train_accs.append(tc / tt)

    model.eval()
    vl, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            pred = out.argmax(1)
            vl += loss.item() * x.size(0)
            vc += (pred == y).sum().item()
            vt += y.size(0)
    val_losses.append(vl / vt)
    val_accs.append(vc / vt)
    print(f"Ep{epoch+1} trainL={train_losses[-1]:.4f} valL={val_losses[-1]:.4f}")

    if val_losses[-1] < best_val:
        best_val, early = val_losses[-1], 0
        best_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), model_save_path)
    else:
        early += 1
        if early >= PATIENCE:
            break

model.load_state_dict(best_wts)
model.eval()
labels, preds = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        labels.extend(y.numpy())
        preds.extend(model(x).argmax(1).cpu().numpy())

acc = accuracy_score(labels, preds)
hi = 0
prec = precision_score(labels, preds, average="binary", pos_label=hi, zero_division=0)
rec = recall_score(labels, preds, average="binary", pos_label=hi, zero_division=0)
cm = confusion_matrix(labels, preds, labels=[0, 1])
print("TEST acc", acc, "prec", prec, "rec", rec, "\ncm\n", cm)

pfx, ep = "resnet18", range(1, len(train_losses) + 1)
for name, a, b, ylab, ttl in (
    ("_train_val_loss.png", train_losses, val_losses, "Loss", "ResNet-18 — Loss"),
    ("_train_val_accuracy.png", train_accs, val_accs, "Accuracy", "ResNet-18 — Accuracy"),
):
    plt.figure(figsize=(9, 5))
    plt.plot(ep, a, label="Train", marker="o", markersize=3)
    plt.plot(ep, b, label="Val", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel(ylab)
    plt.title(ttl)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if "Accuracy" in ttl:
        plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{pfx}{name}", dpi=150)
    plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
th = cm.max() / 2.0 if cm.size else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="w" if cm[i, j] > th else "k")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("ResNet-18 — Confusion (test)")
plt.tight_layout()
plt.savefig(OUT_DIR / f"{pfx}_confusion_matrix.png", dpi=150)
plt.close()

mp = OUT_DIR / f"{pfx}_test_metrics.txt"
with mp.open("w", encoding="utf-8") as f:
    f.write("ResNet-18 (training/train_resnet18.py)\n")
    f.write(f"Classes: {class_names}\n\nAccuracy: {acc:.6f}\nPrecision (hem): {prec:.6f}\nRecall (hem): {rec:.6f}\n\n")
    f.write(np.array2string(cm))
    f.write("\n\n")
    f.write(classification_report(labels, preds, target_names=class_names))
print("Kayıt:", OUT_DIR)
