"""
MyCNN — ImageFolder (train/val/test) ile ikili sınıflandırma.
Klasör: bu dosyanın bulunduğu dizinde split_dataset/ ve çıktı best_mycnn_v2.pth
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_ROOT = Path(__file__).resolve().parent
_DATA = _ROOT / "split_dataset"
_OUT = _ROOT / "best_mycnn_v2.pth"

IMG_SIZE = 224
BATCH_SIZE = 16
LR = 5e-5
EPOCHS = 20
PATIENCE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        x = self.features(x)
        return self.classifier(x)


def main() -> None:
    data_dir = Path(os.environ.get("MYCNN_DATA_DIR", str(_DATA)))
    model_save_path = Path(os.environ.get("MYCNN_OUT_PTH", str(_OUT)))
    if not (data_dir / "train").is_dir():
        print(f"[HATA] Veri yok: {data_dir / 'train'}")
        sys.exit(1)

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
        ]
    )
    val_test_transform = transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    )

    train_dataset = datasets.ImageFolder(str(data_dir / "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(data_dir / "val"), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(str(data_dir / "test"), transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = train_dataset.classes
    print("Cihaz:", device)
    print("Sınıflar:", class_names)
    print("Veri:", data_dir.resolve())
    print("Çıktı:", model_save_path.resolve())

    model = MyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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
                inputs, labels = inputs.to(device), labels.to(device)
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
            print(f"Early stopping: {early_stop_counter}/{PATIENCE}")
        if early_stop_counter >= PATIENCE:
            print("Early stopping.")
            break

    model.load_state_dict(best_model_wts)
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    # ImageFolder alfabetik: genelde hemorrhage=0, no_hemorrhage=1
    hem_idx = int(class_names.index("hemorrhage")) if "hemorrhage" in class_names else 0
    prec = precision_score(all_labels, all_preds, pos_label=hem_idx, average="binary", zero_division=0)
    rec = recall_score(all_labels, all_preds, pos_label=hem_idx, average="binary", zero_division=0)

    print("\nTEST")
    print("Accuracy :", acc)
    print("Precision (hemorrhage):", prec)
    print("Recall (hemorrhage):", rec)
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds, target_names=class_names))

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(_ROOT / "mycnn_train_loss.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_accuracies, label="Train Acc")
    plt.plot(val_accuracies, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(_ROOT / "mycnn_train_accuracy.png", dpi=120)
    plt.close()
    print("Grafikler:", _ROOT / "mycnn_train_loss.png")


if __name__ == "__main__":
    main()
