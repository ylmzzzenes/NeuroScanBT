import os
import shutil
import random

random.seed(42)

source_dir = r"C:\Users\m42ay\Desktop\archive\dataset"
target_dir = r"C:\Users\m42ay\Desktop\archive\split_dataset"

classes = ["hemorrhage", "no_hemorrhage"]
splits = ["train", "val", "test"]

# Hedef klasörleri oluştur
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)

for cls in classes:
    class_path = os.path.join(source_dir, cls)
    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    for file in train_files:
        shutil.copy(
            os.path.join(class_path, file),
            os.path.join(target_dir, "train", cls, file)
        )

    for file in val_files:
        shutil.copy(
            os.path.join(class_path, file),
            os.path.join(target_dir, "val", cls, file)
        )

    for file in test_files:
        shutil.copy(
            os.path.join(class_path, file),
            os.path.join(target_dir, "test", cls, file)
        )

print("Train / Val / Test ayrımı tamamlandı.")