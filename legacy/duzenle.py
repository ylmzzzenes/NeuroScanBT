import os
import shutil
import pandas as pd

csv_path = r"C:\Users\m42ay\Desktop\archive\labels.csv"
image_folder = r"C:\Users\m42ay\Desktop\archive\head_ct\head_ct"
output_folder = r"C:\Users\m42ay\Desktop\archive\dataset"

df = pd.read_csv(csv_path)

# Sütun adlarındaki boşlukları temizle
df.columns = df.columns.str.strip()

hem_folder = os.path.join(output_folder, "hemorrhage")
nohem_folder = os.path.join(output_folder, "no_hemorrhage")

os.makedirs(hem_folder, exist_ok=True)
os.makedirs(nohem_folder, exist_ok=True)

for _, row in df.iterrows():
    img_id = int(row["id"])
    label = int(row["hemorrhage"])

    filename = f"{img_id:03d}.png"
    src = os.path.join(image_folder, filename)

    if label == 1:
        dst = os.path.join(hem_folder, filename)
    else:
        dst = os.path.join(nohem_folder, filename)

    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print("Bulunamadı:", src)

print("Bitti.")

import os

hem_path = r"C:\Users\m42ay\Desktop\archive\dataset\hemorrhage"
nohem_path = r"C:\Users\m42ay\Desktop\archive\dataset\no_hemorrhage"

print("hemorrhage:", len(os.listdir(hem_path)))
print("no_hemorrhage:", len(os.listdir(nohem_path)))