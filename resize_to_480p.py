import cv2
import glob
import os
import random

BASE = os.path.dirname(os.path.abspath(__file__))

SRC_DIRS = [
    os.path.join(BASE, "480p-Test/Originals/DIV2K_train_HR"),
    os.path.join(BASE, "480p-Test/Originals/DIV2K_valid_HR"),
]
DST      = os.path.join(BASE, "480p-Test/Originals_resized")
TARGET_W = 854
TARGET_H = 480
N_IMAGES = 900

os.makedirs(DST, exist_ok=True)

paths = []
for d in SRC_DIRS:
    paths += glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.PNG"))

paths = sorted(paths)
if len(paths) > N_IMAGES:
    random.seed(42)
    paths = random.sample(paths, N_IMAGES)

print(f"Bulunan görsel: {len(paths)}  →  {DST}")

for i, p in enumerate(paths):
    img = cv2.imread(p)
    if img is None:
        continue
    img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
    name = f"{i+1:04d}.png"
    cv2.imwrite(os.path.join(DST, name), img)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(paths)} tamamlandı")

print(f"Bitti. {len(os.listdir(DST))} görsel → {DST}")
