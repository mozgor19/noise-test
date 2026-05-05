import cv2
import os
import numpy as np

NOISE_LEVELS = [10, 30, 50, 70, 90]
METHODS = ['Noisy', 'SMF', 'AMF', 'MDBUTMF', 'EMPR', 'HDMR', 'DAMF', 'DnCNN', 'DnCNN-SP', 'SeConvUNet']

METHOD_DIRS = {
    'Noisy'     : '480p-Test/Noisy_{}',
    'SMF'       : '480p-Test/Cleaned_SMF_{}',
    'AMF'       : '480p-Test/Cleaned_AMF_{}',
    'MDBUTMF'   : '480p-Test/Cleaned_MDBUTMF_{}',
    'EMPR'      : '480p-Test/Cleaned_EMPR_{}',
    'HDMR'      : '480p-Test/Cleaned_HDMR_{}',
    'DAMF'      : '480p-Test/Cleaned_DAMF_{}',
    'DnCNN'     : '480p-Test/Cleaned_DnCNN_{}',
    'DnCNN-SP'  : '480p-Test/Cleaned_DnCNN_SP_{}',
    'SeConvUNet': '480p-Test/Cleaned_SeConvUNet_{}',
}

EXAMPLE_IMG = "0007.png"
OUT_DIR     = "docs/examples"
THUMB_W     = 320
THUMB_H     = 180

os.makedirs(OUT_DIR, exist_ok=True)

def thumb(img):
    return cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)

orig = cv2.imread(os.path.join("480p-Test/Originals_resized", EXAMPLE_IMG))

for method in METHODS:
    row_imgs = []
    for lv in NOISE_LEVELS:
        p = os.path.join(METHOD_DIRS[method].format(lv), EXAMPLE_IMG)
        img = cv2.imread(p) if os.path.exists(p) else np.zeros_like(orig)
        row_imgs.append(thumb(img))

    strip = np.hstack(row_imgs)
    out_path = os.path.join(OUT_DIR, f"{method}.png")
    cv2.imwrite(out_path, strip)
    print(f"Kaydedildi: {out_path}")

orig_path = os.path.join(OUT_DIR, "Original.png")
cv2.imwrite(orig_path, thumb(orig))
print(f"Kaydedildi: {orig_path}")
print("\nBitti. README için docs/examples/ klasörünü kullan.")
