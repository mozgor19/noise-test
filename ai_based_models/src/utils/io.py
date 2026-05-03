import os
import glob
import cv2
import numpy as np

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def list_images(folder: str, exts=(".png", ".jpg", ".jpeg", ".JPEG")):
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*{e}")))
    return sorted(paths)

def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def write_rgb(path: str, rgb_uint8: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, bgr)