import cv2
import numpy as np
import torch

def to_tensor01(rgb_uint8: np.ndarray) -> torch.Tensor:
    # (H,W,C) uint8 -> (C,H,W) float32 [0,1]
    return torch.from_numpy(rgb_uint8.transpose(2, 0, 1)).float() / 255.0

def to_uint8(rgb_tensor01: torch.Tensor) -> np.ndarray:
    # (1,C,H,W) or (C,H,W) -> (H,W,C) uint8
    if rgb_tensor01.dim() == 4:
        rgb_tensor01 = rgb_tensor01.squeeze(0)
    arr = rgb_tensor01.detach().cpu().numpy().transpose(1, 2, 0)
    arr = np.rint(arr * 255.0).clip(0, 255).astype(np.uint8)  # kesme değil yuvarlama
    return arr

def add_sp_noise(clean_uint8: np.ndarray, density: float, rng=np.random) -> np.ndarray:
    """
    MATLAB imnoise(I, 'salt & pepper', d) ile aynı model: her kanal elemanı
    bağımsız olarak u < d/2 ise 0 (biber), d/2 <= u < d ise 255 (tuz) olur.
    """
    u = rng.random(clean_uint8.shape)
    noisy = clean_uint8.copy()
    noisy[u < density / 2] = 0
    noisy[(u >= density / 2) & (u < density)] = 255
    return noisy

def sp_mask_channel(noisy_uint8: np.ndarray) -> np.ndarray:
    """
    Kanal bazlı kör S&P maskesi: değeri 0 veya 255 olan her kanal elemanı şüpheli.
    Klasik karar-tabanlı filtrelerle (MDBUTMF, DAMF, EMPR, HDMR) aynı kural.

    Output: (H,W,C) float32, 1=şüpheli, 0=güvenilir
    """
    return ((noisy_uint8 == 0) | (noisy_uint8 == 255)).astype(np.float32)

def sp_mask_fixed(noisy_rgb_uint8: np.ndarray, median_ksize: int = 3, diff_threshold: int = 20) -> np.ndarray:
    """
    ESKİ (v2) piksel bazlı maske — yalnızca eski teşhis betikleri için tutuluyor.
    Kanal bazlı imnoise gürültüsünü kaçırır; yeni kod sp_mask_channel kullanır.

    Heuristic salt-and-pepper mask.

    Marks a pixel as corrupted only if:
    - it is an RGB extreme (all 0 or all 255), and
    - it disagrees with the local median enough to look like an impulse.

    Output: (H,W,1) float32, 1=corrupted, 0=clean
    """
    is0 = (noisy_rgb_uint8 == 0).all(axis=2)
    is255 = (noisy_rgb_uint8 == 255).all(axis=2)
    extreme = is0 | is255

    if median_ksize < 3 or median_ksize % 2 == 0:
        raise ValueError("median_ksize must be an odd integer >= 3")

    median = np.stack(
        [cv2.medianBlur(noisy_rgb_uint8[..., c], median_ksize) for c in range(noisy_rgb_uint8.shape[2])],
        axis=2,
    )
    diff = np.abs(noisy_rgb_uint8.astype(np.int16) - median.astype(np.int16)).mean(axis=2)
    m = (extreme & (diff >= diff_threshold)).astype(np.float32)[..., None]
    return m

def mask_to_tensor(mask_hwc: np.ndarray) -> torch.Tensor:
    # (H,W,C) -> (C,H,W)
    return torch.from_numpy(mask_hwc.transpose(2, 0, 1)).float()
