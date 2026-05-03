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
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr

def sp_mask_fixed(noisy_rgb_uint8: np.ndarray, median_ksize: int = 3, diff_threshold: int = 20) -> np.ndarray:
    """
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

def mask_to_tensor(mask_hw1: np.ndarray) -> torch.Tensor:
    # (H,W,1) -> (1,H,W)
    return torch.from_numpy(mask_hw1.transpose(2, 0, 1)).float()
