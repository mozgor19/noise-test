"""
Teşhis 1: DnCNN-SP inferans maskesi bozuk mu?
Teşhis 2: MATLAB imnoise kanal davranışı
"""

import os, sys, glob, time
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG_DIR  = os.path.dirname(os.path.abspath(__file__))
AI_SRC    = os.path.join(ROOT, "ai_based_models", "src")
sys.path.insert(0, AI_SRC)

from utils.spmask import sp_mask_fixed, to_tensor01, to_uint8, mask_to_tensor
from utils.io import read_rgb, write_rgb

import torch
from models.dncnn_sp import DnCNNSP
from models.seconv_unet import SeConvUNet
from models.external_wrappers import load_checkpoint

T0 = time.time()

# ──────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ──────────────────────────────────────────────────────────────────────────────

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))

def gt_mask(noisy, clean, detect_thr=5):
    """Ground-truth S&P maskesi: bozuk piksel = uçta VE orijinalden >detect_thr farklı."""
    is_extreme = np.all(noisy == 0, axis=2) | np.all(noisy == 255, axis=2)
    diff_any   = np.any(np.abs(noisy.astype(np.int32) - clean.astype(np.int32)) > detect_thr, axis=2)
    return (is_extreme & diff_any).astype(np.float32)[:, :, None]  # (H,W,1)

def mask_stats(heuristic_hw1, gt_hw1):
    h = heuristic_hw1[:,:,0].astype(bool)
    g = gt_hw1[:,:,0].astype(bool)
    TP = (h & g).sum()
    FP = (h & ~g).sum()
    FN = (~h & g).sum()
    total_gt = g.sum()
    coverage  = TP / total_gt if total_gt > 0 else 0.0    # recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    fill_rate = h.mean()           # maskede 1'lerin oranı (ne kadar dolu?)
    gt_rate   = g.mean()           # gerçekte ne kadar piksel bozuk?
    return dict(TP=int(TP), FP=int(FP), FN=int(FN),
                recall=float(coverage), precision=float(precision),
                fill_rate=float(fill_rate), gt_rate=float(gt_rate))

def run_dncnn_sp(model, rgb_np, mask_hw1, device):
    """Verilen maske ile DnCNN-SP inferansı → (H,W,3) uint8"""
    x   = to_tensor01(rgb_np).unsqueeze(0).to(device)   # (1,3,H,W)
    m_t = mask_to_tensor(mask_hw1.astype(np.float32)).unsqueeze(0).to(device)  # (1,1,H,W)
    x4  = torch.cat([x, m_t], dim=1)                    # (1,4,H,W)
    with torch.no_grad():
        y = model(x4).clamp(0, 1)
    return to_uint8(y)

def run_seconv(model, rgb_np, mask_hw1, device):
    """Verilen maske ile SeConvUNet inferansı (8'in katı padding)"""
    x   = to_tensor01(rgb_np).unsqueeze(0).to(device)
    m_t = mask_to_tensor(mask_hw1.astype(np.float32)).unsqueeze(0).to(device)
    x4  = torch.cat([x, m_t], dim=1)
    _, _, H, W = x4.shape
    ph = (8 - H % 8) % 8
    pw = (8 - W % 8) % 8
    if ph or pw:
        x4 = torch.nn.functional.pad(x4, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        y = model(x4).clamp(0, 1)
    if ph or pw:
        y = y[:, :, :H, :W]
    return to_uint8(y)

# ──────────────────────────────────────────────────────────────────────────────
# Örnek görselleri seç
# ──────────────────────────────────────────────────────────────────────────────
ORIG_DIR  = os.path.join(ROOT, "Tiny-Imagenet", "Tiny_Orijinal")
orig_files = sorted(glob.glob(os.path.join(ORIG_DIR, "*.JPEG")))[:8]  # ilk 8
stems = [os.path.splitext(os.path.basename(p))[0] for p in orig_files]
print(f"Seçilen görsel: {stems}")

# ──────────────────────────────────────────────────────────────────────────────
# Model yükle
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ckpt_dncnn_sp = os.path.join(ROOT, "ai_based_models", "weights", "dncnn_sp_v2.pth")
model_dncnn   = DnCNNSP(channels=3).to(device)
model_dncnn   = load_checkpoint(model_dncnn, ckpt_dncnn_sp, device, strict=False)
model_dncnn.eval()
print(f"DnCNN-SP yüklendi: {ckpt_dncnn_sp}")

ckpt_seconv   = os.path.join(ROOT, "ai_based_models", "weights", "seconv_unet_sp_v2.pth")
model_seconv  = SeConvUNet(base=48).to(device)
model_seconv  = load_checkpoint(model_seconv, ckpt_seconv, device, strict=False)
model_seconv.eval()
print(f"SeConvUNet yüklendi: {ckpt_seconv}")

# ──────────────────────────────────────────────────────────────────────────────
# TEŞHİS 1
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TEŞHİS 1 — DnCNN-SP inferans maskesi kalitesi")
print("="*70)

results = []

for level in [30, 50]:
    NOISY_DIR = os.path.join(ROOT, "Tiny-Imagenet", f"Tiny_Noisy_{level}")
    print(f"\n--- Seviye %{level} ---")

    for stem in stems:
        orig_path  = os.path.join(ORIG_DIR, stem + ".JPEG")
        noisy_path = os.path.join(NOISY_DIR, stem + ".png")
        if not os.path.exists(noisy_path):
            continue

        clean = read_rgb(orig_path)
        noisy = read_rgb(noisy_path)

        # Maskeler
        mask_h = sp_mask_fixed(noisy)           # heuristic (H,W,1)
        mask_g = gt_mask(noisy, clean)           # ground-truth (H,W,1)

        stats = mask_stats(mask_h, mask_g)

        # DnCNN-SP inferansı (her iki maske)
        out_h = run_dncnn_sp(model_dncnn, noisy, mask_h, device)
        out_g = run_dncnn_sp(model_dncnn, noisy, mask_g, device)

        # SeConvUNet
        out_sh = run_seconv(model_seconv, noisy, mask_h, device)
        out_sg = run_seconv(model_seconv, noisy, mask_g, device)

        psnr_noisy  = psnr(noisy, clean)
        psnr_dsp_h  = psnr(out_h, clean)
        psnr_dsp_g  = psnr(out_g, clean)
        psnr_sc_h   = psnr(out_sh, clean)
        psnr_sc_g   = psnr(out_sg, clean)

        results.append(dict(
            level=level, stem=stem,
            gt_rate=stats["gt_rate"], fill_rate=stats["fill_rate"],
            mask_recall=stats["recall"], mask_precision=stats["precision"],
            psnr_noisy=psnr_noisy,
            psnr_dsp_heuristic=psnr_dsp_h, psnr_dsp_gt=psnr_dsp_g,
            psnr_sc_heuristic=psnr_sc_h, psnr_sc_gt=psnr_sc_g,
        ))

        print(f"  {stem} | GT_rate={stats['gt_rate']:.3f}  fill={stats['fill_rate']:.3f} "
              f"| mask_recall={stats['recall']:.3f} mask_prec={stats['precision']:.3f} "
              f"| PSNR noisy={psnr_noisy:.1f} "
              f"dsp_h={psnr_dsp_h:.1f} dsp_gt={psnr_dsp_g:.1f} "
              f"sc_h={psnr_sc_h:.1f} sc_gt={psnr_sc_g:.1f}")

        # Görselleştirme
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        titles = ["Orijinal", "Gürültülü", "GT Maske", "Heuristic Maske",
                  "DnCNN-SP (heuristic)", "DnCNN-SP (GT maske)",
                  "SeConvUNet (heuristic)", "SeConvUNet (GT maske)"]
        imgs   = [clean, noisy,
                  (mask_g[:,:,0]*255).astype(np.uint8),
                  (mask_h[:,:,0]*255).astype(np.uint8),
                  out_h, out_g, out_sh, out_sg]
        cmaps  = [None, None, "gray", "gray", None, None, None, None]
        for ax, img, title, cmap in zip(axes.flat, imgs, titles, cmaps):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        psnr_info = (f"Level %{level} | {stem}\n"
                     f"DnCNN-SP: heuristic={psnr_dsp_h:.2f}dB  GT={psnr_dsp_g:.2f}dB\n"
                     f"SeConvUNet: heuristic={psnr_sc_h:.2f}dB  GT={psnr_sc_g:.2f}dB\n"
                     f"Mask recall={stats['recall']:.3f} prec={stats['precision']:.3f} "
                     f"fill={stats['fill_rate']:.3f}")
        fig.suptitle(psnr_info, fontsize=9)
        plt.tight_layout()
        out_png = os.path.join(DIAG_DIR, f"diag1_{stem}_lv{level}.png")
        plt.savefig(out_png, dpi=100, bbox_inches="tight")
        plt.close()

# Özet
import statistics as st
print("\n--- TEŞHİS 1 ÖZET ---")
for level in [30, 50]:
    sub = [r for r in results if r["level"] == level]
    if not sub: continue
    def avg(key): return sum(r[key] for r in sub) / len(sub)

    print(f"\n%{level} gürültü ({len(sub)} görsel):")
    print(f"  GT_rate (gerçek bozuk piksel oranı)    : {avg('gt_rate'):.4f}")
    print(f"  fill_rate (heuristic maske doluluğu)   : {avg('fill_rate'):.4f}")
    print(f"  mask_recall (heuristic → GT yakalama)  : {avg('mask_recall'):.4f}")
    print(f"  mask_precision                         : {avg('mask_precision'):.4f}")
    print(f"  PSNR noisy                             : {avg('psnr_noisy'):.2f} dB")
    print(f"  PSNR DnCNN-SP (heuristic maske)        : {avg('psnr_dsp_heuristic'):.2f} dB")
    print(f"  PSNR DnCNN-SP (GT maske)               : {avg('psnr_dsp_gt'):.2f} dB")
    print(f"  PSNR SeConvUNet (heuristic maske)      : {avg('psnr_sc_heuristic'):.2f} dB")
    print(f"  PSNR SeConvUNet (GT maske)             : {avg('psnr_sc_gt'):.2f} dB")

    dsp_gain_h = avg("psnr_dsp_heuristic") - avg("psnr_noisy")
    dsp_gain_g = avg("psnr_dsp_gt") - avg("psnr_noisy")
    print(f"\n  DnCNN-SP kazanımı  heuristic: {dsp_gain_h:+.2f} dB | GT: {dsp_gain_g:+.2f} dB")
    if dsp_gain_g > dsp_gain_h + 2:
        print(f"  HÜKÜM: GT maske {dsp_gain_g - dsp_gain_h:.1f} dB daha iyi → SUÇLU MASKE kalitesi")
    elif dsp_gain_g < 3:
        print(f"  HÜKÜM: GT maskesiyle de düşük → SUÇLU EĞİTİM kalitesi, maske değil")
    else:
        print(f"  HÜKÜM: Karma — hem maske hem eğitim katkılı")

# ──────────────────────────────────────────────────────────────────────────────
# TEŞHİS 2 — imnoise kanal davranışı
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TEŞHİS 2 — MATLAB imnoise kanal davranışı")
print("="*70)

for level in [30, 50]:
    NOISY_DIR = os.path.join(ROOT, "Tiny-Imagenet", f"Tiny_Noisy_{level}")
    total_corrupted = 0
    fully_extreme   = 0  # tüm 3 kanal 0 veya 255
    partial_extreme = 0  # 1 ya da 2 kanal 0/255, ötekiler değil

    for stem in stems:
        orig_path  = os.path.join(ORIG_DIR, stem + ".JPEG")
        noisy_path = os.path.join(NOISY_DIR, stem + ".png")
        if not os.path.exists(noisy_path):
            continue
        clean = read_rgb(orig_path)
        noisy = read_rgb(noisy_path)

        # Bozuk piksel: herhangi bir kanalda orijinalden >5 farklı
        diff = np.abs(noisy.astype(np.int32) - clean.astype(np.int32))
        corrupted_mask = np.any(diff > 5, axis=2)   # (H,W) bool

        corrupted_pixels = noisy[corrupted_mask]     # (N,3)
        N = len(corrupted_pixels)
        if N == 0:
            continue

        # Her piksel için her kanalın "uç" olup olmadığını kontrol et
        ch_extreme = (corrupted_pixels == 0) | (corrupted_pixels == 255)  # (N,3) bool

        fully   = np.all(ch_extreme, axis=1).sum()     # her 3 kanal uçta
        partial = (~np.all(ch_extreme, axis=1) & np.any(ch_extreme, axis=1)).sum()

        total_corrupted += N
        fully_extreme   += int(fully)
        partial_extreme += int(partial)

    if total_corrupted > 0:
        pct_full    = 100 * fully_extreme   / total_corrupted
        pct_partial = 100 * partial_extreme / total_corrupted
        pct_other   = 100 - pct_full - pct_partial
        print(f"\n%{level} gürültü — {total_corrupted} bozuk piksel ({len(stems)} görsel):")
        print(f"  Tam uçta (tüm 3 kanal 0 veya 255)  : {fully_extreme:7d}  ({pct_full:.2f}%)")
        print(f"  Kısmi uçta (1-2 kanal uçta)        : {partial_extreme:7d}  ({pct_partial:.2f}%)")
        print(f"  Hiç uç olmayan (sadece fark >5)     : {total_corrupted-fully_extreme-partial_extreme:7d}  ({pct_other:.2f}%)")

        if pct_partial < 1:
            print(f"  HÜKÜM: imnoise EŞ ZAMANLI uygular → tüm kanallar birlikte siyah/beyaz.")
            print(f"          np.all(noisy==0, axis=2) tespiti DOĞRU.")
        else:
            print(f"  HÜKÜM: imnoise KANAL KANAL uygular → kısmi-bozuk pikseller var.")
            print(f"          np.all(noisy==0, axis=2) tespiti bu pikselleri KAÇIRIR → F1 ve maske etkilenir.")

# Görsel — 1 örnek, %50
noisy_ex  = read_rgb(os.path.join(ROOT, "Tiny-Imagenet", "Tiny_Noisy_50", stems[0]+".png"))
clean_ex  = read_rgb(os.path.join(ROOT, "Tiny-Imagenet", "Tiny_Orijinal", stems[0]+".JPEG"))
diff_ex   = np.abs(noisy_ex.astype(np.int32) - clean_ex.astype(np.int32))
corrupt_m = np.any(diff_ex > 5, axis=2)
fully_m   = np.all((noisy_ex == 0) | (noisy_ex == 255), axis=2) & corrupt_m
partial_m = corrupt_m & ~fully_m

vis = clean_ex.copy()
vis[fully_m]   = [0, 255, 0]    # yeşil = tam bozuk
vis[partial_m] = [255, 0, 0]    # kırmızı = kısmi bozuk (olursa)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, img, title, cmap in zip(
    axes,
    [clean_ex, noisy_ex, vis, (corrupt_m*255).astype(np.uint8)],
    ["Orijinal", "Gürültülü (%50)", "Yeşil=tam, Kırmızı=kısmi bozuk", "Tüm bozuk piksel"],
    [None, None, None, "gray"]
):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
plt.suptitle(f"Teşhis 2 — {stems[0]} %50 | imnoise kanal davranışı", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(DIAG_DIR, "diag2_channel_behavior.png"), dpi=100, bbox_inches="tight")
plt.close()
print(f"\nGörsel: diagnostics/diag2_channel_behavior.png")

elapsed = time.time() - T0
print(f"\nToplam süre: {elapsed:.1f}s")
print(f"Tüm çıktılar: {DIAG_DIR}/")
