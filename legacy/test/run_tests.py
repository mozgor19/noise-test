"""
Test 1: DnCNN-SP — doğru maskeyle pipeline vs eğitim ayrımı
Test 2: F1 ground-truth maskesi aynı buga sahip mi?

Tüm çıktılar test/ klasörü altına yazılır.
Mevcut hiçbir dosyaya dokunulmaz.
"""

import os, sys, time
import glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.time()

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
AI_SRC   = os.path.join(ROOT, "ai_based_models", "src")
sys.path.insert(0, AI_SRC)

from utils.spmask import sp_mask_fixed, to_tensor01, to_uint8, mask_to_tensor
from utils.io import read_rgb
import torch
from models.dncnn_sp import DnCNNSP
from models.seconv_unet import SeConvUNet
from models.external_wrappers import load_checkpoint

# ── yardımcılar ───────────────────────────────────────────────────────────────

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

def forward_raw(model, rgb_np, device):
    """Modelin iç mask-gate'ini atla: all-ones maske → saf ağ çıktısı."""
    x  = to_tensor01(rgb_np).unsqueeze(0).to(device)
    m  = torch.ones(1, 1, rgb_np.shape[0], rgb_np.shape[1], device=device)
    with torch.no_grad():
        # forward: pred*1 + rgb*0 = pred  (iç gate nötr)
        return to_uint8(model(torch.cat([x, m], dim=1)))

def forward_with_mask_hw(model, rgb_np, mask_hw, device):
    """HxW bool/float maskesiyle normal forward (model kendi gate'ini uygular)."""
    x  = to_tensor01(rgb_np).unsqueeze(0).to(device)
    mf = mask_hw.astype(np.float32)[:, :, None]      # (H,W,1)
    mt = mask_to_tensor(mf).unsqueeze(0).to(device)   # (1,1,H,W)
    with torch.no_grad():
        return to_uint8(model(torch.cat([x, mt], dim=1)))

def forward_seconv(model, rgb_np, mask_hw, device):
    """SeConvUNet için 8'in katı padding'li forward."""
    x  = to_tensor01(rgb_np).unsqueeze(0).to(device)
    mf = mask_hw.astype(np.float32)[:, :, None]
    mt = mask_to_tensor(mf).unsqueeze(0).to(device)
    x4 = torch.cat([x, mt], dim=1)
    H, W = rgb_np.shape[:2]
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    if ph or pw:
        x4 = torch.nn.functional.pad(x4, (0, pw, 0, ph), "reflect")
    with torch.no_grad():
        y = model(x4)
    if ph or pw:
        y = y[:, :, :H, :W]
    return to_uint8(y)

def seconv_raw(model, rgb_np, device):
    H, W = rgb_np.shape[:2]
    ones = np.ones((H, W), dtype=np.float32)
    return forward_seconv(model, rgb_np, ones, device)

# ── model yükleme ──────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ckpt_dsp = os.path.join(ROOT, "ai_based_models", "weights", "dncnn_sp_v2.pth")
mdsp     = DnCNNSP(channels=3).to(device)
mdsp     = load_checkpoint(mdsp, ckpt_dsp, device, strict=False)
mdsp.eval()

ckpt_sc = os.path.join(ROOT, "ai_based_models", "weights", "seconv_unet_sp_v2.pth")
msc     = SeConvUNet(base=48).to(device)
msc     = load_checkpoint(msc, ckpt_sc, device, strict=False)
msc.eval()
print("Modeller yüklendi.")

# ── görsel seçimi ──────────────────────────────────────────────────────────────

ORIG_DIR = os.path.join(ROOT, "Tiny-Imagenet", "Tiny_Orijinal")
stems    = [os.path.splitext(os.path.basename(p))[0]
            for p in sorted(glob.glob(os.path.join(ORIG_DIR, "*.JPEG")))[:10]]
print(f"Görsel sayısı: {len(stems)}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("TEST 1 — DnCNN-SP pipeline vs eğitim ayrımı")
print("="*70)

t1_rows = []

for level in [30, 50]:
    NOISY_DIR = os.path.join(ROOT, "Tiny-Imagenet", f"Tiny_Noisy_{level}")
    print(f"\n--- Seviye %{level} ---")

    psnr_A, psnr_B, psnr_C, psnr_D, psnr_E = [], [], [], [], []
    mask_stats_list = []

    vis_items = []  # görselleştirme için

    for stem in stems:
        orig_p  = os.path.join(ORIG_DIR, stem + ".JPEG")
        noisy_p = os.path.join(NOISY_DIR, stem + ".png")
        if not (os.path.exists(orig_p) and os.path.exists(noisy_p)):
            continue

        clean = read_rgb(orig_p)
        noisy = read_rgb(noisy_p)

        # ── Maskeler ────────────────────────────────────────────────────
        chmask    = noisy.astype(np.int32) != clean.astype(np.int32)  # (H,W,3) bool
        pmask_any = chmask.any(axis=2)                                 # (H,W)
        pmask_all = chmask.all(axis=2)                                 # (H,W)

        gt_rate_any = pmask_any.mean()
        gt_rate_all = pmask_all.mean()
        ch_pixel_rate = chmask.mean()   # ortalama kanal bazlı oran

        mask_stats_list.append((gt_rate_any, gt_rate_all, ch_pixel_rate))

        # ── A: gürültülü baseline ────────────────────────────────────────
        pA = psnr(noisy, clean)

        # ── B: mevcut pipeline (heuristic sp_mask_fixed) ─────────────────
        mask_h = sp_mask_fixed(noisy)             # (H,W,1)
        out_B  = forward_with_mask_hw(mdsp, noisy, mask_h[:,:,0], device)
        pB     = psnr(out_B, clean)

        # ── Saf ağ çıktısı (all-ones maske → internal gate nötr) ─────────
        raw_dsp = forward_raw(mdsp, noisy, device)          # (H,W,3)
        raw_sc  = seconv_raw(msc, noisy, device)

        # ── C: pmask_any maske → kanal-bazlı birleştirme ─────────────────
        # chmask=True olan kanalda raw_dsp kullan, diğerinde noisy koru
        out_C = np.where(chmask, raw_dsp, noisy).astype(np.uint8)
        pC    = psnr(out_C, clean)

        # ── D: pmask_any maske → tüm-piksel birleştirme ──────────────────
        mask3d = pmask_any[:, :, None]
        out_D  = np.where(mask3d, raw_dsp, noisy).astype(np.uint8)
        pD     = psnr(out_D, clean)

        # ── E: SeConvUNet, kanal-bazlı birleştirme (C benzeri) ───────────
        out_E = np.where(chmask, raw_sc, noisy).astype(np.uint8)
        pE    = psnr(out_E, clean)

        psnr_A.append(pA); psnr_B.append(pB); psnr_C.append(pC)
        psnr_D.append(pD); psnr_E.append(pE)

        print(f"  {stem}  any={gt_rate_any:.3f} all={gt_rate_all:.4f} "
              f"| A={pA:.2f} B={pB:.2f} C={pC:.2f} D={pD:.2f} E={pE:.2f}")

        vis_items.append((stem, clean, noisy, pmask_all, pmask_any,
                          out_B, out_C, raw_dsp))

    def avg(lst): return sum(lst)/len(lst) if lst else float("nan")
    def ms(k): return avg([x[k] for x in mask_stats_list])

    print(f"\n  {'':30s}  %{level} ORTALAMA ({len(psnr_A)} görsel)")
    print(f"  pmask_any (≥1 kanal bozuk):    {ms(0):.4f}  (~%{ms(0)*100:.1f})")
    print(f"  pmask_all (3 kanal bozuk):      {ms(1):.4f}  (~%{ms(1)*100:.2f})")
    print(f"  kanal-bazlı bozuk oran:         {ms(2):.4f}  (~%{ms(2)*100:.1f})")
    print(f"\n  A) Noisy baseline             : {avg(psnr_A):.2f} dB")
    print(f"  B) DnCNN-SP heuristic (mevcut): {avg(psnr_B):.2f} dB  ({avg(psnr_B)-avg(psnr_A):+.2f})")
    print(f"  C) DnCNN-SP + kanal-bazlı gate: {avg(psnr_C):.2f} dB  ({avg(psnr_C)-avg(psnr_A):+.2f})")
    print(f"  D) DnCNN-SP + piksel-bazlı gate: {avg(psnr_D):.2f} dB  ({avg(psnr_D)-avg(psnr_A):+.2f})")
    print(f"  E) SeConvUNet + kanal-bazlı gate:{avg(psnr_E):.2f} dB  ({avg(psnr_E)-avg(psnr_A):+.2f})")

    diff_CB = avg(psnr_C) - avg(psnr_B)
    diff_Cbaseline = avg(psnr_C) - avg(psnr_A)
    print(f"\n  C − B = {diff_CB:+.2f} dB  (doğru maske vs mevcut pipeline farkı)")
    if diff_CB > 2.0:
        verdict = "SUÇLU PIPELINE: doğru maskeyle model değer katıyor, eğitim kurtarılabilir."
    elif diff_Cbaseline < 1.0:
        verdict = "SUÇLU EĞİTİM: doğru maskeyle bile model çıktı kötü, maske değil eğitim yetersiz."
    else:
        verdict = f"KARMA: doğru maske {diff_CB:.1f} dB kazandırıyor ama eğitim de kısıtlıyor."
    print(f"  HÜKÜM: {verdict}")

    t1_rows.append(dict(level=level,
        avg_pmask_any=ms(0), avg_pmask_all=ms(1),
        psnr_A=avg(psnr_A), psnr_B=avg(psnr_B), psnr_C=avg(psnr_C),
        psnr_D=avg(psnr_D), psnr_E=avg(psnr_E)))

    # ── Görselleştirme ────────────────────────────────────────────────────
    for stem, clean, noisy, pall, pany, outB, outC, rawdsp in vis_items[:3]:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        items = [
            ("Orijinal", clean, None),
            ("Gürültülü", noisy, None),
            ("pmask_all (mevcut ≈np.all)", (pall*255).astype(np.uint8), "gray"),
            ("pmask_any (doğru: ≥1 kanal)", (pany*255).astype(np.uint8), "gray"),
            (f"B: heuristic ({psnr(outB,clean):.1f}dB)", outB, None),
            (f"C: kanal-gate ({psnr(outC,clean):.1f}dB)", outC, None),
            (f"Raw ağ çıktısı ({psnr(rawdsp,clean):.1f}dB)", rawdsp, None),
            (f"Referans noisy ({psnr(noisy,clean):.1f}dB)", noisy, None),
        ]
        for ax, (title, img, cmap) in zip(axes.flat, items):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        fig.suptitle(f"Test1 | {stem} | %{level} gürültü", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(TEST_DIR, f"test1_{stem}_lv{level}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("TEST 2 — F1 ground-truth maskesi aynı buga sahip mi?")
print("="*70)

DETECT_THR  = 30  # notebook'taki mevcut değer
RESTORE_THR = 10

def sp_noise_mask_current(noisy, clean):
    """Mevcut notebook sp_noise_mask — DETECT_THR=30, np.all extreme check."""
    is_extreme = np.all(noisy == 0, axis=2) | np.all(noisy == 255, axis=2)
    diff_any   = np.any(np.abs(noisy.astype(np.int32) - clean.astype(np.int32)) > DETECT_THR, axis=2)
    return is_extreme & diff_any  # (H,W) bool

def sp_noise_mask_correct(noisy, clean):
    """Doğru kanal-bazlı maske: herhangi bir kanalda değişiklik var mı?"""
    return (noisy.astype(np.int32) != clean.astype(np.int32)).any(axis=2)  # (H,W) bool

def compute_f1(denoised, clean, noise_mask_hw):
    restored = np.all(
        np.abs(denoised.astype(np.int32) - clean.astype(np.int32)) <= RESTORE_THR,
        axis=2,
    )
    TP = int(np.sum( noise_mask_hw &  restored))
    FP = int(np.sum(~noise_mask_hw & ~restored))
    FN = int(np.sum( noise_mask_hw & ~restored))
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, TP, FP, FN

print(f"\nMevcut notebook sp_noise_mask: np.all(noisy==0) | np.all(noisy==255) "
      f"AND diff>DETECT_THR({DETECT_THR})")
print(f"Doğru maske:                   (noisy != clean).any(axis=2)  [kanal-bazlı]\n")

methods_t2 = ["HDMR", "DAMF", "EMPR"]
t2_summary = []

for level in [30, 50]:
    NOISY_DIR = os.path.join(ROOT, "Tiny-Imagenet", f"Tiny_Noisy_{level}")
    print(f"--- Seviye %{level} ---")

    for method in methods_t2:
        CLEAN_DIR = os.path.join(ROOT, "Tiny-Imagenet", f"Tiny_Cleaned_{method}_{level}")
        if not os.path.isdir(CLEAN_DIR):
            print(f"  {method} klasörü yok, atlandı.")
            continue

        prec_cur, rec_cur, f1_cur = [], [], []
        prec_cor, rec_cor, f1_cor = [], [], []
        mask_cur_rates, mask_cor_rates = [], []

        for stem in stems:
            orig_p    = os.path.join(ORIG_DIR, stem + ".JPEG")
            noisy_p   = os.path.join(NOISY_DIR, stem + ".png")
            cleaned_p = os.path.join(CLEAN_DIR, stem + ".png")
            if not all(os.path.exists(p) for p in [orig_p, noisy_p, cleaned_p]):
                continue

            clean    = read_rgb(orig_p)
            noisy    = read_rgb(noisy_p)
            denoised = read_rgb(cleaned_p)

            m_cur = sp_noise_mask_current(noisy, clean)
            m_cor = sp_noise_mask_correct(noisy, clean)

            mask_cur_rates.append(m_cur.mean())
            mask_cor_rates.append(m_cor.mean())

            p, r, f, *_ = compute_f1(denoised, clean, m_cur)
            prec_cur.append(p); rec_cur.append(r); f1_cur.append(f)

            p, r, f, *_ = compute_f1(denoised, clean, m_cor)
            prec_cor.append(p); rec_cor.append(r); f1_cor.append(f)

        def avg(lst): return sum(lst)/len(lst) if lst else float("nan")
        print(f"\n  {method} %{level}  ({len(f1_cur)} görsel)")
        print(f"    Maske kapsamı  mevcut={avg(mask_cur_rates):.4f}  doğru={avg(mask_cor_rates):.4f}")
        print(f"    Precision  mevcut={avg(prec_cur):.4f}  doğru={avg(prec_cor):.4f}  "
              f"Δ={avg(prec_cor)-avg(prec_cur):+.4f}")
        print(f"    Recall     mevcut={avg(rec_cur):.4f}  doğru={avg(rec_cor):.4f}  "
              f"Δ={avg(rec_cor)-avg(rec_cur):+.4f}")
        print(f"    F1         mevcut={avg(f1_cur):.4f}  doğru={avg(f1_cor):.4f}  "
              f"Δ={avg(f1_cor)-avg(f1_cur):+.4f}")

        t2_summary.append(dict(
            method=method, level=level,
            mask_cur=avg(mask_cur_rates), mask_cor=avg(mask_cor_rates),
            prec_cur=avg(prec_cur), prec_cor=avg(prec_cor),
            rec_cur=avg(rec_cur),   rec_cor=avg(rec_cor),
            f1_cur=avg(f1_cur),     f1_cor=avg(f1_cor),
        ))

print("\n--- TEST 2 ÖZET HÜKÜM ---")
f1_deltas = [r["f1_cor"] - r["f1_cur"] for r in t2_summary]
rec_deltas = [r["rec_cor"] - r["rec_cur"] for r in t2_summary]
print(f"Ortalama F1 artışı (doğru maske): {sum(f1_deltas)/len(f1_deltas):+.4f}")
print(f"Ortalama Recall artışı:           {sum(rec_deltas)/len(rec_deltas):+.4f}")
print(f"Mevcut maskenin kapsamı (np.all): ~{100*sum(r['mask_cur'] for r in t2_summary)/len(t2_summary):.2f}%")
print(f"Doğru maskenin kapsamı (any):     ~{100*sum(r['mask_cor'] for r in t2_summary)/len(t2_summary):.2f}%")

big_delta = any(abs(d) > 0.05 for d in f1_deltas)
if big_delta:
    print("\nHÜKÜM: Evet, F1 ground-truth maskesi aynı bug'dan etkileniyor.")
    print("        Mevcut F1 değerleri yalnızca ~%0.7'lik tam-bozuk piksel üzerinde hesaplanıyor.")
    print("        Doğru maskeyle F1 belirgin değişiyor.")
else:
    print("\nHÜKÜM: F1 değerlerindeki delta küçük — bug etkili ama ölçümü temelden çürütmüyor.")

elapsed = time.time() - T0
print(f"\nToplam süre: {elapsed:.1f}s")
print(f"Çıktılar: {TEST_DIR}/")
print("Mevcut dosyalara dokunulmadı. test/ dışına hiçbir şey yazılmadı.")
