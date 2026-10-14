import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT         = os.path.dirname(os.path.abspath(__file__))
NOISE_LEVELS = [10, 30, 50, 70, 90]
DETECT_THR   = 5    # Değişen tek şey bu (eski: 30)
RESTORE_THR  = 10   # Değişmiyor

DATASETS = {
    "Tiny": {
        "orig_dir":    "Tiny-Imagenet/Tiny_Orijinal",
        "orig_exts":   ("*.JPEG", "*.jpg", "*.jpeg", "*.png"),
        "noisy_tmpl":  "Tiny-Imagenet/Tiny_Noisy_{}",
        "methods": {
            "Noisy":      "Tiny-Imagenet/Tiny_Noisy_{}",
            "SMF":        "Tiny-Imagenet/Tiny_Cleaned_SMF_{}",
            "AMF":        "Tiny-Imagenet/Tiny_Cleaned_AMF_{}",
            "MDBUTMF":    "Tiny-Imagenet/Tiny_Cleaned_MDBUTMF_{}",
            "EMPR":       "Tiny-Imagenet/Tiny_Cleaned_EMPR_{}",
            "HDMR":       "Tiny-Imagenet/Tiny_Cleaned_HDMR_{}",
            "DnCNN":      "Tiny-Imagenet/Tiny_Cleaned_DnCNN_{}",
            "DnCNN-SP":   "Tiny-Imagenet/Tiny_Cleaned_DnCNN_SP_{}",
            "DAMF":       "Tiny-Imagenet/Tiny_Cleaned_DAMF_{}",
            "SeConvUNet": "Tiny-Imagenet/Tiny_Cleaned_SeConvUNet_{}",
        },
    },
    "480p": {
        "orig_dir":    "480p-Test/Originals_resized",
        "orig_exts":   ("*.png",),
        "noisy_tmpl":  "480p-Test/Noisy_{}",
        "methods": {
            "Noisy":      "480p-Test/Noisy_{}",
            "SMF":        "480p-Test/Cleaned_SMF_{}",
            "AMF":        "480p-Test/Cleaned_AMF_{}",
            "MDBUTMF":    "480p-Test/Cleaned_MDBUTMF_{}",
            "EMPR":       "480p-Test/Cleaned_EMPR_{}",
            "HDMR":       "480p-Test/Cleaned_HDMR_{}",
            "DnCNN":      "480p-Test/Cleaned_DnCNN_{}",
            "DnCNN-SP":   "480p-Test/Cleaned_DnCNN_SP_{}",
            "DAMF":       "480p-Test/Cleaned_DAMF_{}",
            "SeConvUNet": "480p-Test/Cleaned_SeConvUNet_{}",
        },
    },
}


def load_originals(folder, exts):
    orig = {}
    for ext in exts:
        for p in glob.glob(os.path.join(ROOT, folder, ext)):
            stem = os.path.splitext(os.path.basename(p))[0]
            orig[stem] = np.array(Image.open(p).convert("RGB"))
    return orig


def sp_noise_mask(noisy, clean):
    is_extreme = np.all(noisy == 0, axis=2) | np.all(noisy == 255, axis=2)
    diff_any   = np.any(
        np.abs(noisy.astype(np.int32) - clean.astype(np.int32)) > DETECT_THR,
        axis=2,
    )
    return is_extreme & diff_any


def compute_f1(denoised, clean, noise_mask):
    restored = np.all(
        np.abs(denoised.astype(np.int32) - clean.astype(np.int32)) <= RESTORE_THR,
        axis=2,
    )
    TP = int(np.sum( noise_mask &  restored))
    FP = int(np.sum(~noise_mask & ~restored))
    FN = int(np.sum( noise_mask & ~restored))
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def run_dataset(ds_label, cfg):
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_label}  |  DETECT_THR={DETECT_THR}")
    print(f"{'='*60}")

    originals = load_originals(cfg["orig_dir"], cfg["orig_exts"])
    print(f"  Orijinal: {len(originals)} görsel")

    rows = []
    for method, tmpl in cfg["methods"].items():
        for level in NOISE_LEVELS:
            folder    = os.path.join(ROOT, tmpl.format(level))
            noisy_dir = os.path.join(ROOT, cfg["noisy_tmpl"].format(level))

            if not os.path.isdir(folder):
                print(f"  [{method} %{level}] Klasör yok: {folder}")
                continue

            img_paths = sorted(
                glob.glob(os.path.join(folder, "*.png")) +
                glob.glob(os.path.join(folder, "*.JPEG"))
            )
            if not img_paths:
                continue

            prec_list, rec_list, f1_list = [], [], []

            for p in tqdm(img_paths, desc=f"  {method} %{level}", leave=False):
                stem = os.path.splitext(os.path.basename(p))[0]
                if stem not in originals:
                    continue

                clean    = originals[stem]
                denoised = np.array(Image.open(p).convert("RGB"))

                if denoised.shape != clean.shape:
                    denoised = np.array(
                        Image.fromarray(denoised).resize(
                            (clean.shape[1], clean.shape[0]), Image.BILINEAR
                        )
                    )

                noisy_path = os.path.join(noisy_dir, stem + ".png")
                if not os.path.exists(noisy_path):
                    noisy_path = os.path.join(noisy_dir, stem + ".JPEG")
                if not os.path.exists(noisy_path):
                    continue

                noisy  = np.array(Image.open(noisy_path).convert("RGB"))
                nmask  = sp_noise_mask(noisy, clean)
                p_, r_, f_ = compute_f1(denoised, clean, nmask)
                prec_list.append(p_)
                rec_list.append(r_)
                f1_list.append(f_)

            def avg(lst): return float(np.mean(lst)) if lst else float("nan")

            rows.append({
                "Dataset":    ds_label,
                "Yöntem":     method,
                "Seviye":     f"%{level}",
                "Precision":  avg(prec_list),
                "Recall":     avg(rec_list),
                "F1":         avg(f1_list),
                "N":          len(prec_list),
            })
            r = rows[-1]
            print(f"  {method:12s} %{level}: Prec={r['Precision']:.4f}  Rec={r['Recall']:.4f}  F1={r['F1']:.4f}  (n={r['N']})")

    return pd.DataFrame(rows)


def build_avg_table(df):
    rows = []
    for method in df["Yöntem"].unique():
        sub = df[df["Yöntem"] == method]
        rows.append({
            "Yöntem":    method,
            "Precision": sub["Precision"].mean(),
            "Recall":    sub["Recall"].mean(),
            "F1":        sub["F1"].mean(),
        })
    return pd.DataFrame(rows).set_index("Yöntem")


# ── Çalıştır ──────────────────────────────────────────────────────────────────

all_dfs = {}
for ds_label, cfg in DATASETS.items():
    df = run_dataset(ds_label, cfg)
    all_dfs[ds_label] = df

# Kaydet
tiny_df = all_dfs["Tiny"]
p480_df = all_dfs["480p"]

tiny_df.to_csv(os.path.join(ROOT, "results_f1_thr5_tiny.csv"), index=False)
p480_df.to_csv(os.path.join(ROOT, "results_f1_thr5_480p.csv"), index=False)
print("\nresults_f1_thr5_tiny.csv ve results_f1_thr5_480p.csv kaydedildi.")

# Ortalama tablolar
print("\n" + "="*70)
print("ORTALAMA F1 SONUÇLARI — THR=5  (5 seviye üzerinden)")
print("="*70)

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 120)

print("\n[Tiny ImageNet]")
print(build_avg_table(tiny_df).to_string())

print("\n[480p]")
print(build_avg_table(p480_df).to_string())

# ESKİ (THR=30) ile karşılaştırma — önceki CSV'den oku
print("\n" + "="*70)
print("ESKİ (THR=30) vs YENİ (THR=5) KARŞILAŞTIRMA")
print("="*70)

old_tiny_path = os.path.join(ROOT, "results_average_tiny.csv")
old_480p_path = os.path.join(ROOT, "results_average_480p.csv")

if os.path.exists(old_tiny_path) and os.path.exists(old_480p_path):
    old_tiny = pd.read_csv(old_tiny_path).set_index("Yöntem")
    old_480p = pd.read_csv(old_480p_path).set_index("Yöntem")

    new_tiny = build_avg_table(tiny_df)
    new_480p = build_avg_table(p480_df)

    print("\n[Tiny] F1 — Eski (THR=30) → Yeni (THR=5):")
    for m in new_tiny.index:
        old_f1 = old_tiny.loc[m, "f1"] if m in old_tiny.index else float("nan")
        new_f1 = new_tiny.loc[m, "F1"]
        print(f"  {m:12s}  eski={old_f1:.4f}  yeni={new_f1:.4f}  Δ={new_f1-old_f1:+.4f}")

    print("\n[480p] F1 — Eski (THR=30) → Yeni (THR=5):")
    for m in new_480p.index:
        old_f1 = old_480p.loc[m, "f1"] if m in old_480p.index else float("nan")
        new_f1 = new_480p.loc[m, "F1"]
        print(f"  {m:12s}  eski={old_f1:.4f}  yeni={new_f1:.4f}  Δ={new_f1-old_f1:+.4f}")
else:
    print("Eski CSV'ler bulunamadı, sadece yeni sonuçlar yazdırıldı.")
