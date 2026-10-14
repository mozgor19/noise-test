"""
Tek değerlendirme betiği: tüm yöntemler (klasik + DL) aynı fonksiyonlarla, aynı anda ölçülür.
evaluate_all_metrics.ipynb ve test/full_eval_v2.py'nin yerini alır.

Metrikler (görsel başına hesaplanıp ortalanır):
  PSNR (dB, RGB, MSE=0 ise 100'e sabitlenir), SSIM (RGB, data_range=255), MAE, RMSE
  Piksel Precision / Recall / F1:
    bozuk piksel   = gürültülü görselde en az bir kanalı orijinalden farklı olan piksel (kanal bazlı imnoise)
    düzeltilmiş    = çıktının tüm kanalları orijinale <= RESTORE_THR yakın
    TP = bozuk & düzeltilmiş, FN = bozuk & düzeltilmemiş, FP = temiz & bozulmuş (yöntem zarar vermiş)
  Top1-Acc% (yalnız Tiny): ResNet-18'in çıktı için tahmini, orijinal görsel için tahminiyle aynı mı.

Kullanım:
  python evaluate.py                    # iki veri seti
  python evaluate.py --datasets Tiny    # yalnız Tiny
Çıktılar: results/summary.csv, results/average_{tiny,480p}.csv, results/delta_480p_minus_tiny.csv,
          results/psnr_ssim.{png,pdf}, results/per_image.csv (istatistik testleri için: stats.py)
"""
import os, glob, argparse, multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = os.path.dirname(os.path.abspath(__file__))
NOISE_LEVELS = [10, 30, 50, 70, 90]
RESTORE_THR = 10
PSNR_CAP = 100.0

METHODS = ["Noisy", "SMF", "AMF", "MDBUTMF", "EMPR", "HDMR", "DAMF", "DnCNN", "DnCNN-SP", "SeConvUNet"]
FOLDER_TAG = {"DnCNN-SP": "DnCNN_SP"}  # yöntem adı -> klasör etiketi

DATASETS = {
    "Tiny": dict(label="Tiny-ImageNet", orig="Tiny-Imagenet/Tiny_Orijinal", orig_ext=".JPEG",
                 noisy="Tiny-Imagenet/Tiny_Noisy_{lv}", cleaned="Tiny-Imagenet/Tiny_Cleaned_{tag}_{lv}",
                 classify=True),
    "480p": dict(label="480p", orig="480p-Test/Originals_resized", orig_ext=".png",
                 noisy="480p-Test/Noisy_{lv}", cleaned="480p-Test/Cleaned_{tag}_{lv}",
                 classify=False),
}

METRIC_COLS = ["PSNR(dB)", "SSIM", "MAE", "RMSE", "Precision", "Recall", "F1"]


def method_dir(ds, method, lv):
    if method == "Noisy":
        return os.path.join(ROOT, ds["noisy"].format(lv=lv))
    return os.path.join(ROOT, ds["cleaned"].format(tag=FOLDER_TAG.get(method, method), lv=lv))

def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def image_metrics(paths):
    orig_p, noisy_p, out_p = paths
    clean, noisy, out = load_rgb(orig_p), load_rgb(noisy_p), load_rgb(out_p)
    if not (clean.shape == noisy.shape == out.shape):
        raise ValueError(f"Boyut uyuşmazlığı: {orig_p} {clean.shape}, {noisy_p} {noisy.shape}, {out_p} {out.shape}")

    diff = out.astype(np.float64) - clean.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    psnr = PSNR_CAP if mse == 0 else min(PSNR_CAP, 10 * np.log10(255.0 ** 2 / mse))
    ssim = float(structural_similarity(clean, out, data_range=255, channel_axis=2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))

    corrupted = (noisy != clean).any(axis=2)
    restored = (np.abs(diff) <= RESTORE_THR).all(axis=2)
    tp = int(np.sum(corrupted & restored))
    fn = int(np.sum(corrupted & ~restored))
    fp = int(np.sum(~corrupted & ~restored))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return psnr, ssim, mae, rmse, prec, rec, f1


class Classifier:
    def __init__(self):
        import torch
        from torchvision import models, transforms as T
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(self.device).eval()
        self.tf = T.Compose([T.Resize(224), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def predict(self, paths, bs=128):
        preds = []
        with self.torch.no_grad():
            for i in range(0, len(paths), bs):
                x = self.torch.stack([self.tf(Image.open(p).convert("RGB")) for p in paths[i:i+bs]])
                preds.append(self.model(x.to(self.device)).argmax(1).cpu().numpy())
        return np.concatenate(preds) if preds else np.array([], dtype=int)


def evaluate_dataset(key, pool, clf):
    ds = DATASETS[key]
    origs = sorted(glob.glob(os.path.join(ROOT, ds["orig"], "*" + ds["orig_ext"])))
    if not origs:
        raise RuntimeError(f"Orijinal görsel bulunamadı: {ds['orig']}")
    stems = [os.path.splitext(os.path.basename(p))[0] for p in origs]
    orig_by_stem = dict(zip(stems, origs))
    print(f"\n[{ds['label']}] {len(stems)} orijinal görsel")

    baseline = dict(zip(stems, clf.predict(origs))) if clf else None

    rows, per_image, problems = [], [], []
    for method in METHODS:
        for lv in NOISE_LEVELS:
            folder = method_dir(ds, method, lv)
            noisy_dir = os.path.join(ROOT, ds["noisy"].format(lv=lv))
            if not os.path.isdir(folder):
                problems.append(f"{ds['label']} {method} %{lv}: klasör yok ({os.path.relpath(folder, ROOT)})")
                continue

            tasks, used = [], []
            for s in stems:
                out_p = os.path.join(folder, s + ".png")
                noisy_p = os.path.join(noisy_dir, s + ".png")
                if os.path.exists(out_p) and os.path.exists(noisy_p):
                    tasks.append((orig_by_stem[s], noisy_p, out_p))
                    used.append(s)
            missing = len(stems) - len(tasks)
            if missing:
                problems.append(f"{ds['label']} {method} %{lv}: {missing} görsel eksik")
            if not tasks:
                continue

            m = np.array(list(pool.map(image_metrics, tasks, chunksize=8)))
            row = {"Dataset": ds["label"], "Yöntem": method, "Seviye": lv, "n": len(tasks), "eksik": missing}
            row.update(dict(zip(METRIC_COLS, m.mean(axis=0))))
            if clf:
                agree = clf.predict([t[2] for t in tasks]) == np.array([baseline[s] for s in used])
                row["Top1-Acc%"] = 100.0 * float(np.mean(agree))
            else:
                agree = np.full(len(tasks), np.nan)
                row["Top1-Acc%"] = np.nan
            rows.append(row)
            for s, mi, a in zip(used, m, agree):
                per_image.append({"Dataset": ds["label"], "Yöntem": method, "Seviye": lv, "img": s,
                                  **dict(zip(METRIC_COLS, mi)), "Top1-uyum": float(a)})
            print(f"  {method:11s} %{lv}: n={len(tasks):4d}  PSNR={row['PSNR(dB)']:6.2f}  SSIM={row['SSIM']:.4f}"
                  f"  F1={row['F1']:.4f}  Acc={row['Top1-Acc%']:5.1f}", flush=True)
    return rows, per_image, problems


def plot(summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(summary["Dataset"].unique())
    fig, axes = plt.subplots(len(labels), 2, figsize=(14, 5 * len(labels)), squeeze=False)
    colors = plt.cm.tab10.colors
    for r, lab in enumerate(labels):
        sub = summary[summary["Dataset"] == lab]
        for c, metric in enumerate(["PSNR(dB)", "SSIM"]):
            ax = axes[r, c]
            for i, method in enumerate(METHODS):
                s = sub[sub["Yöntem"] == method].sort_values("Seviye")
                if len(s):
                    ax.plot(s["Seviye"], s[metric], marker="o", color=colors[i % 10],
                            linestyle="--" if method == "Noisy" else "-", label=method)
            ax.set(title=f"{lab} — {metric}", xlabel="Gürültü Seviyesi (%)", ylabel=metric, xticks=NOISE_LEVELS)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"psnr_ssim.{ext}"), dpi=150, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    clf = Classifier() if any(DATASETS[k]["classify"] for k in args.datasets) else None

    all_rows, all_per_image, all_problems = [], [], []
    # spawn: sınıflandırıcı CUDA'yı başlattıktan sonra fork güvenli değil
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for key in args.datasets:
            rows, per_image, problems = evaluate_dataset(key, pool, clf if DATASETS[key]["classify"] else None)
            all_rows += rows
            all_per_image += per_image
            all_problems += problems

    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    pd.DataFrame(all_per_image).to_csv(os.path.join(args.out, "per_image.csv"), index=False)

    cols = METRIC_COLS + ["Top1-Acc%"]
    averages = {}
    for key in args.datasets:
        lab = DATASETS[key]["label"]
        sub = summary[summary["Dataset"] == lab]
        avg = sub.groupby("Yöntem", sort=False)[cols].mean()
        avg.insert(0, "seviye_sayısı", sub.groupby("Yöntem", sort=False)["Seviye"].count())
        avg.to_csv(os.path.join(args.out, f"average_{key.lower()}.csv"))
        averages[key] = avg
        print(f"\nOrtalama ({lab}, 5 seviye):\n{avg.round(4).to_string()}")

    if len(averages) == 2:
        common = averages["Tiny"].index.intersection(averages["480p"].index)
        delta = averages["480p"].loc[common, METRIC_COLS] - averages["Tiny"].loc[common, METRIC_COLS]
        delta.to_csv(os.path.join(args.out, "delta_480p_minus_tiny.csv"))

    plot(summary, args.out)

    print(f"\nÇıktılar: {os.path.relpath(args.out, ROOT)}/")
    if all_problems:
        print("\n⚠ EKSİKLER:\n  " + "\n  ".join(all_problems))
    else:
        print("Eksik yok: tüm yöntem/seviye klasörleri tam.")


if __name__ == "__main__":
    main()
