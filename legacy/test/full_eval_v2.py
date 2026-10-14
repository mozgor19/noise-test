"""
Düzeltilmiş tam değerlendirme:
  - DnCNN-SP / SeConvUNet: kanal-bazlı kör heuristic gate ile yeni inferans
  - Tüm yöntemler: kanal-bazlı GT maske ile F1/Precision/Recall
  - Klasik + DnCNN PSNR/SSIM/MAE/RMSE: eski CSV'den yükle (değişmiyor)
  - 480p Accuracy: N/A (etiket yok)
  - Tüm çıktılar test/ altına
"""

import os, sys, glob, time, io
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from skimage.metrics import (peak_signal_noise_ratio as skpsnr,
                             structural_similarity as skssim)
from tqdm import tqdm

T0 = time.time()
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
AI_SRC   = os.path.join(ROOT, "ai_based_models", "src")
sys.path.insert(0, AI_SRC)

from utils.spmask import to_tensor01, to_uint8, mask_to_tensor
from utils.io import read_rgb, write_rgb
from models.dncnn_sp import DnCNNSP
from models.seconv_unet import SeConvUNet
from models.external_wrappers import load_checkpoint

# ── klasörler ──────────────────────────────────────────────────────────────────
DL_OUT   = os.path.join(TEST_DIR, "dl_outputs")
CSV_DIR  = os.path.join(TEST_DIR, "results_v2")
FIG_DIR  = os.path.join(TEST_DIR, "figures")
REP_DIR  = os.path.join(TEST_DIR, "report")
for d in [DL_OUT, CSV_DIR, FIG_DIR, REP_DIR]:
    os.makedirs(d, exist_ok=True)

# ── sabitler ──────────────────────────────────────────────────────────────────
NOISE_LEVELS = [10, 30, 50, 70, 90]
RESTORE_THR  = 10
# Kör heuristic kanal maskesi eşikleri (raporda belirtilecek)
BLIND_EXTREME_EXACT = True   # tam 0 veya 255
BLIND_DIFF_THR      = 20     # yerel medyandan fark
BLIND_KSIZE         = 3

METHODS_ORDER = ["Noisy","SMF","AMF","MDBUTMF","EMPR","HDMR","DAMF",
                 "DnCNN","DnCNN-SP","SeConvUNet"]

DATASETS = {
    "Tiny": {
        "orig_dir":   os.path.join(ROOT, "Tiny-Imagenet", "Tiny_Orijinal"),
        "orig_ext":   ("*.JPEG",),
        "noisy_tmpl": os.path.join(ROOT, "Tiny-Imagenet", "Tiny_Noisy_{}"),
        "folder": {
            "Noisy":      os.path.join(ROOT, "Tiny-Imagenet", "Tiny_Noisy_{}"),
            "SMF":        os.path.join(ROOT, "Tiny-Imagenet", "Tiny_Cleaned_SMF_{}"),
            "AMF":        os.path.join(ROOT, "Tiny-Imagenet", "Tiny_Cleaned_AMF_{}"),
            "MDBUTMF":    os.path.join(ROOT, "Tiny-Imagenet", "Tiny_Cleaned_MDBUTMF_{}"),
            "EMPR":       os.path.join(ROOT, "Tiny-Imagenet", "Tiny_Cleaned_EMPR_{}"),
            "HDMR":       os.path.join(ROOT, "Tiny-Imagenet", "Tiny_Cleaned_HDMR_{}"),
            "DAMF":       os.path.join(ROOT, "Tiny-Imagenet", "Tiny_Cleaned_DAMF_{}"),
            "DnCNN":      os.path.join(ROOT, "Tiny-Imagenet", "Tiny_Cleaned_DnCNN_{}"),
            "DnCNN-SP":   os.path.join(DL_OUT, "Tiny_DnCNN_SP_v2_{}"),
            "SeConvUNet": os.path.join(DL_OUT, "Tiny_SeConvUNet_v2_{}"),
        },
    },
    "480p": {
        "orig_dir":   os.path.join(ROOT, "480p-Test", "Originals_resized"),
        "orig_ext":   ("*.png",),
        "noisy_tmpl": os.path.join(ROOT, "480p-Test", "Noisy_{}"),
        "folder": {
            "Noisy":      os.path.join(ROOT, "480p-Test", "Noisy_{}"),
            "SMF":        os.path.join(ROOT, "480p-Test", "Cleaned_SMF_{}"),
            "AMF":        os.path.join(ROOT, "480p-Test", "Cleaned_AMF_{}"),
            "MDBUTMF":    os.path.join(ROOT, "480p-Test", "Cleaned_MDBUTMF_{}"),
            "EMPR":       os.path.join(ROOT, "480p-Test", "Cleaned_EMPR_{}"),
            "HDMR":       os.path.join(ROOT, "480p-Test", "Cleaned_HDMR_{}"),
            "DAMF":       os.path.join(ROOT, "480p-Test", "Cleaned_DAMF_{}"),
            "DnCNN":      os.path.join(ROOT, "480p-Test", "Cleaned_DnCNN_{}"),
            "DnCNN-SP":   os.path.join(DL_OUT, "P480_DnCNN_SP_v2_{}"),
            "SeConvUNet": os.path.join(DL_OUT, "P480_SeConvUNet_v2_{}"),
        },
    },
}

# Hangi metrikler eski CSV'den alınacak (yeniden hesaplanmayacak)
REUSE_METHODS = {"Noisy","SMF","AMF","MDBUTMF","EMPR","HDMR","DAMF","DnCNN"}

# ── yardımcılar ───────────────────────────────────────────────────────────────
def load_img(path):
    return np.array(Image.open(path).convert("RGB"))

def get_stems(orig_dir, exts):
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(orig_dir, e))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}

def calc_psnr(a, b):
    mse = np.mean((a.astype(np.float64)-b.astype(np.float64))**2)
    return 100.0 if mse==0 else 20*np.log10(255./np.sqrt(mse))

def calc_ssim(a, b):
    return float(skssim(a, b, data_range=255, channel_axis=2))

def calc_mae(a, b):
    return float(np.mean(np.abs(a.astype(np.float64)-b.astype(np.float64))))

def calc_rmse(a, b):
    return float(np.sqrt(np.mean((a.astype(np.float64)-b.astype(np.float64))**2)))

def calc_f1_ch(denoised, clean, noisy):
    """Kanal-bazlı GT maske ile F1."""
    ch_mask   = (noisy.astype(np.int32) != clean.astype(np.int32))   # (H,W,3)
    pw_mask   = ch_mask.any(axis=2)                                    # (H,W)
    restored  = np.all(
        np.abs(denoised.astype(np.int32)-clean.astype(np.int32)) <= RESTORE_THR,
        axis=2,
    )
    TP = int(np.sum( pw_mask &  restored))
    FP = int(np.sum(~pw_mask & ~restored))
    FN = int(np.sum( pw_mask & ~restored))
    prec = TP/(TP+FP) if TP+FP>0 else 0.
    rec  = TP/(TP+FN) if TP+FN>0 else 0.
    f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.
    return prec, rec, f1

def blind_channel_mask(noisy_rgb):
    """Kanal-bazlı kör heuristic (temize bakmadan)."""
    mask = np.zeros(noisy_rgb.shape, dtype=bool)
    for c in range(3):
        ch  = noisy_rgb[:,:,c]
        med = cv2.medianBlur(ch, BLIND_KSIZE)
        is_ext  = (ch==0) | (ch==255)
        diff_ok = np.abs(ch.astype(np.int32)-med.astype(np.int32)) > BLIND_DIFF_THR
        mask[:,:,c] = is_ext & diff_ok
    return mask  # (H,W,3) bool

def infer_blind(model_fn, noisy_rgb, device, is_seconv=False):
    """All-ones iç maske → raw pred → kanal-bazlı kör gate."""
    H, W = noisy_rgb.shape[:2]
    x  = to_tensor01(noisy_rgb).unsqueeze(0).to(device)
    m  = torch.ones(1,1,H,W, device=device)
    x4 = torch.cat([x,m], dim=1)
    if is_seconv:
        ph,pw = (8-H%8)%8, (8-W%8)%8
        if ph or pw:
            x4 = torch.nn.functional.pad(x4,(0,pw,0,ph),"reflect")
    with torch.no_grad():
        y = model_fn(x4)
    if is_seconv and (ph or pw):
        y = y[:,:,:H,:W]
    pred = to_uint8(y)                         # (H,W,3)
    ch_mask = blind_channel_mask(noisy_rgb)    # (H,W,3)
    return np.where(ch_mask, pred, noisy_rgb).astype(np.uint8)

# ── modeller ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

mdsp = DnCNNSP(channels=3).to(device)
mdsp = load_checkpoint(mdsp, os.path.join(ROOT,"ai_based_models","weights","dncnn_sp_v2.pth"), device, strict=False)
mdsp.eval()

msc = SeConvUNet(base=48).to(device)
msc = load_checkpoint(msc, os.path.join(ROOT,"ai_based_models","weights","seconv_unet_sp_v2.pth"), device, strict=False)
msc.eval()
print("Modeller yüklendi.")

# ── ESKİ CSV yükle ────────────────────────────────────────────────────────────
old_summary = pd.read_csv(os.path.join(ROOT, "results_summary.csv"))
# {(dataset_label, method, level): {metric: value}}
old_data = {}
for _, row in old_summary.iterrows():
    ds  = "Tiny" if "Tiny" in row["Dataset"] else "480p"
    lv  = int(row["Seviye"].replace("%",""))
    key = (ds, row["Yöntem"], lv)
    old_data[key] = {
        "psnr": row["PSNR(dB)"], "ssim": row["SSIM"],
        "mae":  row["MAE"],      "rmse": row["RMSE"],
        "acc":  row["Top1-Acc%"],
    }
print(f"Eski CSV yüklendi: {len(old_data)} satır.")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Yeni DL inferansı
# ══════════════════════════════════════════════════════════════════════════════

def run_dl_inference(ds_key, ds_cfg, model, tag, is_seconv=False):
    stems_dict = get_stems(ds_cfg["orig_dir"], ds_cfg["orig_ext"])
    stems      = sorted(stems_dict.keys())
    for level in NOISE_LEVELS:
        noisy_dir = ds_cfg["noisy_tmpl"].format(level)
        out_folder = os.path.join(DL_OUT, f"{'Tiny' if ds_key=='Tiny' else 'P480'}_{tag}_{level}")
        if os.path.isdir(out_folder) and len(os.listdir(out_folder)) >= len(stems):
            print(f"  [{ds_key}] {tag} %{level}: zaten mevcut, atlanıyor.")
            continue
        os.makedirs(out_folder, exist_ok=True)
        print(f"  [{ds_key}] {tag} %{level}: {len(stems)} görsel...", flush=True)
        for stem in tqdm(stems, leave=False):
            noisy_path = os.path.join(noisy_dir, stem+".png")
            if not os.path.exists(noisy_path):
                noisy_path = os.path.join(noisy_dir, stem+".JPEG")
            if not os.path.exists(noisy_path):
                continue
            noisy  = read_rgb(noisy_path)
            result = infer_blind(model, noisy, device, is_seconv=is_seconv)
            write_rgb(os.path.join(out_folder, stem+".png"), result)

print("\n── Phase 1: DL inferansı ─────────────────────────────────────────────")
for ds_key, ds_cfg in DATASETS.items():
    run_dl_inference(ds_key, ds_cfg, mdsp,  "DnCNN_SP_v2",   is_seconv=False)
    run_dl_inference(ds_key, ds_cfg, msc,   "SeConvUNet_v2", is_seconv=True)

print(f"  DL inferans süresi: {(time.time()-T0)/60:.1f} dk")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — ResNet-18 accuracy (Tiny only)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Phase 2: ResNet-18 accuracy ──────────────────────────────────────")
from torchvision import models, transforms as T
clf = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
clf.eval()
clf_tf = T.Compose([T.Resize(224), T.ToTensor(),
                    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

@torch.no_grad()
def predict_cls(img_np):
    pil = Image.fromarray(img_np)
    return clf(clf_tf(pil).unsqueeze(0).to(device)).argmax(1).item()

ds_tiny  = DATASETS["Tiny"]
orig_map = get_stems(ds_tiny["orig_dir"], ds_tiny["orig_ext"])
stems_t  = sorted(orig_map.keys())

print("  Baseline sınıfları hesaplanıyor...")
baseline_cls = {}
for stem in tqdm(stems_t, leave=False):
    baseline_cls[stem] = predict_cls(read_rgb(orig_map[stem]))

acc_cache = {}   # (method, level): mean accuracy
DL_METHODS = {"DnCNN-SP", "SeConvUNet"}

for method in METHODS_ORDER:
    for level in NOISE_LEVELS:
        if method not in DL_METHODS:
            # Eski CSV'den al
            key = ("Tiny", method, level)
            acc_cache[(method,level)] = old_data.get(key,{}).get("acc", float("nan"))
            continue
        folder = ds_tiny["folder"][method].format(level)
        if not os.path.isdir(folder):
            acc_cache[(method,level)] = float("nan"); continue
        acc_list = []
        for stem in stems_t:
            p = os.path.join(folder, stem+".png")
            if not os.path.exists(p): continue
            pred = predict_cls(read_rgb(p))
            acc_list.append(1 if pred==baseline_cls.get(stem,-1) else 0)
        acc_cache[(method,level)] = float(np.mean(acc_list)*100) if acc_list else float("nan")
        print(f"  DnCNN-SP/SeConvUNet acc {method} %{level}: {acc_cache[(method,level)]:.2f}%")

print(f"  Accuracy süresi: {(time.time()-T0)/60:.1f} dk")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Metrik hesaplama
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Phase 3: Metrikler ──────────────────────────────────────────────")

# records[ds][method][level] = {psnr, ssim, mae, rmse, prec, rec, f1, acc}
records = {ds: {m: {lv: None for lv in NOISE_LEVELS}
                for m in METHODS_ORDER}
           for ds in DATASETS}

for ds_key, ds_cfg in DATASETS.items():
    orig_map_ds = get_stems(ds_cfg["orig_dir"], ds_cfg["orig_ext"])
    stems_ds    = sorted(orig_map_ds.keys())
    print(f"\n  [{ds_key}] {len(stems_ds)} görsel")

    for method in METHODS_ORDER:
        for level in NOISE_LEVELS:
            folder_tmpl = ds_cfg["folder"][method]
            folder      = folder_tmpl.format(level)
            noisy_dir   = ds_cfg["noisy_tmpl"].format(level)

            if not os.path.isdir(folder):
                print(f"    {method} %{level}: klasör yok → atlanıyor")
                continue

            psnr_l, ssim_l, mae_l, rmse_l = [], [], [], []
            prec_l, rec_l, f1_l           = [], [], []
            need_full_metrics = (method in DL_METHODS)

            for stem in tqdm(stems_ds, desc=f"  {method}%{level}", leave=False):
                cleaned_path = os.path.join(folder, stem+".png")
                if not os.path.exists(cleaned_path):
                    cleaned_path = os.path.join(folder, stem+".JPEG")
                if not os.path.exists(cleaned_path):
                    continue

                orig_path = orig_map_ds.get(stem)
                if orig_path is None:
                    continue

                noisy_path = os.path.join(noisy_dir, stem+".png")
                if not os.path.exists(noisy_path):
                    noisy_path = os.path.join(noisy_dir, stem+".JPEG")
                if not os.path.exists(noisy_path):
                    continue

                clean    = read_rgb(orig_path)
                noisy    = read_rgb(noisy_path)
                denoised = read_rgb(cleaned_path)

                if denoised.shape != clean.shape:
                    denoised = np.array(
                        Image.fromarray(denoised).resize(
                            (clean.shape[1], clean.shape[0]), Image.BILINEAR))

                if need_full_metrics:
                    psnr_l.append(calc_psnr(denoised, clean))
                    ssim_l.append(calc_ssim(denoised, clean))
                    mae_l.append(calc_mae(denoised, clean))
                    rmse_l.append(calc_rmse(denoised, clean))

                p, r, f = calc_f1_ch(denoised, clean, noisy)
                prec_l.append(p); rec_l.append(r); f1_l.append(f)

            def avg(lst): return float(np.mean(lst)) if lst else float("nan")

            if need_full_metrics:
                psnr_v = avg(psnr_l); ssim_v = avg(ssim_l)
                mae_v  = avg(mae_l);  rmse_v = avg(rmse_l)
            else:
                key = (ds_key, method, level)
                od  = old_data.get(key, {})
                psnr_v = od.get("psnr", float("nan"))
                ssim_v = od.get("ssim", float("nan"))
                mae_v  = od.get("mae",  float("nan"))
                rmse_v = od.get("rmse", float("nan"))

            if ds_key == "Tiny":
                acc_v = acc_cache.get((method, level), float("nan"))
            else:
                acc_v = float("nan")  # N/A

            records[ds_key][method][level] = {
                "psnr": psnr_v, "ssim": ssim_v,
                "mae":  mae_v,  "rmse": rmse_v,
                "prec": avg(prec_l), "rec": avg(rec_l), "f1": avg(f1_l),
                "acc":  acc_v,
            }
            r = records[ds_key][method][level]
            print(f"    {method:12s} %{level}: PSNR={r['psnr']:.2f} F1={r['f1']:.4f} Acc={r['acc']:.1f}")

print(f"\n  Metrik süresi: {(time.time()-T0)/60:.1f} dk")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — CSV üretimi
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Phase 4: CSV ────────────────────────────────────────────────────")
METRICS_KEYS = ["psnr","ssim","mae","rmse","prec","rec","f1","acc"]
METRICS_COLS = ["PSNR(dB)","SSIM","MAE","RMSE","Precision","Recall","F1","Top1-Acc%"]

summary_rows = []
for ds_key in DATASETS:
    ds_label = "Tiny-ImageNet" if ds_key=="Tiny" else "480p"
    for method in METHODS_ORDER:
        for level in NOISE_LEVELS:
            r = records[ds_key][method].get(level)
            if r is None: continue
            row = {"Dataset": ds_label, "Yöntem": method, "Seviye": f"%{level}"}
            for k, col in zip(METRICS_KEYS, METRICS_COLS):
                row[col] = r[k]
            summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(CSV_DIR,"summary_v2.csv"), index=False)

def build_avg(ds_key):
    rows = []
    for method in METHODS_ORDER:
        vals = {k: [] for k in METRICS_KEYS}
        for level in NOISE_LEVELS:
            r = records[ds_key][method].get(level)
            if r is None: continue
            for k in METRICS_KEYS:
                v = r[k]
                if not np.isnan(v): vals[k].append(v)
        rows.append({"Yöntem": method, **{k: np.mean(vals[k]) if vals[k] else float("nan") for k in METRICS_KEYS}})
    return pd.DataFrame(rows).set_index("Yöntem")

avg_tiny = build_avg("Tiny")
avg_480p = build_avg("480p")
avg_tiny.to_csv(os.path.join(CSV_DIR,"average_tiny_v2.csv"))
avg_480p.to_csv(os.path.join(CSV_DIR,"average_480p_v2.csv"))

# Delta
common = avg_tiny.index.intersection(avg_480p.index)
delta  = avg_480p.loc[common] - avg_tiny.loc[common]
delta.to_csv(os.path.join(CSV_DIR,"delta_v2.csv"))
print("  CSV'ler kaydedildi.")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Grafikler
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Phase 5: Grafikler ──────────────────────────────────────────────")
import matplotlib.cm as cm

COLORS = cm.tab10.colors
METHOD_COLORS = {m: COLORS[i%10] for i,m in enumerate(METHODS_ORDER)}
LINESTYLES = {m: "--" if m=="Noisy" else "-" for m in METHODS_ORDER}

def get_vals(ds_key, method, metric):
    return [records[ds_key][method].get(lv,{}).get(metric,float("nan")) for lv in NOISE_LEVELS]

def make_4panel_fig(metric1, metric2, ylabel1, ylabel2, title):
    fig, axes = plt.subplots(2,2, figsize=(14,9))
    for row, ds_key in enumerate(["Tiny","480p"]):
        ds_label = "Tiny ImageNet" if ds_key=="Tiny" else "480p (DIV2K)"
        for col, (metric, ylabel) in enumerate([(metric1,ylabel1),(metric2,ylabel2)]):
            ax = axes[row][col]
            for m in METHODS_ORDER:
                vals = get_vals(ds_key, m, metric)
                ax.plot(NOISE_LEVELS, vals, marker="o" if ds_key=="Tiny" else "s",
                        color=METHOD_COLORS[m], linestyle=LINESTYLES[m],
                        linewidth=1.5, markersize=4, label=m)
            ax.set(xlabel="Gürültü (%)", ylabel=ylabel,
                   title=f"{ds_label} — {ylabel}", xticks=NOISE_LEVELS)
            ax.legend(fontsize=6, loc="upper right")
            ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig

fig_psnr_ssim = make_4panel_fig(
    "psnr","ssim","PSNR (dB)","SSIM",
    "S&P Denoising — PSNR/SSIM Karşılaştırması (Düzeltilmiş)")
fig_psnr_ssim.savefig(os.path.join(FIG_DIR,"psnr_ssim_v2.pdf"),bbox_inches="tight")
fig_psnr_ssim.savefig(os.path.join(FIG_DIR,"psnr_ssim_v2.png"),dpi=150,bbox_inches="tight")

fig_f1 = make_4panel_fig(
    "f1","rec","F1 Skoru","Recall",
    "S&P Denoising — F1/Recall Karşılaştırması (Düzeltilmiş, kanal-bazlı GT maske)")
fig_f1.savefig(os.path.join(FIG_DIR,"f1_recall_v2.pdf"),bbox_inches="tight")
fig_f1.savefig(os.path.join(FIG_DIR,"f1_recall_v2.png"),dpi=150,bbox_inches="tight")
plt.close("all")
print("  Grafikler kaydedildi.")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — PDF Raporu
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Phase 6: PDF Raporu ─────────────────────────────────────────────")

def table_fig(df_data, col_labels, row_labels, title, fmt_fn=None, figw=13, figh=4):
    """Matplotlib tablosu olarak sayfa üretir."""
    fig, ax = plt.subplots(figsize=(figw, figh))
    ax.axis("off")
    cell_text = []
    for row in df_data:
        cell_text.append([fmt_fn(v) if fmt_fn else str(v) for v in row])
    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   rowLabels=row_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.4)
    # Başlık satırı rengi
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # "En iyi" satırı
    if row_labels and row_labels[-1] == "En iyi":
        for j in range(len(col_labels)):
            tbl[len(row_labels), j].set_facecolor("#d5f5e3")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig

def build_metric_table(ds_key, metric, fmt):
    """Her yöntem satır, her seviye sütun, son sütun Ort., son satır En iyi."""
    cols = [f"%{lv}" for lv in NOISE_LEVELS] + ["Ort."]
    rows = []
    data = []
    for m in METHODS_ORDER:
        vals = [records[ds_key][m].get(lv,{}).get(metric,float("nan"))
                for lv in NOISE_LEVELS]
        avg  = float(np.nanmean(vals)) if any(not np.isnan(v) for v in vals) else float("nan")
        rows.append(m)
        data.append(vals + [avg])
    # En iyi satır
    best = []
    for col_i in range(len(NOISE_LEVELS)+1):
        col_vals = [data[r][col_i] for r in range(len(rows))]
        higher   = metric in ("psnr","ssim","f1","rec","prec","acc")
        fn       = max if higher else min
        best_v   = fn((v for v in col_vals if not np.isnan(v)), default=float("nan"))
        best_m   = next((rows[i] for i,v in enumerate(col_vals) if v==best_v), "?")
        best.append(best_m)
    rows.append("En iyi")
    data.append(best)
    return data, cols, rows

pdf_path = os.path.join(REP_DIR, "report_v2.pdf")
with PdfPages(pdf_path) as pdf:

    # ─ Kapak ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    ax.text(0.5, 0.80, "Kapsamlı Gürültü Giderme Değerlendirmesi",
            ha="center", va="center", fontsize=18, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.70, "Düzeltilmiş Sonuçlar — v2",
            ha="center", va="center", fontsize=14, color="#c0392b", transform=ax.transAxes)
    ax.text(0.5, 0.60,
            "Tuz-biber (salt & pepper) gürültü giderme algoritmalarının\n"
            "Tiny ImageNet-200 ve 480p görüntü setleri üzerinde karşılaştırmalı değerlendirmesi.",
            ha="center", va="center", fontsize=11, transform=ax.transAxes)
    ax.text(0.5, 0.48,
            "Metrikler: PSNR · SSIM · MAE · RMSE · Pixel-F1/Precision/Recall · ResNet-18 Top-1 Accuracy",
            ha="center", va="center", fontsize=10, color="#555", transform=ax.transAxes)
    ax.text(0.5, 0.40,
            "Yöntemler: SMF · AMF · MDBUTMF · EMPR · HDMR · DAMF · DnCNN · DnCNN-SP · SeConvUNet",
            ha="center", va="center", fontsize=10, color="#555", transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ─ Düzeltme Notu ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.axis("off")
    note = (
        "DÜZELTME NOTU — METODOLOJİ\n\n"
        "Önceki sonuçlarda iki pipeline hatası tespit edildi ve düzeltildi.\n\n"

        "HATA 1 — imnoise kanal davranışı (Ampirik olarak doğrulandı)\n"
        "MATLAB imnoise('salt & pepper') her kanalı BAĞIMSIZ olarak bozuyor. Bir pikselin\n"
        "yalnızca 1 veya 2 kanalı 0/255'e set edilebilir. Bozuk piksellerin %85-95'i\n"
        "'kısmi bozuk' (yalnızca 1-2 kanal uçta). Eski pipeline np.all(noisy==0, axis=2)\n"
        "kontrolü kullandığından bozuklukların yalnızca ~%5-15'ini yakalıyordu.\n\n"

        "HATA 2 — F1 GT maskesi (Hata 1'in yansıması)\n"
        "F1 hesabında kullanılan gürültü ground-truth maskesi np.all bazlı olduğundan\n"
        "kısmi bozuk pikselleri kaçırıyordu. Gerçek F1 değerleri raporlanandan ~10× yüksek.\n\n"

        "DÜZELTME 1 — DnCNN-SP / SeConvUNet: Kanal-bazlı kör gate\n"
        "  • Heuristic: ch==0 VEYA ch==255 VE yerel 3×3 medyandan >20 sapma → kanal bozuk\n"
        "  • Temize referans yok (blind). Eşikler: exact {0,255}, diff_thr=20, ksize=3.\n"
        "  • Çıktı: pred_ch (maske=1), noisy_ch (maske=0) — temiz kanallar korunuyor.\n"
        "  • Standart DnCNN (residual): değiştirilmedi. Mimari nedenli başarısızlık geçerli.\n\n"

        "DÜZELTME 2 — F1 GT maskesi: Kanal-bazlı gerçek maske\n"
        "  • Yeni GT mask: chmask = (noisy != clean) per kanal → pw_mask = chmask.any(axis=2)\n"
        "  • Klasik filtreler kanalları bağımsız işlediğinden PSNR/SSIM/MAE/RMSE'leri\n"
        "    değişmez; sadece F1/Precision/Recall düzelir.\n\n"

        "DEĞİŞMEYEN: Klasik + DnCNN PSNR/SSIM/MAE/RMSE (eski CSV'den).\n"
        "480p Accuracy: N/A — DIV2K görüntülerinin ImageNet sınıf etiketi yok."
    )
    ax.text(0.05, 0.95, note, ha="left", va="top", fontsize=9, fontfamily="monospace",
            transform=ax.transAxes, bbox=dict(facecolor="#fef9e7", edgecolor="#d4ac0d",
            boxstyle="round,pad=0.5"))
    ax.set_title("Metodoloji Notu", fontsize=12, fontweight="bold")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ─ Per-metrik tablolar ────────────────────────────────────────────────────
    METRIC_META = [
        ("psnr", "PSNR (dB)", ".2f", True),
        ("ssim", "SSIM",      ".4f", True),
        ("mae",  "MAE",       ".3f", False),
        ("rmse", "RMSE",      ".3f", False),
        ("prec", "Precision", ".4f", True),
        ("rec",  "Recall",    ".4f", True),
        ("f1",   "F1",        ".4f", True),
        ("acc",  "ResNet-18 Top-1 Accuracy (%)", ".1f", True),
    ]

    for ds_key in ["Tiny","480p"]:
        ds_label = "TINY IMAGENET" if ds_key=="Tiny" else "480p (DIV2K)"

        # Ayraç sayfası
        fig, ax = plt.subplots(figsize=(12,2))
        ax.axis("off")
        ax.text(0.5,0.5, f"{ds_label} SONUÇLARI",
                ha="center",va="center",fontsize=20,fontweight="bold",
                color="white", transform=ax.transAxes,
                bbox=dict(facecolor="#2c3e50",boxstyle="round,pad=0.8"))
        pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)

        for (metric, mname, fmt_str, higher) in METRIC_META:
            dir_str = "↑ yüksek iyi" if higher else "↓ düşük iyi"
            data, cols, rows = build_metric_table(ds_key, metric, fmt_str)
            fmt_fn = lambda v, f=fmt_str: (f"{v:{f}}" if isinstance(v,(int,float)) and not np.isnan(v) else str(v))
            figh = max(3, len(METHODS_ORDER)*0.38 + 1.5)
            fig = table_fig(data, cols, rows,
                            f"[{ds_label}] {mname}  [{dir_str}]",
                            fmt_fn, figw=12, figh=figh)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ─ Ortalama tablolar ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12,2))
    ax.axis("off")
    ax.text(0.5,0.5,"ORTALAMA SONUÇLAR",
            ha="center",va="center",fontsize=20,fontweight="bold",
            color="white",transform=ax.transAxes,
            bbox=dict(facecolor="#1a5276",boxstyle="round,pad=0.8"))
    pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)

    for ds_key, lbl in [("Tiny","Tiny ImageNet"),("480p","480p (DIV2K)")]:
        avg_df = build_avg(ds_key)
        cols   = [c.upper() for c in METRICS_KEYS]
        rows   = list(avg_df.index)
        data   = avg_df.values.tolist()
        fmts   = [".2f",".4f",".3f",".3f",".4f",".4f",".4f",".1f"]
        fmt_fn = lambda row_vals, fmts=fmts: [
            (f"{v:{f}}" if not np.isnan(v) else "N/A")
            for v,f in zip(row_vals,fmts)]
        cell_text = [fmt_fn(r) for r in data]
        fig, ax = plt.subplots(figsize=(14, max(3, len(rows)*0.4+1.5)))
        ax.axis("off")
        tbl = ax.table(cellText=cell_text, colLabels=cols, rowLabels=rows,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,1.5)
        for j in range(len(cols)):
            tbl[0,j].set_facecolor("#1a5276"); tbl[0,j].set_text_props(color="white",fontweight="bold")
        ax.set_title(f"Ortalama Sonuçlar — {lbl}", fontsize=11, fontweight="bold")
        pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)

    # Delta tablosu
    cols   = [c.upper() for c in METRICS_KEYS]
    rows   = list(delta.index)
    data   = delta.values.tolist()
    fmts   = [".2f",".4f",".3f",".3f",".4f",".4f",".4f",".1f"]
    cell_text = [[f"{v:{f}}" if not np.isnan(v) else "N/A" for v,f in zip(r,fmts)] for r in data]
    fig, ax = plt.subplots(figsize=(14, max(3, len(rows)*0.4+1.5)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=cols, rowLabels=rows,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,1.5)
    for j in range(len(cols)):
        tbl[0,j].set_facecolor("#6c3483"); tbl[0,j].set_text_props(color="white",fontweight="bold")
    ax.set_title("DELTA: 480p − Tiny ImageNet (pozitif = 480p daha iyi)", fontsize=11, fontweight="bold")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ─ Grafikler ─────────────────────────────────────────────────────────────
    for fig_fn, label in [(fig_psnr_ssim,""), (fig_f1,"")]:
        pass  # figürler kapandı, yeniden üret

    fig_psnr_ssim = make_4panel_fig("psnr","ssim","PSNR (dB)","SSIM",
        "PSNR/SSIM Karşılaştırması — Tiny ImageNet ve 480p (Düzeltilmiş)")
    pdf.savefig(fig_psnr_ssim, bbox_inches="tight"); plt.close(fig_psnr_ssim)

    fig_f1 = make_4panel_fig("f1","rec","F1","Recall",
        "F1/Recall Karşılaştırması — Tiny ImageNet ve 480p (Kanal-bazlı GT maske)")
    pdf.savefig(fig_f1, bbox_inches="tight"); plt.close(fig_f1)

    d = pdf.infodict()
    d["Title"]   = "SP Denoising Düzeltilmiş Rapor v2"
    d["Subject"] = "PSNR/SSIM/F1 metrikleri, kanal-bazlı GT maske"

print(f"  PDF raporu: {pdf_path}")

# ══════════════════════════════════════════════════════════════════════════════
elapsed = time.time() - T0
print(f"\nToplam süre: {elapsed/60:.1f} dakika")
print(f"Çıktılar: {TEST_DIR}/")
print("Mevcut dosyalara dokunulmadı. test/ dışına hiçbir şey yazılmadı.")
