import os, glob, argparse, random
import numpy as np
import cv2
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.config import load_config
from utils.spmask import to_tensor01, mask_to_tensor, add_sp_noise, sp_mask_channel

from models.dncnn import DnCNN
from models.dncnn_sp import DnCNNSP
from models.seconv_unet import SeConvUNet
from models.external_wrappers import ExternalModelWrapper, NOISE_MODEL

VAL_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]

def make_input(clean_rgb, density, use_mask_channel, rng=np.random):
    """Temiz uint8 RGB -> (model girdisi, hedef). Test hattıyla aynı gürültü ve maske."""
    noisy = add_sp_noise(clean_rgb, density, rng)
    x = to_tensor01(noisy)                                   # (3,H,W)
    if use_mask_channel:
        x = torch.cat([x, mask_to_tensor(sp_mask_channel(noisy))], dim=0)  # (6,H,W)
    return x, to_tensor01(clean_rgb)

def read_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class SPTrainDataset(Dataset):
    def __init__(self, clean_paths, patch=64, use_mask_channel=False):
        self.paths = clean_paths
        self.patch = patch
        self.use_mask_channel = use_mask_channel

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = read_rgb(self.paths[idx])

        H, W, _ = img.shape
        p = self.patch
        if H < p or W < p:
            # büyütme dağılımı bozar (interpolasyon); sessizce yapmak yerine hata ver
            raise ValueError(f"Görsel ({H}x{W}) patch={p} için küçük: {self.paths[idx]}")

        y0 = np.random.randint(0, H - p + 1)
        x0 = np.random.randint(0, W - p + 1)
        clean = img[y0:y0+p, x0:x0+p].copy()

        density = np.random.uniform(0.1, 0.9)
        return make_input(clean, density, self.use_mask_channel)


def weighted_l1_loss(pred, target, mask_bad=None, mask_weight=6.0):
    err = (pred - target).abs()
    if mask_bad is None:
        return err.mean()

    weight = 1.0 + mask_bad * (mask_weight - 1.0)
    return (err * weight).sum() / weight.sum().clamp_min(1.0)

def build_model(cfg):
    if cfg.model_name.lower() == "dncnn":
        return DnCNN(channels=3)
    if cfg.model_name.lower() == "dncnn_sp":
        return DnCNNSP(channels=3)
    if cfg.model_name.lower() == "seconv_unet":
        return SeConvUNet(base=48)

    if cfg.external_module_path and cfg.external_class_name:
        return ExternalModelWrapper(cfg.external_module_path, cfg.external_class_name, cfg.external_kwargs)
    raise ValueError(f"Unknown model_name={cfg.model_name}. External info missing?")

def build_val_set(cfg):
    """Sabit seed'li doğrulama seti: val görüntüleri x 5 gürültü seviyesi (test setinden ayrı)."""
    if not cfg.val_dir:
        return None
    paths = sorted(glob.glob(os.path.join(cfg.val_dir, "*.JPEG")))[:cfg.val_count]
    if not paths:
        raise RuntimeError(f"Val images bulunamadı: {cfg.val_dir}")
    clean = [read_rgb(p) for p in paths]
    val = {}
    for i, d in enumerate(VAL_LEVELS):
        rng = np.random.default_rng(cfg.seed + 1000 + i)
        xs, ys = zip(*(make_input(c, d, cfg.use_mask_channel, rng) for c in clean))
        val[d] = (torch.stack(xs), torch.stack(ys))
    print(f"Val images: {len(paths)} x {len(VAL_LEVELS)} seviye")
    return val

@torch.no_grad()
def evaluate(model, val, device, bs=250):
    model.eval()
    out = {}
    for d, (X, Y) in val.items():
        psnrs = []
        for i in range(0, len(X), bs):
            pred = model(X[i:i+bs].to(device)).clamp(0, 1)
            pred = torch.round(pred * 255.0)
            tgt = torch.round(Y[i:i+bs].to(device) * 255.0)
            mse = ((pred - tgt) ** 2).flatten(1).mean(1).clamp_min(1e-10)
            psnrs.append(10 * torch.log10(255.0 ** 2 / mse))
        out[d] = torch.cat(psnrs).mean().item()
    model.train()
    return out

def last_ckpt_path(cfg):
    root, ext = os.path.splitext(cfg.ckpt_path)
    return f"{root}_last{ext}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="<ckpt>_last.pth varsa kaldığı epoch'tan devam et")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(os.path.dirname(cfg.ckpt_path), exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clean_paths = sorted(glob.glob(os.path.join(cfg.train_dir, "**", "*.JPEG"), recursive=True))
    if len(clean_paths) == 0:
        raise RuntimeError(f"Train images bulunamadı: {cfg.train_dir}")
    print(f"Train images: {len(clean_paths)}")

    ds = SPTrainDataset(clean_paths, patch=cfg.patch, use_mask_channel=cfg.use_mask_channel)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    pin_memory=True, drop_last=True,
                    generator=torch.Generator().manual_seed(cfg.seed))
    val = build_val_set(cfg)

    model = build_model(cfg).to(device)
    print("device selected:", device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = None
    if cfg.lr_schedule == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    elif cfg.lr_schedule != "constant":
        raise ValueError(f"Bilinmeyen lr_schedule: {cfg.lr_schedule}")

    start_ep, best = 1, -float("inf")
    last_path = last_ckpt_path(cfg)
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        if sched is not None:
            sched.load_state_dict(ck["scheduler"])
        start_ep, best = ck["epoch"] + 1, ck["best_val"]
        torch.set_rng_state(ck["torch_rng"].cpu())  # map_location GPU'ya taşımış olabilir
        print(f"Resume: {last_path} (epoch {ck['epoch']}, best val PSNR {best:.3f})")

    for ep in range(start_ep, cfg.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in tqdm(dl, desc=f"Epoch {ep}/{cfg.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(x)
            mask_bad = x[:, 3:] if cfg.use_mask_channel else None
            loss = weighted_l1_loss(pred, y, mask_bad=mask_bad, mask_weight=cfg.mask_loss_weight)
            loss.backward()
            opt.step()
            tot += loss.item()

        avg = tot / max(1, len(dl))
        msg = f"Epoch {ep}: L1={avg:.6f}  lr={opt.param_groups[0]['lr']:.2e}"
        if sched is not None:
            sched.step()

        if val is not None:
            v = evaluate(model, val, device)
            score = float(np.mean(list(v.values())))
            msg += "  val PSNR " + " ".join(f"{int(d*100)}%={p:.2f}" for d, p in v.items()) + f"  ort={score:.3f}"
        else:
            score = float(ep)  # val yoksa son epoch en iyi sayılır

        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "epoch": ep, "val_psnr": score,
                        "noise_model": NOISE_MODEL}, cfg.ckpt_path)
            msg += "  [best kaydedildi]"
        print(msg, flush=True)

        torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "epoch": ep,
                    "best_val": best, "torch_rng": torch.get_rng_state(),
                    "scheduler": sched.state_dict() if sched is not None else None}, last_path)

    print(f"Saved best: {cfg.ckpt_path}  (val ort PSNR {best:.3f})")

if __name__ == "__main__":
    main()
