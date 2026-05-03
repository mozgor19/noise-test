import os, glob, argparse
import numpy as np
import cv2
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.config import load_config
from utils.spmask import to_tensor01, mask_to_tensor

from models.dncnn import DnCNN
from models.dncnn_sp import DnCNNSP
from models.seconv_unet import SeConvUNet
from models.external_wrappers import ExternalModelWrapper, load_checkpoint

class SPTrainDataset(Dataset):
    def __init__(self, clean_paths, patch=128, use_mask_channel=False):
        self.paths = clean_paths
        self.patch = patch
        self.use_mask_channel = use_mask_channel

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        if img is None:
            raise FileNotFoundError(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        H, W, _ = img.shape
        p = self.patch

        if H < p or W < p:
            img = cv2.resize(img, (max(W, p), max(H, p)), interpolation=cv2.INTER_AREA)
            H, W, _ = img.shape

        y0 = np.random.randint(0, H - p + 1)
        x0 = np.random.randint(0, W - p + 1)
        clean = img[y0:y0+p, x0:x0+p].copy()

        density = np.random.uniform(0.1, 0.9)
        noisy = clean.copy()
        mask = np.zeros((p, p, 1), dtype=np.float32)

        num = int(np.ceil(density * p * p))
        ys = np.random.randint(0, p, size=num)
        xs = np.random.randint(0, p, size=num)
        salt = np.random.rand(num) < 0.5
        noisy[ys, xs] = np.where(salt[:, None], 255, 0)
        mask[ys, xs, 0] = 1.0

        y = to_tensor01(clean)     # (3,p,p)
        x = to_tensor01(noisy)     # (3,p,p)

        if self.use_mask_channel:
            m_t = mask_to_tensor(mask)    # (1,p,p)
            x = torch.cat([x, m_t], dim=0)  # (4,p,p)

        return x, y


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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="ckpt_path varsa yükleyip devam et")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(os.path.dirname(cfg.ckpt_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clean_paths = glob.glob(os.path.join(cfg.train_dir, "**", "*.JPEG"), recursive=True)
    if len(clean_paths) == 0:
        raise RuntimeError(f"Train images bulunamadı: {cfg.train_dir}")
    print(f"Train images: {len(clean_paths)}")

    ds = SPTrainDataset(clean_paths, patch=cfg.patch, use_mask_channel=cfg.use_mask_channel)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    pin_memory=True, drop_last=True)

    model = build_model(cfg).to(device)
    print("cuda available:", torch.cuda.is_available())
    print("torch cuda version:", torch.version.cuda)
    print("device selected:", device)
    print("model on cuda:", next(model.parameters()).is_cuda)
    if args.resume and os.path.exists(cfg.ckpt_path):
        print(f"Resume: {cfg.ckpt_path}")
        model = load_checkpoint(model, cfg.ckpt_path, device, strict=False)

    opt = optim.AdamW(model.parameters(), lr=cfg.lr)
    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in tqdm(dl, desc=f"Epoch {ep}/{cfg.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(x)
            mask_bad = x[:, 3:4] if cfg.use_mask_channel and x.shape[1] > 3 else None
            loss = weighted_l1_loss(pred, y, mask_bad=mask_bad, mask_weight=cfg.mask_loss_weight)
            loss.backward()
            opt.step()
            tot += loss.item()

        avg = tot / max(1, len(dl))
        print(f"Epoch {ep}: L1={avg:.6f}")

        torch.save({"state_dict": model.state_dict()}, cfg.ckpt_path)

    print(f"Saved: {cfg.ckpt_path}")

if __name__ == "__main__":
    main()
