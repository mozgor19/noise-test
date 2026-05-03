import os, argparse
import torch
from tqdm import tqdm

from utils.config import load_config
from utils.io import list_images, read_rgb, write_rgb, ensure_dir
from utils.spmask import to_tensor01, to_uint8, sp_mask_fixed, mask_to_tensor

from models.dncnn import DnCNN
from models.dncnn_sp import DnCNNSP
from models.seconv_unet import SeConvUNet
from models.external_wrappers import ExternalModelWrapper, load_checkpoint

NOISE_LEVELS = [10, 30, 50, 70, 90]

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

@torch.no_grad()
def run_infer(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    if not os.path.exists(cfg.ckpt_path):
        raise FileNotFoundError(f"Checkpoint bulunamadı: {cfg.ckpt_path}")
    model = load_checkpoint(model, cfg.ckpt_path, device, strict=False)
    model.eval()

    ensure_dir(cfg.out_root)

    for level in NOISE_LEVELS:
        in_dir = os.path.join(cfg.noisy_root, f"Tiny_Noisy_{level}")
        if not os.path.isdir(in_dir):
            print(f"[WARN] Yok: {in_dir} (atlanıyor)")
            continue

        out_dir = os.path.join(cfg.out_root, f"Tiny_Cleaned_{cfg.model_tag}_{level}")
        ensure_dir(out_dir)

        paths = list_images(in_dir, exts=(".png",))
        print(f"\n{cfg.model_tag} | {level}% | {len(paths)} image")

        for p in tqdm(paths):
            rgb = read_rgb(p)
            x = to_tensor01(rgb).unsqueeze(0).to(device)

            if cfg.use_mask_channel:
                m = sp_mask_fixed(rgb)               # (H,W,1)
                m_t = mask_to_tensor(m).unsqueeze(0).to(device)  # (1,1,H,W)
                x_in = torch.cat([x, m_t], dim=1)    # (1,4,H,W)
            else:
                x_in = x

            y = model(x_in).clamp(0, 1)
            out = to_uint8(y)
            out_path = os.path.join(out_dir, os.path.basename(p))
            write_rgb(out_path, out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--noisy_root", default=None, help="VM’de Tiny_Noisy_* klasörlerinin kökü")
    ap.add_argument("--out_root", default=None, help="outputs kökü")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.noisy_root is not None:
        cfg.noisy_root = args.noisy_root
    if args.out_root is not None:
        cfg.out_root = args.out_root

    run_infer(cfg)
    print("[DONE]")

if __name__ == "__main__":
    main()