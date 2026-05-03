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

DATASET_PROFILES = {
    "tiny": {
        "noisy_tmpl":   "Tiny_Noisy_{}",
        "cleaned_tmpl": "Tiny_Cleaned_{tag}_{level}",
    },
    "480p": {
        "noisy_tmpl":   "Noisy_{}",
        "cleaned_tmpl": "Cleaned_{tag}_{level}",
    },
}

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
def run_infer(cfg, dataset="tiny"):
    profile = DATASET_PROFILES[dataset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    if not os.path.exists(cfg.ckpt_path):
        raise FileNotFoundError(f"Checkpoint bulunamadı: {cfg.ckpt_path}")
    model = load_checkpoint(model, cfg.ckpt_path, device, strict=False)
    model.eval()

    ensure_dir(cfg.out_root)

    for level in NOISE_LEVELS:
        noisy_folder = profile["noisy_tmpl"].format(level)
        in_dir = os.path.join(cfg.noisy_root, noisy_folder)
        if not os.path.isdir(in_dir):
            print(f"[WARN] Yok: {in_dir} (atlanıyor)")
            continue

        cleaned_folder = profile["cleaned_tmpl"].format(tag=cfg.model_tag, level=level)
        out_dir = os.path.join(cfg.out_root, cleaned_folder)
        ensure_dir(out_dir)

        paths = list_images(in_dir, exts=(".png",))
        print(f"\n{cfg.model_tag} | {level}% | {len(paths)} image | dataset={dataset}")

        for p in tqdm(paths):
            rgb = read_rgb(p)
            x = to_tensor01(rgb).unsqueeze(0).to(device)

            if cfg.use_mask_channel:
                m = sp_mask_fixed(rgb)
                m_t = mask_to_tensor(m).unsqueeze(0).to(device)
                x_in = torch.cat([x, m_t], dim=1)
            else:
                x_in = x

            y = model(x_in).clamp(0, 1)
            out = to_uint8(y)
            write_rgb(os.path.join(out_dir, os.path.basename(p)), out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--noisy_root", default=None)
    ap.add_argument("--out_root",   default=None)
    ap.add_argument("--dataset", default="tiny", choices=["tiny", "480p"],
                    help="'tiny' for Tiny-Imagenet, '480p' for 480p-Test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.noisy_root is not None:
        cfg.noisy_root = args.noisy_root
    if args.out_root is not None:
        cfg.out_root = args.out_root

    run_infer(cfg, dataset=args.dataset)
    print("[DONE]")

if __name__ == "__main__":
    main()
