import os
import yaml
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class RunConfig:
    # model
    model_name: str
    model_tag: str
    use_mask_channel: bool
    ckpt_path: str

    # train (optional)
    train_dir: str = "tiny-imagenet-200/train"
    epochs: int = 0
    batch_size: int = 0
    lr: float = 0.0
    patch: int = 128
    num_workers: int = 4
    mask_loss_weight: float = 6.0

    # inference (optional)
    noisy_root: str = "."
    out_root: str = "outputs"

    # external repo (optional)
    external_module_path: Optional[str] = None
    external_class_name: Optional[str] = None
    external_kwargs: Optional[Dict[str, Any]] = None

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _abspath_if_relative(base_dir: str, p: str) -> str:
    if p is None:
        return p
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))

def load_config(config_path: str) -> RunConfig:
    base_dir = os.path.dirname(os.path.abspath(config_path))
    d = load_yaml(config_path) or {}

    # normalize keys
    model_name = d["model_name"]
    model_tag = d.get("model_tag", model_name)
    use_mask_channel = bool(d.get("use_mask_channel", False))

    ckpt_path = d.get("ckpt_path", f"weights/{model_name}.pth")
    ckpt_path = _abspath_if_relative(base_dir, ckpt_path)

    # train
    train_dir = _abspath_if_relative(base_dir, d.get("train_dir", "tiny-imagenet-200/train"))
    epochs = int(d.get("epochs", 0))
    batch_size = int(d.get("batch_size", 0))
    lr = float(d.get("lr", 0.0))
    patch = int(d.get("patch", 128))
    num_workers = int(d.get("num_workers", 4))
    mask_loss_weight = float(d.get("mask_loss_weight", 6.0))

    # inference
    noisy_root = _abspath_if_relative(base_dir, d.get("noisy_root", "."))
    out_root = _abspath_if_relative(base_dir, d.get("out_root", "outputs"))

    # external
    external_module_path = d.get("external_module_path")
    external_class_name = d.get("external_class_name")
    external_kwargs = d.get("external_kwargs", None)

    return RunConfig(
        model_name=model_name,
        model_tag=model_tag,
        use_mask_channel=use_mask_channel,
        ckpt_path=ckpt_path,
        train_dir=train_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patch=patch,
        num_workers=num_workers,
        mask_loss_weight=mask_loss_weight,
        noisy_root=noisy_root,
        out_root=out_root,
        external_module_path=external_module_path,
        external_class_name=external_class_name,
        external_kwargs=external_kwargs,
    )
