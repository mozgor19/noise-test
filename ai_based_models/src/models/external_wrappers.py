import importlib
import torch
import torch.nn as nn
from typing import Any, Dict, Optional

class ExternalModelWrapper(nn.Module):
    """
    Harici repo modelini tek tip arayüze sokar:
      forward(x_in) -> y
    x_in: (B,3,H,W) veya (B,4,H,W)
    """
    def __init__(self, module_path: str, class_name: str, kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        kwargs = kwargs or {}
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        self.net = cls(**kwargs)

    def forward(self, x):
        return self.net(x)

def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device, strict: bool = False) -> nn.Module:
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # strip "module." if present
    new_sd = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        new_sd[nk] = v

    model.load_state_dict(new_sd, strict=strict)
    return model