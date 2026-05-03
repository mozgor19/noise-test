import torch
import torch.nn as nn


class DnCNNSP(nn.Module):

    def __init__(self, channels: int = 3, num_layers: int = 17, features: int = 64):
        super().__init__()
        layers: list = [
            nn.Conv2d(channels + 1, features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x4: torch.Tensor) -> torch.Tensor:
        rgb  = x4[:, :3]
        mask = x4[:, 3:4].clamp(0, 1)

        pred = self.net(x4).clamp(0.0, 1.0)

        return pred * mask + rgb * (1.0 - mask)
