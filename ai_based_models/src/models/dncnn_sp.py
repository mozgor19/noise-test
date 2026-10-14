import torch
import torch.nn as nn


class DnCNNSP(nn.Module):
    """
    Maske koşullu DnCNN: girdi = RGB + kanal bazlı S&P maskesi (1=şüpheli).
    Doğrudan tahmin yapar; yalnızca maskeli kanal elemanları değiştirilir,
    güvenilir kanallar girdiden aynen kopyalanır.
    """

    def __init__(self, channels: int = 3, mask_channels: int = 3, num_layers: int = 17, features: int = 64):
        super().__init__()
        self.channels = channels
        layers: list = [
            nn.Conv2d(channels + mask_channels, features, kernel_size=3, padding=1, bias=False),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb  = x[:, :self.channels]
        mask = x[:, self.channels:].clamp(0, 1)

        pred = self.net(x).clamp(0.0, 1.0)

        return pred * mask + rgb * (1.0 - mask)
