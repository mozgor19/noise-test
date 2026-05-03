import torch.nn as nn

class DnCNN(nn.Module):
    """
    DnCNN residual denoiser:
    pred_noise = net(x), output = x - pred_noise
    """
    def __init__(self, channels=3, num_of_layers=17, features=64):
        super().__init__()
        k = 3
        p = 1
        layers = [
            nn.Conv2d(channels, features, k, padding=p, bias=False),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_of_layers - 2):
            layers += [
                nn.Conv2d(features, features, k, padding=p, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(features, channels, k, padding=p, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.net(x)
        return (x - noise).clamp(0.0, 1.0)