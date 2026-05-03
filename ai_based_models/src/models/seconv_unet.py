import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.k = k
        self.p = p
        self.kernel_area = float(k * k)
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=p, bias=True)
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.conv.bias)
        self.register_buffer("mask_kernel", torch.ones((1, 1, k, k)))

    def forward(self, x, mask_bad):
        valid = 1.0 - mask_bad
        xw = x * valid
        num = self.conv(xw)

        denom = F.conv2d(valid, self.mask_kernel.to(dtype=x.dtype), padding=self.p)
        update_mask = (denom > 0).to(x.dtype)
        denom = torch.clamp(denom, min=1.0)

        bias = self.conv.bias.view(1, -1, 1, 1)
        num = (num - bias) * (self.kernel_area / denom) + bias
        return num * update_mask

class SCBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.sc1 = SelectiveConv2d(in_ch, out_ch)
        self.sc2 = SelectiveConv2d(out_ch, out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, mask_bad):
        x = self.act(self.sc1(x, mask_bad))
        x = self.act(self.sc2(x, mask_bad))
        return x

class SeConvUNet(nn.Module):
    def __init__(self, base=48):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = SCBlock(3, base)
        self.enc2 = SCBlock(base, base*2)
        self.enc3 = SCBlock(base*2, base*4)

        self.mid  = SCBlock(base*4, base*8)

        self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = SCBlock(base*8, base*4)

        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = SCBlock(base*4, base*2)

        self.up1  = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = SCBlock(base*2, base)

        self.out  = nn.Conv2d(base, 3, 1)

    def forward(self, x4):
        rgb = x4[:, :3]
        mask_bad = x4[:, 3:4].clamp(0, 1)  # 1=bad

        # downsample masks to match feature scales
        m1 = mask_bad
        m2 = F.avg_pool2d(m1, 2)
        m3 = F.avg_pool2d(m2, 2)
        m4 = F.avg_pool2d(m3, 2)

        e1 = self.enc1(rgb, m1)
        e2 = self.enc2(self.pool(e1), m2)
        e3 = self.enc3(self.pool(e2), m3)

        b  = self.mid(self.pool(e3), m4)

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3, m3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2, m2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1, m1)

        repaired = self.out(d1)

        y = repaired * mask_bad + rgb * (1.0 - mask_bad)
        return y.clamp(0.0, 1.0)
