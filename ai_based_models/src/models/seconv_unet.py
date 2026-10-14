import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveConv2d(nn.Module):
    """
    Partial convolution: yalnızca güvenilir (valid=1) girdileri kullanır, kapsanan
    güvenilir girdi sayısına göre yeniden ölçekler ve güncellenmiş valid maskesini
    (pencerede en az bir güvenilir girdi varsa 1) bir sonraki katmana aktarır.

    valid: (B,1,H,W) veya (B,C_in,H,W)
    """
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.k = k
        self.p = p
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=p, bias=True)
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.conv.bias)
        self.register_buffer("mask_kernel", torch.ones((1, 1, k, k)))

    def forward(self, x, valid):
        c_in = x.shape[1]
        num = self.conv(x * valid)

        # penceredeki güvenilir girdi sayısı (kanal x konum)
        cover = F.conv2d(valid.sum(1, keepdim=True), self.mask_kernel.to(dtype=x.dtype), padding=self.p)
        cover = cover * (c_in / valid.shape[1])
        new_valid = (cover > 0).to(x.dtype)

        bias = self.conv.bias.view(1, -1, 1, 1)
        ratio = (c_in * self.k * self.k) / cover.clamp(min=1.0)
        out = ((num - bias) * ratio + bias) * new_valid
        return out, new_valid

class SCBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.sc1 = SelectiveConv2d(in_ch, out_ch)
        self.sc2 = SelectiveConv2d(out_ch, out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, valid):
        x, valid = self.sc1(x, valid)
        x = self.act(x)
        x, valid = self.sc2(x, valid)
        x = self.act(x)
        return x, valid

def _cat_valid(va, ca, vb, cb):
    # birleştirilen öznitelikler için kanal bazlı valid maskesi
    return torch.cat([va.expand(-1, ca, -1, -1), vb.expand(-1, cb, -1, -1)], dim=1)

class SeConvUNet(nn.Module):
    """
    Girdi: RGB + kanal bazlı S&P maskesi (1=şüpheli). Çıkış yalnızca maskeli
    kanal elemanlarında değişir.
    """
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

    def forward(self, x):
        rgb = x[:, :3]
        mask_bad = x[:, 3:].clamp(0, 1)  # (B,3,H,W) 1=şüpheli

        e1, v1 = self.enc1(rgb, 1.0 - mask_bad)
        e2, v2 = self.enc2(self.pool(e1), self.pool(v1))
        e3, v3 = self.enc3(self.pool(e2), self.pool(v2))

        b, vb = self.mid(self.pool(e3), self.pool(v3))

        d3 = self.up3(b)
        vu = F.interpolate(vb, scale_factor=2, mode="nearest")
        d3, vd = self.dec3(torch.cat([d3, e3], dim=1), _cat_valid(vu, d3.shape[1], v3, e3.shape[1]))

        d2 = self.up2(d3)
        vu = F.interpolate(vd, scale_factor=2, mode="nearest")
        d2, vd = self.dec2(torch.cat([d2, e2], dim=1), _cat_valid(vu, d2.shape[1], v2, e2.shape[1]))

        d1 = self.up1(d2)
        vu = F.interpolate(vd, scale_factor=2, mode="nearest")
        d1, _ = self.dec1(torch.cat([d1, e1], dim=1), _cat_valid(vu, d1.shape[1], v1, e1.shape[1]))

        repaired = self.out(d1)

        y = repaired * mask_bad + rgb * (1.0 - mask_bad)
        return y.clamp(0.0, 1.0)
