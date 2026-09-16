import torch
import torch.nn as nn

CH_BOOST = 2

class InputMix(nn.Module):
    """
    Проекция + SE (опционально) + learnable per-channel scale.
    Точная копия из обучающего кода!
    """
    def __init__(self, in_ch, use_se=True, se_reduction=8, ch_indices=None, ch_boost=CH_BOOST):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.use_se = use_se
        if use_se:
            mid = max(1, in_ch // se_reduction)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch, mid, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, in_ch, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.se = None
            
        self.scale = nn.Parameter(torch.ones(1, in_ch, 1, 1))
        self.ch_boost = ch_boost

    def forward(self, x):
        x = self.proj(x)
        if self.se is not None:
            w = self.se(x)
            x = x * w
        x = x * self.scale
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=8, out_ch=1, base_filters=64):
        super().__init__()
        self.input_mix = InputMix(in_ch, use_se=True, ch_boost=CH_BOOST)
        
        bf = base_filters
        # encoder
        self.down1 = DoubleConv(in_ch, bf)
        self.down2 = DoubleConv(bf, bf*2)
        self.down3 = DoubleConv(bf*2, bf*4)
        self.down4 = DoubleConv(bf*4, bf*8)
        self.down5 = DoubleConv(bf*8, bf*8)
        self.pool = nn.MaxPool2d(2)

        # decoder
        self.uptrans1 = nn.ConvTranspose2d(bf*8, bf*8, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(bf*8 + bf*8, bf*4)
        
        self.uptrans2 = nn.ConvTranspose2d(bf*4, bf*4, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(bf*4 + bf*4, bf*2)
        
        self.uptrans3 = nn.ConvTranspose2d(bf*2, bf*2, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(bf*2 + bf*2, bf)
        
        self.uptrans4 = nn.ConvTranspose2d(bf, bf, kernel_size=2, stride=2)
        self.upconv4 = DoubleConv(bf + bf, bf)
        
        self.out_conv = nn.Conv2d(bf, out_ch, 1)

    def forward(self, x):
        x = self.input_mix(x)
        
        c1 = self.down1(x)
        p1 = self.pool(c1)
        c2 = self.down2(p1)
        p2 = self.pool(c2)
        c3 = self.down3(p2)
        p3 = self.pool(c3)
        c4 = self.down4(p3)
        p4 = self.pool(c4)
        c5 = self.down5(p4)

        u1 = self.uptrans1(c5)
        u1 = torch.cat([u1, c4], dim=1)
        u1 = self.upconv1(u1)

        u2 = self.uptrans2(u1)
        u2 = torch.cat([u2, c3], dim=1)
        u2 = self.upconv2(u2)

        u3 = self.uptrans3(u2)
        u3 = torch.cat([u3, c2], dim=1)
        u3 = self.upconv3(u3)

        u4 = self.uptrans4(u3)
        u4 = torch.cat([u4, c1], dim=1)
        u4 = self.upconv4(u4)

        out = self.out_conv(u4)
        return out