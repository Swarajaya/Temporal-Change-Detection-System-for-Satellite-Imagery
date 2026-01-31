import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Basic Conv Block
# ------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

# ------------------------------
# Encoder
# ------------------------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(3, 64)
        self.c2 = ConvBlock(64, 128)
        self.c3 = ConvBlock(128, 256)
        self.c4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(self.pool(x1))
        x3 = self.c3(self.pool(x2))
        x4 = self.c4(self.pool(x3))
        return x4, [x3, x2, x1]

# ------------------------------
# Decoder
# ------------------------------
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.d3 = ConvBlock(512 + 256, 256)
        self.d2 = ConvBlock(256 + 128, 128)
        self.d1 = ConvBlock(128 + 64, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2)
        x = self.d3(torch.cat([x, skips[0]], dim=1))

        x = F.interpolate(x, scale_factor=2)
        x = self.d2(torch.cat([x, skips[1]], dim=1))

        x = F.interpolate(x, scale_factor=2)
        x = self.d1(torch.cat([x, skips[2]], dim=1))

        return torch.sigmoid(self.out(x))

# ------------------------------
# Siamese U-Net
# ------------------------------
class SiameseUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, t1, t2):
        f1, s1 = self.encoder(t1)
        f2, s2 = self.encoder(t2)

        diff = torch.abs(f1 - f2)
        skips = [
            torch.abs(s1[0] - s2[0]),
            torch.abs(s1[1] - s2[1]),
            torch.abs(s1[2] - s2[2]),
        ]

        return self.decoder(diff, skips)
