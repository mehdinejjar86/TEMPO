# film.py
#  modules for Feature-wise Linear Modulation (FiLM) based temporal conditioning
from model.utility import ConvBlock, ResBlock
import torch
import torch.nn as nn

# ==========================
# Temporal FiLM conditioning
# ==========================

class TemporalFiLM(nn.Module):
    def __init__(self, feature_channels, temporal_channels):
        super().__init__()
        self.gamma_net = nn.Linear(temporal_channels, feature_channels)
        self.beta_net  = nn.Linear(temporal_channels, feature_channels)

    def forward(self, features, temporal_encoding):  # features: [B,C,H,W], enc: [B,Ct]
        gamma = self.gamma_net(temporal_encoding).unsqueeze(-1).unsqueeze(-1)
        beta  = self.beta_net(temporal_encoding).unsqueeze(-1).unsqueeze(-1)
        return gamma * features + beta


# ==========================
# Frame encoder (FiLM at scales)
# ==========================

class FrameEncoder(nn.Module):
    def __init__(self, base_channels=64, temporal_channels=64):
        super().__init__()
        C = base_channels
        self.conv1 = ConvBlock(3, C, 7, 1, 3)
        self.film1 = TemporalFiLM(C, temporal_channels)

        self.down1 = ConvBlock(C, C*2, 3, 2, 1)
        self.res1  = ResBlock(C*2)
        self.film2 = TemporalFiLM(C*2, temporal_channels)

        self.down2 = ConvBlock(C*2, C*4, 3, 2, 1)
        self.res2  = ResBlock(C*4)
        self.film3 = TemporalFiLM(C*4, temporal_channels)

        self.down3 = ConvBlock(C*4, C*8, 3, 2, 1)
        self.res3  = ResBlock(C*8)
        self.res4  = ResBlock(C*8)
        self.film4 = TemporalFiLM(C*8, temporal_channels)

    def forward(self, frame, temporal_encoding):
        f1 = self.conv1(frame); f1 = self.film1(f1, temporal_encoding)
        f2 = self.down1(f1);    f2 = self.res1(f2);   f2 = self.film2(f2, temporal_encoding)
        f3 = self.down2(f2);    f3 = self.res2(f3);   f3 = self.film3(f3, temporal_encoding)
        f4 = self.down3(f3);    f4 = self.res3(f4);   f4 = self.res4(f4); f4 = self.film4(f4, temporal_encoding)
        return f1, f2, f3, f4


# ==========================
# Decoder with target-time & speed token
# ==========================

class DecoderTimeFiLM(nn.Module):
    def __init__(self, base_channels=64, temporal_channels=64):
        super().__init__()
        C = base_channels
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_up1 = nn.Sequential(ConvBlock(C*8, C*4, 3, 1, 1), ResBlock(C*4))
        self.film_up1 = TemporalFiLM(C*4, temporal_channels)

        # conv_up2 expects input at 1/4 scale: (x: C*4) + (f3: C*4) -> C*2
        self.conv_up2 = nn.Sequential(ConvBlock(C*4 + C*4, C*2, 3, 1, 1), ResBlock(C*2))
        self.film_up2 = TemporalFiLM(C*2, temporal_channels)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # conv_up3 expects input at 1/2 scale: (x: C*2) + (f2: C*2) -> C
        self.conv_up3 = nn.Sequential(ConvBlock(C*2 + C*2, C, 3, 1, 1), ResBlock(C))
        self.film_up3 = TemporalFiLM(C, temporal_channels)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.final = nn.Sequential(
            ConvBlock(C + C, C, 3, 1, 1),
            ResBlock(C), ResBlock(C),
            ConvBlock(C, 3, 3, 1, 1, norm='none', activation='sigmoid')
        )

    def forward(self, f1, f2, f3, f4, tenc_with_speed):
        # f4: 1/8 → up to 1/4
        x = self.up1(f4)
        x = self.conv_up1(x)
        x = self.film_up1(x, tenc_with_speed)     # still 1/4

        # concat with 1/4 skip (f3), THEN conv + FiLM, THEN up to 1/2
        x = torch.cat([x, f3], dim=1)             # 1/4
        x = self.conv_up2(x)                      # 1/4
        x = self.film_up2(x, tenc_with_speed)     # 1/4
        x = self.up2(x)                           # 1/2

        # concat with 1/2 skip (f2), THEN conv + FiLM, THEN up to full
        x = torch.cat([x, f2], dim=1)             # 1/2
        x = self.conv_up3(x)                      # 1/2
        x = self.film_up3(x, tenc_with_speed)     # 1/2
        x = self.up3(x)                           # 1× (full)

        # final concat with full-res skip (f1)
        x = torch.cat([x, f1], dim=1)             # full
        return self.final(x)