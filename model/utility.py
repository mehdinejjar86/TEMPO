# utility.py
#  utility functions and basic building blocks for neural networks.

import torch.nn as nn
import torch.nn.functional as F

# =========================
# Utilities
# =========================

def global_pool(feat, type="mean"):
    # feat: [B,C,H,W]
    if type == "mean":
        return feat.mean(dim=(2, 3))
    elif type == "absmean":
        return feat.abs().mean(dim=(2, 3))
    else:
        return feat.mean(dim=(2, 3))
    

# ===================
# Basic building blocks
# ===================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1,
                 norm='gn', activation='leaky'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        if norm == 'gn':
            self.norm = nn.GroupNorm(min(32, max(1, out_ch // 4)), out_ch)
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(out_ch)
        else:
            self.norm = nn.Identity()
        if activation == 'leaky':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'none':
            self.act = nn.Identity()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels, activation='none')
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        return self.act(out)


# ==========================
# Scene Cut Detector
# ==========================

class CutDetector(nn.Module):
    """
    Cosine similarity on pooled features; if dissimilarity>thresh, we consider it a cut.
    """
    def __init__(self, thresh=0.35):
        super().__init__()
        self.thresh = thresh

    def forward(self, feats_left, feats_right):
        # feats_*: [B,C,H,W]
        gl = F.normalize(global_pool(feats_left), dim=-1)   # [B,C]
        gr = F.normalize(global_pool(feats_right), dim=-1)
        cos = (gl * gr).sum(dim=-1)                         # [B]
        dissim = (1 - cos) * 0.5 * 2                        # [B] in [0,2]
        return (dissim > self.thresh).float(), dissim       # [B], [B]