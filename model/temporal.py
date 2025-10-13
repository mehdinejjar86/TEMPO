# temporal.py
#  modules for temporal modeling
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Temporal Position Encoding
# =========================

class TemporalPositionEncoding(nn.Module):
    """
    Sinusoidal temporal encoding with fixed bases, learnable projection.
    """
    def __init__(self, channels=64, max_freq=10):
        super().__init__()
        assert channels % 2 == 0, "channels should be even"
        self.channels = channels
        self.max_freq = max_freq
        freq_bands = 2.0 ** torch.linspace(0, max_freq - 1, max_freq)
        self.register_buffer('freq_bands', freq_bands)
        in_dim = max_freq * 2
        self.proj = nn.Linear(in_dim, channels) if in_dim != channels else nn.Identity()

    def forward(self, t):  # t: [B,N] or [B,1]
        t = t.unsqueeze(-1)  # [B,N,1]
        feats = []
        for f in self.freq_bands:
            feats.append(torch.sin(2 * math.pi * f * t))
            feats.append(torch.cos(2 * math.pi * f * t))
        enc = torch.cat(feats, dim=-1)           # [B,N,2*max_freq]
        return self.proj(enc)                    # [B,N,C]
    
# ==========================
# Temporal Weighter (adaptive, bimodal prior)
# ==========================

class TemporalWeighter(nn.Module):
    """
    Softmax over anchors combining:
      - MLP on sinusoid encodings
      - Adaptive Gaussian distance prior (sharpness α predicted from gap stats)
      - Optional bimodal prior mass on nearest left/right when bracketing exists
    """
    def __init__(self, temporal_channels=64, use_bimodal=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(temporal_channels, temporal_channels),
            nn.ReLU(inplace=True),
            nn.Linear(temporal_channels, 1)
        )
        self.use_bimodal = use_bimodal
        # predict alpha from simple gap stats (max gap, median gap, mean |Δt|)
        self.alpha_head = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        # temperature for bimodal mixing
        self.mix_logit = nn.Parameter(torch.tensor(1.0))

    def forward(self, rel_enc, rel_scalar, anchor_times, target_time):
        """
        rel_enc   : [B,N,Ct]
        rel_scalar: [B,N]   normalized Δt
        anchor_times: [B,N] raw times
        target_time: [B,1]
        returns weights [B,N], plus priors for diagnostics
        """
        B, N, _ = rel_enc.shape
        # encoding score
        s1 = self.mlp(rel_enc).squeeze(-1)  # [B,N]

        # adaptive α from gap stats
        with torch.no_grad():
            at = anchor_times
            if N > 1:
                sorted_at, _ = at.sort(dim=1)
                gaps = sorted_at[:, 1:] - sorted_at[:, :-1]  # [B,N-1]
                max_gap = gaps.max(dim=1).values.unsqueeze(-1)
                med_gap = gaps.median(dim=1).values.unsqueeze(-1)
            else:
                max_gap = torch.zeros(B, 1, device=at.device)
                med_gap = torch.zeros(B, 1, device=at.device)
            mean_abs = (anchor_times - target_time).abs().mean(dim=1, keepdim=True)
            stats = torch.cat([max_gap, med_gap, mean_abs], dim=-1)  # [B,3]
        alpha = F.softplus(self.alpha_head(stats)).clamp(min=1e-3)    # [B,1]
        s2 = - alpha * (rel_scalar ** 2)                              # [B,N]

        logits = s1 + s2

        # bimodal prior (left/right) if target is bracketed by anchors
        if self.use_bimodal:
            left_mask  = (anchor_times <= target_time)  # [B,N]
            right_mask = (anchor_times >= target_time)
            big = torch.tensor(1e6, device=anchor_times.device)
            dist_left  = torch.where(left_mask,  (target_time - anchor_times).abs(), big)   # [B,N]
            dist_right = torch.where(right_mask, (anchor_times - target_time).abs(), big)
            iL = dist_left.argmin(dim=1)   # [B]
            iR = dist_right.argmin(dim=1)  # [B]
            prior = torch.zeros_like(logits)
            prior[torch.arange(B, device=anchor_times.device), iL] += 1.0
            prior[torch.arange(B, device=anchor_times.device), iR] += 1.0
            mix = torch.sigmoid(self.mix_logit)
            logits = logits + mix * prior

        weights = F.softmax(logits, dim=1)
        priors = {"alpha": alpha.detach(), "bimix": torch.sigmoid(self.mix_logit).detach()}
        return weights, priors