# tempo.py
# TEMPO v2: Temporal Multi-View Frame Synthesis with Time-Aware Pyramid Attention (no optical flow)
# Includes fixes:
#  - Δt tiling across windows for attention bias
#  - 4D reference grid for grid_sample (batch sizes match)
#  - Vectorized scene-cut fallback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from model.film import FrameEncoder, DecoderTimeFiLM
from model.pyramid import TimeAwareSlidingWindowPyramidAttention
from model.temporal import TemporalPositionEncoding, TemporalWeighter
from model.utility import CutDetector

# ==========================
# TEMPO v2 (time-aware attention, robustness)
# ==========================

class TEMPO(nn.Module):
    def __init__(self, base_channels=64, temporal_channels=64,
                 attn_heads=4, attn_points=4, attn_levels_max=6,
                 window_size=8, shift_size=0, dt_bias_gain=1.0, max_offset_scale=1.5,
                 cut_thresh=0.35):
        super().__init__()
        C = base_channels
        Ct = temporal_channels

        # time enc
        self.temporal_encoder = TemporalPositionEncoding(channels=Ct)

        # encoder
        self.encoder = FrameEncoder(C, Ct)

        # attention fusion per scale
        mk = lambda ch: TimeAwareSlidingWindowPyramidAttention(
            ch, attn_heads, attn_points, min(attn_levels_max, 8),
            window_size, shift_size, temporal_channels=Ct,
            dt_bias_gain=dt_bias_gain, max_offset_scale=max_offset_scale
        )
        self.fuse1 = mk(C)
        self.fuse2 = mk(C*2)
        self.fuse3 = mk(C*4)
        self.fuse4 = mk(C*8)

        # temporal weighter
        self.weighter = TemporalWeighter(Ct, use_bimodal=True)

        # decoder (+1 for speed token)
        self.decoder = DecoderTimeFiLM(C, Ct + 1)

        # cut detector
        self.cut_detector = CutDetector(thresh=cut_thresh)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None: init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None: init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None: init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: init.constant_(m.bias, 0)

    def _build_speed_token(self, anchor_times, target_time):
        # scalar describing global spacing; here use median gap / (span + eps)
        B, N = anchor_times.shape
        at_sorted, _ = anchor_times.sort(dim=1)
        gaps = at_sorted[:, 1:] - at_sorted[:, :-1] if N > 1 else torch.zeros(B, 0, device=anchor_times.device)
        span = (anchor_times.max(dim=1, keepdim=True).values -
                anchor_times.min(dim=1, keepdim=True).values).clamp(min=1e-6)
        if gaps.numel() > 0:
            med_gap = gaps.median(dim=1).values.view(B, 1)
        else:
            med_gap = torch.zeros(B, 1, device=anchor_times.device)
        speed = (med_gap / span).clamp(0, 1)  # [B,1]
        return speed

    def forward(self, frames, anchor_times, target_time):
        """
        frames: [B,N,3,H,W]
        anchor_times: [B,N] (not necessarily even)
        target_time : [B] or [B,1]
        """
        B, N, _, H, W = frames.shape
        device = frames.device
        if target_time.dim() == 1: target_time = target_time.unsqueeze(1)

        # ----- time normalization -----
        span = (anchor_times.max(dim=1, keepdim=True).values -
                anchor_times.min(dim=1, keepdim=True).values).clamp(min=1e-6)
        rel_norm = (anchor_times - target_time) / span
        rel_norm = rel_norm.clamp(-2.0, 2.0)

        rel_enc = self.temporal_encoder(rel_norm)    # [B,N,Ct]
        weights, prior_info = self.weighter(rel_enc, rel_norm, anchor_times, target_time)  # [B,N]

        # ----- encode views -----
        feats1, feats2, feats3, feats4 = [], [], [], []
        low_feats = []
        for i in range(N):
            f1, f2, f3, f4 = self.encoder(frames[:, i], rel_enc[:, i])
            feats1.append(f1); feats2.append(f2); feats3.append(f3); feats4.append(f4)
            low_feats.append(f4)

        # ----- scene cut robustness (vectorized) -----
        lf = torch.stack(low_feats, dim=1)  # [B,N,C8,H/8,W/8]
        nearest_idx = (anchor_times - target_time).abs().argmin(dim=1)  # [B]
        dir_right = (anchor_times.gather(1, nearest_idx.unsqueeze(1)) <= target_time).squeeze(1)  # [B] bool
        alt_idx = nearest_idx + torch.where(dir_right, torch.tensor(1, device=device), torch.tensor(-1, device=device))
        alt_idx = alt_idx.clamp(0, N - 1)
        batch_ids = torch.arange(B, device=device)
        f_near = lf[batch_ids, nearest_idx]  # [B,C8,H/8,W/8]
        f_alt  = lf[batch_ids, alt_idx]      # [B,C8,H/8,W/8]
        cut_mask, cut_score = self.cut_detector(f_near, f_alt)  # [B], [B]
        use_fallback = cut_mask > 0.5

        # ----- fusion per scale -----
        s1 = torch.stack(feats1, dim=1)  # [B,N,C,H,W]
        s2 = torch.stack(feats2, dim=1)
        s3 = torch.stack(feats3, dim=1)
        s4 = torch.stack(feats4, dim=1)

        wv = weights.view(B, N, 1, 1, 1)
        q1 = (s1 * wv).sum(dim=1)
        q2 = (s2 * wv).sum(dim=1)
        q3 = (s3 * wv).sum(dim=1)
        q4 = (s4 * wv).sum(dim=1)

        tgt_zero_enc = self.temporal_encoder(torch.zeros(B, 1, device=device))[:, 0]  # encode Δt=0 once

        fused1, conf1, ent1 = self.fuse1(q1, s1, rel_norm, tgt_zero_enc)
        fused2, conf2, ent2 = self.fuse2(q2, s2, rel_norm, tgt_zero_enc)
        fused3, conf3, ent3 = self.fuse3(q3, s3, rel_norm, tgt_zero_enc)
        fused4, conf4, ent4 = self.fuse4(q4, s4, rel_norm, tgt_zero_enc)

        if use_fallback.any():
            idx = torch.nonzero(use_fallback).squeeze(1)
            fused1[idx] = s1[idx, nearest_idx[idx]]
            fused2[idx] = s2[idx, nearest_idx[idx]]
            fused3[idx] = s3[idx, nearest_idx[idx]]
            fused4[idx] = s4[idx, nearest_idx[idx]]

        # ----- better targets at decoder: concat speed token -----
        speed   = self._build_speed_token(anchor_times, target_time)           # [B,1]
        tenc_with_speed = torch.cat([tgt_zero_enc, speed], dim=-1)  # speed is [B,1]


        # ----- decode -----
        out = self.decoder(fused1, fused2, fused3, fused4, tenc_with_speed)

        def _to_full(x, H, W):
            return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        conf1_f = _to_full(conf1, H, W)
        conf2_f = _to_full(conf2, H, W)
        conf3_f = _to_full(conf3, H, W)
        conf4_f = _to_full(conf4, H, W)

        ent1_f = _to_full(ent1, H, W)
        ent2_f = _to_full(ent2, H, W)
        ent3_f = _to_full(ent3, H, W)
        ent4_f = _to_full(ent4, H, W)

        conf = torch.clamp_min(torch.stack([conf1_f, conf2_f, conf3_f, conf4_f], dim=0).mean(dim=0), 1e-4)
        entropy = torch.stack([ent1_f, ent2_f, ent3_f, ent4_f], dim=0).mean(dim=0)

        aux = {
            "weights": weights.detach(),
            "prior_alpha": prior_info["alpha"],
            "prior_bimix": prior_info["bimix"],
            "conf_map": conf.detach(),
            "attn_entropy": entropy.detach(),
            "cut_score": cut_score.detach(),
            "fallback_mask": use_fallback.detach().float()
        }
        return out, aux


# ==========================
# Factory
# ==========================

def build_tempo(base_channels=64, temporal_channels=64,
                   attn_heads=4, attn_points=4, attn_levels_max=6,
                   window_size=8, shift_size=0, dt_bias_gain=1.0, max_offset_scale=1.5,
                   cut_thresh=0.35):
    return TEMPO(base_channels, temporal_channels, attn_heads, attn_points, attn_levels_max,
                 window_size, shift_size, dt_bias_gain, max_offset_scale, cut_thresh)



