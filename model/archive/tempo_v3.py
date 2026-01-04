# tempo_v3.py
# TEMPO v3: ConvNeXt Encoder + NAFNet Decoder with AdaLN-Zero
# Hybrid architecture for maximum PSNR/SSIM

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from model.convnext_nafnet import ConvNeXtEncoder, NAFNetDecoder
from model.pyramid import TimeAwareSlidingWindowPyramidAttention
from model.temporal import TemporalPositionEncoding, TemporalWeighter
from model.utility import CutDetector


class TEMPOv3(nn.Module):
    """
    TEMPO v3: Temporal Multi-View Frame Synthesis
    
    Architecture:
        - ConvNeXt Encoder: Rich hierarchical features with AdaLN-Zero
        - TimeAware Pyramid Attention: Multi-view temporal fusion
        - NAFNet Decoder: Maximum reconstruction quality
    
    Improvements over v2:
        - ConvNeXt blocks with 7×7 depthwise convs (larger receptive field)
        - AdaLN-Zero conditioning (stable, precise temporal modulation)
        - NAFNet decoder with SimpleGate (minimal information loss)
        - Layer scale + stochastic depth (better training dynamics)
    """
    
    def __init__(
        self,
        base_channels: int = 64,
        temporal_channels: int = 64,
        # Encoder settings
        encoder_depths: list = None,
        encoder_drop_path: float = 0.1,
        # Decoder settings
        decoder_depths: list = None,
        decoder_drop_path: float = 0.05,
        # Attention settings
        attn_heads: int = 4,
        attn_points: int = 4,
        attn_levels_max: int = 6,
        window_size: int = 8,
        shift_size: int = 0,
        dt_bias_gain: float = 1.0,
        max_offset_scale: float = 1.5,
        # Robustness
        cut_thresh: float = 0.35
    ):
        super().__init__()
        
        C = base_channels
        Ct = temporal_channels
        
        # Default depths optimized for quality
        if encoder_depths is None:
            encoder_depths = [3, 3, 9, 3]  # ~18 encoder blocks
        if decoder_depths is None:
            decoder_depths = [2, 2, 2, 2]  # ~8 decoder blocks
        
        # Temporal encoding
        self.temporal_encoder = TemporalPositionEncoding(channels=Ct)
        
        # ConvNeXt Encoder
        self.encoder = ConvNeXtEncoder(
            base_channels=C,
            temporal_channels=Ct,
            depths=encoder_depths,
            drop_path_rate=encoder_drop_path
        )
        
        # Attention fusion per scale (same as v2)
        def make_attention(ch):
            return TimeAwareSlidingWindowPyramidAttention(
                ch, attn_heads, attn_points, min(attn_levels_max, 8),
                window_size, shift_size, temporal_channels=Ct,
                dt_bias_gain=dt_bias_gain, max_offset_scale=max_offset_scale
            )
        
        self.fuse1 = make_attention(C)
        self.fuse2 = make_attention(C * 2)
        self.fuse3 = make_attention(C * 4)
        self.fuse4 = make_attention(C * 8)
        
        # Temporal weighter
        self.weighter = TemporalWeighter(Ct, use_bimodal=True)
        
        # NAFNet Decoder (+1 for speed token)
        self.decoder = NAFNetDecoder(
            base_channels=C,
            temporal_channels=Ct + 1,
            depths=decoder_depths,
            drop_path_rate=decoder_drop_path
        )
        
        # Cut detector (scene change robustness)
        self.cut_detector = CutDetector(thresh=cut_thresh)
    
    def _build_speed_token(self, anchor_times, target_time):
        """Build speed token describing temporal spacing."""
        B, N = anchor_times.shape
        at_sorted, _ = anchor_times.sort(dim=1)
        gaps = at_sorted[:, 1:] - at_sorted[:, :-1] if N > 1 else torch.zeros(B, 0, device=anchor_times.device)
        span = (anchor_times.max(dim=1, keepdim=True).values -
                anchor_times.min(dim=1, keepdim=True).values).clamp(min=1e-6)
        if gaps.numel() > 0:
            med_gap = gaps.median(dim=1).values.view(B, 1)
        else:
            med_gap = torch.zeros(B, 1, device=anchor_times.device)
        speed = (med_gap / span).clamp(0, 1)
        return speed
    
    def forward(self, frames, anchor_times, target_time):
        """
        Args:
            frames: [B, N, 3, H, W] - N anchor frames
            anchor_times: [B, N] - timestamps of anchor frames
            target_time: [B] or [B, 1] - target timestamp to synthesize
        
        Returns:
            out: [B, 3, H, W] - synthesized frame
            aux: dict - auxiliary outputs for loss computation
        """
        B, N, _, H, W = frames.shape
        device = frames.device
        
        if target_time.dim() == 1:
            target_time = target_time.unsqueeze(1)
        
        # ===== Temporal normalization =====
        span = (anchor_times.max(dim=1, keepdim=True).values -
                anchor_times.min(dim=1, keepdim=True).values).clamp(min=1e-6)
        rel_norm = (anchor_times - target_time) / span
        rel_norm = rel_norm.clamp(-2.0, 2.0)
        
        # Temporal encodings and weights
        rel_enc = self.temporal_encoder(rel_norm)  # [B, N, Ct]
        weights, prior_info = self.weighter(rel_enc, rel_norm, anchor_times, target_time)  # [B, N]
        
        # ===== Encode all views =====
        feats1, feats2, feats3, feats4 = [], [], [], []
        low_feats = []
        
        for i in range(N):
            f1, f2, f3, f4 = self.encoder(frames[:, i], rel_enc[:, i])
            feats1.append(f1)
            feats2.append(f2)
            feats3.append(f3)
            feats4.append(f4)
            low_feats.append(f4)
        
        # ===== Scene cut detection (vectorized) =====
        lf = torch.stack(low_feats, dim=1)  # [B, N, C*8, H/8, W/8]
        nearest_idx = (anchor_times - target_time).abs().argmin(dim=1)  # [B]
        dir_right = (anchor_times.gather(1, nearest_idx.unsqueeze(1)) <= target_time).squeeze(1)
        alt_idx = nearest_idx + torch.where(dir_right, torch.tensor(1, device=device), torch.tensor(-1, device=device))
        alt_idx = alt_idx.clamp(0, N - 1)
        
        batch_ids = torch.arange(B, device=device)
        f_near = lf[batch_ids, nearest_idx]
        f_alt = lf[batch_ids, alt_idx]
        cut_mask, cut_score = self.cut_detector(f_near, f_alt)
        use_fallback = cut_mask > 0.5
        
        # ===== Fusion per scale =====
        s1 = torch.stack(feats1, dim=1)  # [B, N, C, H, W]
        s2 = torch.stack(feats2, dim=1)
        s3 = torch.stack(feats3, dim=1)
        s4 = torch.stack(feats4, dim=1)
        
        # Weighted query for attention
        wv = weights.view(B, N, 1, 1, 1)
        q1 = (s1 * wv).sum(dim=1)
        q2 = (s2 * wv).sum(dim=1)
        q3 = (s3 * wv).sum(dim=1)
        q4 = (s4 * wv).sum(dim=1)
        
        # Target time encoding (Δt = 0)
        tgt_zero_enc = self.temporal_encoder(torch.zeros(B, 1, device=device))[:, 0]
        
        # Attention fusion
        fused1, conf1, ent1 = self.fuse1(q1, s1, rel_norm, tgt_zero_enc)
        fused2, conf2, ent2 = self.fuse2(q2, s2, rel_norm, tgt_zero_enc)
        fused3, conf3, ent3 = self.fuse3(q3, s3, rel_norm, tgt_zero_enc)
        fused4, conf4, ent4 = self.fuse4(q4, s4, rel_norm, tgt_zero_enc)
        
        # Scene cut fallback
        if use_fallback.any():
            idx = torch.nonzero(use_fallback).squeeze(1)
            fused1[idx] = s1[idx, nearest_idx[idx]]
            fused2[idx] = s2[idx, nearest_idx[idx]]
            fused3[idx] = s3[idx, nearest_idx[idx]]
            fused4[idx] = s4[idx, nearest_idx[idx]]
        
        # ===== Build decoder conditioning =====
        speed = self._build_speed_token(anchor_times, target_time)  # [B, 1]
        tenc_with_speed = torch.cat([tgt_zero_enc, speed], dim=-1)  # [B, Ct+1]
        
        # ===== Decode =====
        out = self.decoder(fused1, fused2, fused3, fused4, tenc_with_speed)
        
        # ===== Auxiliary outputs =====
        def _to_full(x, H, W):
            return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        conf = torch.clamp_min(torch.stack([
            _to_full(conf1, H, W),
            _to_full(conf2, H, W),
            _to_full(conf3, H, W),
            _to_full(conf4, H, W)
        ], dim=0).mean(dim=0), 1e-4)
        
        entropy = torch.stack([
            _to_full(ent1, H, W),
            _to_full(ent2, H, W),
            _to_full(ent3, H, W),
            _to_full(ent4, H, W)
        ], dim=0).mean(dim=0)
        
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


# ==============================================================================
# Factory
# ==============================================================================

def build_tempo_v3(
    base_channels: int = 64,
    temporal_channels: int = 64,
    encoder_depths: list = None,
    decoder_depths: list = None,
    attn_heads: int = 4,
    attn_points: int = 4,
    attn_levels_max: int = 6,
    window_size: int = 8,
    shift_size: int = 0,
    dt_bias_gain: float = 1.0,
    max_offset_scale: float = 1.5,
    cut_thresh: float = 0.35
):
    """
    Build TEMPO v3 model.
    
    Presets:
        Quality (default):
            encoder_depths=[3, 3, 9, 3], decoder_depths=[2, 2, 2, 2]
        Balanced:
            encoder_depths=[2, 2, 6, 2], decoder_depths=[2, 2, 2, 2]
        Fast:
            encoder_depths=[2, 2, 2, 2], decoder_depths=[1, 1, 2, 2]
    """
    return TEMPOv3(
        base_channels=base_channels,
        temporal_channels=temporal_channels,
        encoder_depths=encoder_depths,
        decoder_depths=decoder_depths,
        attn_heads=attn_heads,
        attn_points=attn_points,
        attn_levels_max=attn_levels_max,
        window_size=window_size,
        shift_size=shift_size,
        dt_bias_gain=dt_bias_gain,
        max_offset_scale=max_offset_scale,
        cut_thresh=cut_thresh
    )


# ==============================================================================
# Quick test
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_tempo_v3(base_channels=64, temporal_channels=64).to(device)
    
    # Test input
    B, N, H, W = 2, 3, 256, 256
    frames = torch.randn(B, N, 3, H, W).to(device)
    anchor_times = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.3, 1.0]]).to(device)
    target_time = torch.tensor([0.25, 0.65]).to(device)
    
    # Forward
    out, aux = model(frames, anchor_times, target_time)
    
    print(f"Input: {frames.shape}")
    print(f"Output: {out.shape}")
    print(f"Weights: {aux['weights']}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nTotal parameters: {total_params:.2f}M")
