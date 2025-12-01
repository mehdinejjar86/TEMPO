# tempo.py
# TEMPO: Temporal Multi-View Frame Synthesis with Time-Aware Pyramid Attention
# ConvNeXt Encoder + NAFNet Decoder with AdaLN-Zero conditioning

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.convnext_nafnet import ConvNeXtEncoder, NAFNetDecoder
from model.pyramid import TimeAwareSlidingWindowPyramidAttention
from model.temporal import TemporalPositionEncoding, TemporalWeighter
from model.utility import CutDetector


class TEMPO(nn.Module):
    """
    TEMPO: Temporal Multi-View Frame Synthesis
    
    Architecture:
        - ConvNeXt Encoder: Rich hierarchical features with AdaLN-Zero
        - TimeAware Pyramid Attention: Multi-view temporal fusion
        - NAFNet Decoder: Maximum reconstruction quality (PSNR/SSIM)
    
    Key features:
        - ConvNeXt blocks with 7×7 depthwise convs (large receptive field)
        - AdaLN-Zero conditioning (stable, precise temporal modulation)
        - NAFNet decoder with SimpleGate (minimal information loss)
        - Layer scale + stochastic depth (better training dynamics)
    """
    
    def __init__(
        self,
        base_channels: int = 64,
        temporal_channels: int = 64,
        # Encoder settings (ConvNeXt)
        encoder_depths: list = None,
        encoder_drop_path: float = 0.1,
        # Decoder settings (NAFNet)
        decoder_depths: list = None,
        decoder_drop_path: float = 0.05,
        # Attention fusion settings
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
        
        # =================================================================
        # OPTIMAL DEFAULTS (tuned for maximum PSNR/SSIM)
        # =================================================================
        # Encoder: Deep feature extraction with rich representations
        #   - Stage 1 (1×):   3 blocks  - initial feature extraction
        #   - Stage 2 (1/2×): 3 blocks  - local patterns
        #   - Stage 3 (1/4×): 12 blocks - main processing (deepest)
        #   - Stage 4 (1/8×): 3 blocks  - high-level semantics
        #   Total: 21 encoder blocks
        #
        # Decoder: Balanced reconstruction with skip connections
        #   - Each stage: 3 blocks for thorough refinement
        #   Total: 12 decoder blocks
        # =================================================================
        
        if encoder_depths is None:
            encoder_depths = [3, 3, 12, 3]  # Optimal for quality
        if decoder_depths is None:
            decoder_depths = [3, 3, 3, 3]   # Balanced refinement
        
        # Temporal encoding
        self.temporal_encoder = TemporalPositionEncoding(channels=Ct)
        
        # ConvNeXt Encoder (rich feature extraction)
        self.encoder = ConvNeXtEncoder(
            base_channels=C,
            temporal_channels=Ct,
            depths=encoder_depths,
            drop_path_rate=encoder_drop_path
        )
        
        # Time-aware pyramid attention fusion (per scale)
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
        
        # Temporal weighter (bimodal prior for interpolation)
        self.weighter = TemporalWeighter(Ct, use_bimodal=True)
        
        # NAFNet Decoder (+1 for speed token)
        self.decoder = NAFNetDecoder(
            base_channels=C,
            temporal_channels=Ct + 1,
            depths=decoder_depths,
            drop_path_rate=decoder_drop_path
        )
        
        # Scene cut detector (robustness to scene changes)
        self.cut_detector = CutDetector(thresh=cut_thresh)
    
    def _build_speed_token(self, anchor_times, target_time):
        """
        Build speed token describing temporal spacing.
        Helps decoder understand motion magnitude.
        """
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
        Synthesize a frame at target_time given N anchor frames.
        
        Args:
            frames: [B, N, 3, H, W] - N anchor frames at different timestamps
            anchor_times: [B, N] - timestamps of anchor frames (arbitrary spacing)
            target_time: [B] or [B, 1] - target timestamp to synthesize
        
        Returns:
            out: [B, 3, H, W] - synthesized frame at target_time
            aux: dict - auxiliary outputs for loss computation and debugging
                - weights: [B, N] temporal blending weights
                - prior_alpha, prior_bimix: weighter internals
                - conf_map: [B, 1, H, W] confidence map
                - attn_entropy: [B, 1, H, W] attention entropy
                - cut_score: [B] scene cut scores
                - fallback_mask: [B] which samples used fallback
        """
        B, N, _, H, W = frames.shape
        device = frames.device
        
        if target_time.dim() == 1:
            target_time = target_time.unsqueeze(1)
        
        # =====================================================================
        # 1. TEMPORAL NORMALIZATION
        # =====================================================================
        # Normalize time differences relative to span for scale invariance
        span = (anchor_times.max(dim=1, keepdim=True).values -
                anchor_times.min(dim=1, keepdim=True).values).clamp(min=1e-6)
        rel_norm = (anchor_times - target_time) / span
        rel_norm = rel_norm.clamp(-2.0, 2.0)
        
        # Encode temporal positions
        rel_enc = self.temporal_encoder(rel_norm)  # [B, N, Ct]
        
        # Compute temporal blending weights
        weights, prior_info = self.weighter(rel_enc, rel_norm, anchor_times, target_time)  # [B, N]
        
        # =====================================================================
        # 2. ENCODE ALL VIEWS (ConvNeXt)
        # =====================================================================
        feats1, feats2, feats3, feats4 = [], [], [], []
        low_feats = []
        
        for i in range(N):
            f1, f2, f3, f4 = self.encoder(frames[:, i], rel_enc[:, i])
            feats1.append(f1)
            feats2.append(f2)
            feats3.append(f3)
            feats4.append(f4)
            low_feats.append(f4)
        
        # =====================================================================
        # 3. SCENE CUT DETECTION (vectorized)
        # =====================================================================
        # Detect if there's a scene cut between nearest frames to target
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
        
        # =====================================================================
        # 4. MULTI-SCALE FUSION (Time-Aware Pyramid Attention)
        # =====================================================================
        s1 = torch.stack(feats1, dim=1)  # [B, N, C, H, W]
        s2 = torch.stack(feats2, dim=1)
        s3 = torch.stack(feats3, dim=1)
        s4 = torch.stack(feats4, dim=1)
        
        # Weighted query for attention (soft blend as initialization)
        wv = weights.view(B, N, 1, 1, 1)
        q1 = (s1 * wv).sum(dim=1)
        q2 = (s2 * wv).sum(dim=1)
        q3 = (s3 * wv).sum(dim=1)
        q4 = (s4 * wv).sum(dim=1)
        
        # Target time encoding (Δt = 0 for the frame we're synthesizing)
        tgt_zero_enc = self.temporal_encoder(torch.zeros(B, 1, device=device))[:, 0]
        
        # Attention fusion at each scale
        fused1, conf1, ent1 = self.fuse1(q1, s1, rel_norm, tgt_zero_enc)
        fused2, conf2, ent2 = self.fuse2(q2, s2, rel_norm, tgt_zero_enc)
        fused3, conf3, ent3 = self.fuse3(q3, s3, rel_norm, tgt_zero_enc)
        fused4, conf4, ent4 = self.fuse4(q4, s4, rel_norm, tgt_zero_enc)
        
        # Scene cut fallback: use nearest frame features directly
        if use_fallback.any():
            idx = torch.nonzero(use_fallback).squeeze(1)
            fused1[idx] = s1[idx, nearest_idx[idx]]
            fused2[idx] = s2[idx, nearest_idx[idx]]
            fused3[idx] = s3[idx, nearest_idx[idx]]
            fused4[idx] = s4[idx, nearest_idx[idx]]
        
        # =====================================================================
        # 5. DECODE (NAFNet)
        # =====================================================================
        # Build temporal conditioning with speed token
        speed = self._build_speed_token(anchor_times, target_time)  # [B, 1]
        tenc_with_speed = torch.cat([tgt_zero_enc, speed], dim=-1)  # [B, Ct+1]
        
        # Decode fused features to output frame
        out = self.decoder(fused1, fused2, fused3, fused4, tenc_with_speed)
        
        # =====================================================================
        # 6. AUXILIARY OUTPUTS
        # =====================================================================
        def _to_full(x, H, W):
            return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        # Aggregate confidence and entropy maps
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

def build_tempo(
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
    Build TEMPO model.
    
    Presets (encoder_depths, decoder_depths):
        Quality (default): [3, 3, 12, 3], [3, 3, 3, 3]  (~45M params)
        Balanced:          [3, 3, 9, 3],  [2, 2, 2, 2]  (~35M params)
        Fast:              [2, 2, 6, 2],  [2, 2, 2, 2]  (~25M params)
        Lite:              [2, 2, 2, 2],  [1, 1, 2, 2]  (~15M params)
    
    Args:
        base_channels: Base channel count (scales as C, 2C, 4C, 8C)
        temporal_channels: Dimension of temporal embeddings
        encoder_depths: Number of ConvNeXt blocks per encoder stage
        decoder_depths: Number of NAFNet blocks per decoder stage
        attn_heads: Number of attention heads in fusion
        attn_points: Number of sampling points per head
        attn_levels_max: Maximum pyramid levels
        window_size: Window size for windowed attention
        shift_size: Shift amount for shifted windows
        dt_bias_gain: Temporal bias strength in attention
        max_offset_scale: Maximum deformable offset scale
        cut_thresh: Threshold for scene cut detection
    
    Returns:
        TEMPO model instance
    """
    return TEMPO(
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
# Test
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build model with optimal defaults
    model = build_tempo(base_channels=64, temporal_channels=64).to(device)
    
    # Test input
    B, N, H, W = 2, 3, 256, 256
    frames = torch.randn(B, N, 3, H, W).to(device)
    anchor_times = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.3, 1.0]]).to(device)
    target_time = torch.tensor([0.25, 0.65]).to(device)
    
    # Forward pass
    with torch.no_grad():
        out, aux = model(frames, anchor_times, target_time)
    
    print(f"\nInput frames: {frames.shape}")
    print(f"Anchor times: {anchor_times}")
    print(f"Target times: {target_time}")
    print(f"\nOutput: {out.shape}")
    print(f"Temporal weights: {aux['weights']}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*50}")
    print(f"Total parameters:     {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"{'='*50}")
    
    # Per-component breakdown
    enc_params = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    dec_params = sum(p.numel() for p in model.decoder.parameters()) / 1e6
    fuse_params = sum(
        sum(p.numel() for p in fuse.parameters())
        for fuse in [model.fuse1, model.fuse2, model.fuse3, model.fuse4]
    ) / 1e6
    
    print(f"Encoder (ConvNeXt):  {enc_params:.2f}M")
    print(f"Decoder (NAFNet):    {dec_params:.2f}M")
    print(f"Fusion (Attention):  {fuse_params:.2f}M")
    print(f"{'='*50}")
    
    print("\n✓ TEMPO model ready!")
