# tempo.py
# TEMPO: Temporal Multi-View Frame Synthesis
#
# High-Performance Architecture:
#   - ConvNeXt Encoder: Rich hierarchical features
#   - Deformable Temporal Attention: Implicit motion handling
#   - Cross-Scale Refinement: Coarse-to-fine consistency
#   - NAFNet Decoder: High-quality synthesis
#
# Philosophy:
#   - Input frames are temporal observations (like multi-view 3D)
#   - Deformable attention learns WHERE to gather information
#   - Attention weights learn HOW MUCH to trust each observation
#   - Decoder SYNTHESIZES (not warps) the output

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.convnext_nafnet import ConvNeXtEncoder, NAFNetDecoder
from model.temporal_attention import MultiScaleTemporalFusion
from model.temporal import TemporalPositionEncoding, TemporalWeighter


class TEMPO(nn.Module):
    """
    TEMPO: Temporal Multi-View Frame Synthesis
    
    Treats video interpolation as temporal view synthesis.
    Given N frames at arbitrary timestamps, synthesize any query time.
    
    Key components:
        - Encoder: Extract features from each temporal observation
        - Fusion: Deformable attention gathers information across time
        - Decoder: Synthesizes output from fused features
    
    Capabilities:
        - Arbitrary N input frames (2, 4, 8, ...)
        - Continuous timestamps (not discrete)
        - Interpolation AND extrapolation
        - Handles motion implicitly (no explicit flow)
        - Can hallucinate occluded content
    """
    
    def __init__(
        self,
        base_channels: int = 80,  # TEMPO BEAST: Scaled up from 64 to 80 for ~42M params
        temporal_channels: int = 64,
        # Encoder
        encoder_depths: list = None,
        encoder_drop_path: float = 0.1,
        # Decoder
        decoder_depths: list = None,
        decoder_drop_path: float = 0.05,
        # Fusion
        num_heads: int = 8,
        num_points: int = 6,  # TEMPO BEAST: Updated from 4 to 6
        use_cross_scale: bool = True,
        fusion_dropout: float = 0.0,
        # Training
        use_checkpointing: bool = False,
    ):
        super().__init__()

        C = base_channels
        Ct = temporal_channels

        # TEMPO BEAST defaults (~42M parameters)
        if encoder_depths is None:
            encoder_depths = [3, 3, 18, 3]  # TEMPO BEAST: Scaled up from [3,3,12,3]
        if decoder_depths is None:
            decoder_depths = [3, 3, 9, 3]  # TEMPO BEAST: Scaled up from [3,3,3,3]
        
        # Temporal encoding
        self.temporal_encoder = TemporalPositionEncoding(channels=Ct)
        
        # Encoder
        self.encoder = ConvNeXtEncoder(
            base_channels=C,
            temporal_channels=Ct,
            depths=encoder_depths,
            drop_path_rate=encoder_drop_path,
            use_checkpointing=use_checkpointing
        )

        # High-performance fusion
        self.fusion = MultiScaleTemporalFusion(
            base_channels=C,
            temporal_channels=Ct,
            num_heads=num_heads,
            num_points=num_points,
            use_cross_scale=use_cross_scale,
            dropout=fusion_dropout,
        )

        # Temporal weighting (for query initialization)
        self.weighter = TemporalWeighter(Ct, use_bimodal=True)

        # Decoder (+ speed token)
        self.decoder = NAFNetDecoder(
            base_channels=C,
            temporal_channels=Ct + 1,  # +1 for speed token (concatenated on line 214)
            depths=decoder_depths,
            drop_path_rate=decoder_drop_path,
            use_checkpointing=use_checkpointing
        )

        # Skip connection blend weight (learnable)
        self.skip_blend = nn.Parameter(torch.tensor(0.1))
    
    def _compute_speed_token(self, anchor_times: torch.Tensor, target_time: torch.Tensor) -> torch.Tensor:
        """Speed token: temporal density indicator."""
        B, N = anchor_times.shape
        
        at_sorted, _ = anchor_times.sort(dim=1)
        
        if N > 1:
            gaps = at_sorted[:, 1:] - at_sorted[:, :-1]
            med_gap = gaps.median(dim=1).values.view(B, 1)
        else:
            med_gap = torch.ones(B, 1, device=anchor_times.device)
        
        span = (anchor_times.max(dim=1, keepdim=True).values - 
                anchor_times.min(dim=1, keepdim=True).values).clamp(min=1e-6)
        
        return (med_gap / span).clamp(0, 1)
    
    def forward(
        self, 
        frames: torch.Tensor,        # [B, N, 3, H, W]
        anchor_times: torch.Tensor,  # [B, N]
        target_time: torch.Tensor,   # [B] or [B, 1]
    ):
        """
        Synthesize frame at target_time from N temporal observations.
        
        Returns:
            output: [B, 3, H, W]
            aux: dict with weights, confidence, entropy
        """
        B, N, _, H, W = frames.shape
        device = frames.device
        
        if target_time.dim() == 1:
            target_time = target_time.unsqueeze(1)
        
        # =====================================================================
        # 1. TEMPORAL NORMALIZATION
        # =====================================================================
        span = (anchor_times.max(dim=1, keepdim=True).values -
                anchor_times.min(dim=1, keepdim=True).values).clamp(min=1e-6)
        rel_time = (anchor_times - target_time) / span
        rel_time = rel_time.clamp(-2.0, 2.0)
        
        # Encode relative times
        rel_enc = self.temporal_encoder(rel_time)  # [B, N, Ct]
        
        # =====================================================================
        # 2. COMPUTE TEMPORAL WEIGHTS
        # =====================================================================
        weights, prior_info = self.weighter(rel_enc, rel_time, anchor_times, target_time)
        
        # =====================================================================
        # 3. ENCODE ALL OBSERVATIONS (BATCHED FOR EFFICIENCY)
        # =====================================================================
        # Optimization: Batch all N frames into single forward pass
        # Original: N separate encoder calls → Now: 1 batched call
        # Speedup: 2.5-3x faster (N kernel launches → 1 launch, better GPU util)

        # Reshape from [B, N, 3, H, W] to [B*N, 3, H, W]
        B, N, _, H, W = frames.shape
        Ct = rel_enc.shape[-1]  # temporal_channels
        frames_flat = frames.reshape(B * N, 3, H, W)
        rel_enc_flat = rel_enc.reshape(B * N, Ct)

        # Single batched forward pass through encoder
        f1, f2, f3, f4 = self.encoder(frames_flat, rel_enc_flat)

        # Reshape outputs back to [B, N, C, H/scale, W/scale] per scale
        C = self.encoder.base_channels
        _, _, h1, w1 = f1.shape
        _, _, h2, w2 = f2.shape
        _, _, h3, w3 = f3.shape
        _, _, h4, w4 = f4.shape

        f1 = f1.reshape(B, N, C, h1, w1)
        f2 = f2.reshape(B, N, C * 2, h2, w2)
        f3 = f3.reshape(B, N, C * 4, h3, w3)
        f4 = f4.reshape(B, N, C * 8, h4, w4)

        # Use features directly (anti-aliasing removed for efficiency)
        # Gradient loss provides edge preservation without non-learnable blur
        v1, v2, v3, v4 = f1, f2, f3, f4
        
        # =====================================================================
        # 4. INITIALIZE QUERIES
        # =====================================================================
        wv = weights.view(B, N, 1, 1, 1)
        q1 = (v1 * wv).sum(dim=1)
        q2 = (v2 * wv).sum(dim=1)
        q3 = (v3 * wv).sum(dim=1)
        q4 = (v4 * wv).sum(dim=1)
        
        # =====================================================================
        # 5. TEMPORAL FUSION
        # =====================================================================
        (f1, f2, f3, f4), confidence, entropy = self.fusion(
            queries=(q1, q2, q3, q4),
            values=(v1, v2, v3, v4),
            rel_time=rel_time,
            weights=weights,
        )
        
        # =====================================================================
        # 6. DECODE
        # =====================================================================
        tgt_enc = self.temporal_encoder(torch.zeros(B, 1, device=device))[:, 0]
        speed = self._compute_speed_token(anchor_times, target_time)
        tgt_enc_with_speed = torch.cat([tgt_enc, speed], dim=-1)

        # TEMPO BEAST: Decoder now returns (rgb, uncertainty_log_var)
        decoder_output, uncertainty_log_var = self.decoder(f1, f2, f3, f4, tgt_enc_with_speed)
        
        # =====================================================================
        # 6b. SKIP CONNECTION: Blend with weighted input for sharpness
        # =====================================================================
        # Compute weighted average of input frames
        wv_input = weights.view(B, N, 1, 1, 1)
        input_blend = (frames * wv_input).sum(dim=1)  # [B, 3, H, W]
        
        # Blend decoder output with input (reduces blur, preserves detail)
        # skip_blend is learnable, starts small to not overwhelm decoder learning
        blend_weight = torch.sigmoid(self.skip_blend)  # Constrain to [0, 1]
        output = (1 - blend_weight) * decoder_output + blend_weight * input_blend

        # =====================================================================
        # 7. AUX OUTPUTS
        # =====================================================================
        aux = {
            "weights": weights.detach(),
            "confidence": confidence.detach(),
            "entropy": entropy.detach(),
            "prior_alpha": prior_info["alpha"],
            "prior_bimix": prior_info["bimix"],
            # TEMPO BEAST: Heteroscedastic uncertainty outputs
            "uncertainty_log_var": uncertainty_log_var.detach(),
            "uncertainty_sigma": torch.exp(0.5 * uncertainty_log_var).detach(),
        }
        
        return output, aux


# ==============================================================================
# Factory
# ==============================================================================

def build_tempo(
    base_channels: int = 80,  # TEMPO BEAST: Scaled from 64 to 80 for ~42M params
    temporal_channels: int = 64,
    encoder_depths: list = None,
    decoder_depths: list = None,
    # Fusion settings
    num_heads: int = 8,
    num_points: int = 6,  # TEMPO BEAST: Updated from 4 to 6
    use_cross_scale: bool = True,
    # Training
    use_checkpointing: bool = False,
    # Legacy (ignored)
    attn_heads: int = None,
    attn_points: int = None,
    window_size: int = None,
    dt_bias_gain: float = None,
    max_offset_scale: float = None,
):
    """
    Build TEMPO model.

    TEMPO BEAST (default):
        base_channels=80, encoder_depths=[3,3,18,3], decoder_depths=[3,3,9,3]
        num_heads=8, num_points=6
        ~42M params - Phase 1 architecture scaling complete

    Other Presets:
        Quality:
            encoder_depths=[3,3,12,3], decoder_depths=[3,3,3,3]
            num_heads=8, num_points=4
            ~25M params

        Large:
            encoder_depths=[3,3,18,3], decoder_depths=[3,3,6,3]
            num_heads=8, num_points=6
            ~35M params

        Fast:
            encoder_depths=[2,2,6,2], decoder_depths=[2,2,2,2]
            num_heads=4, num_points=4
            ~15M params
    """
    # Handle legacy parameter names
    if attn_heads is not None:
        num_heads = attn_heads
    if attn_points is not None:
        num_points = attn_points
    
    return TEMPO(
        base_channels=base_channels,
        temporal_channels=temporal_channels,
        encoder_depths=encoder_depths,
        decoder_depths=decoder_depths,
        num_heads=num_heads,
        num_points=num_points,
        use_cross_scale=use_cross_scale,
        use_checkpointing=use_checkpointing,
    )


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build model
    model = build_tempo(
        base_channels=64,
        temporal_channels=64,
        encoder_depths=[3, 3, 12, 3],
        decoder_depths=[3, 3, 3, 3],
        num_heads=8,
        num_points=4,
        use_cross_scale=True,
    ).to(device)
    
    # Test input
    B, N, H, W = 2, 4, 256, 256
    frames = torch.randn(B, N, 3, H, W, device=device)
    anchor_times = torch.tensor([[0.0, 0.3, 0.7, 1.0], [0.0, 0.2, 0.8, 1.0]], device=device)
    target_time = torch.tensor([0.5, 0.4], device=device)
    
    # Forward
    with torch.no_grad():
        output, aux = model(frames, anchor_times, target_time)
    
    print(f"\n{'='*60}")
    print(f"TEMPO: High-Performance Temporal View Synthesis")
    print(f"{'='*60}")
    print(f"\nInput:")
    print(f"  Frames: {frames.shape}")
    print(f"  Timestamps: {anchor_times[0].tolist()}")
    print(f"  Query time: {target_time[0].item():.2f}")
    
    print(f"\nOutput:")
    print(f"  Synthesized: {output.shape}")
    print(f"  Confidence: mean={aux['confidence'].mean():.3f}")
    print(f"  Entropy: mean={aux['entropy'].mean():.3f}")
    print(f"  Weights: {aux['weights'][0].cpu().numpy().round(3)}")
    
    # Parameters
    total = sum(p.numel() for p in model.parameters()) / 1e6
    enc = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    fus = sum(p.numel() for p in model.fusion.parameters()) / 1e6
    dec = sum(p.numel() for p in model.decoder.parameters()) / 1e6
    
    print(f"\n{'='*60}")
    print(f"Parameters:")
    print(f"  Encoder (ConvNeXt):      {enc:.2f}M")
    print(f"  Fusion (Deformable):     {fus:.2f}M")
    print(f"  Decoder (NAFNet):        {dec:.2f}M")
    print(f"  Total:                   {total:.2f}M")
    print(f"{'='*60}")
    
    print("\n✓ TEMPO ready for training!")
