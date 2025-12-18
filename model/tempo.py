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
            drop_path_rate=encoder_drop_path
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
            temporal_channels=Ct + 1,
            depths=decoder_depths,
            drop_path_rate=decoder_drop_path
        )
    
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
    
    def compute_backward_synthesis(
        self,
        forward_pred: torch.Tensor,      # [B, 3, H, W] - forward synthesized frame
        frames: torch.Tensor,             # [B, N, 3, H, W] - original anchor frames
        anchor_times: torch.Tensor,       # [B, N] - anchor timestamps
        target_time: torch.Tensor,        # [B] or [B, 1] - target timestamp
        anchor_idx: int = 0,              # Which anchor to reconstruct
    ):
        """
        TEMPO BEAST Phase 3: Backward synthesis for bidirectional consistency.

        Given the forward prediction, synthesize one of the original anchor frames.
        This creates a cycle: [F0, F1, ...] → Ft → [Ft, ...] → F0'

        Args:
            forward_pred: Forward synthesized frame at target_time
            frames: Original anchor frames
            anchor_times: Timestamps of anchor frames
            target_time: Target timestamp
            anchor_idx: Index of anchor frame to reconstruct (default: 0)

        Returns:
            reconstructed_anchor: [B, 3, H, W] - backward synthesized anchor frame
        """
        B, N = anchor_times.shape

        # Create new observation set: [Ft, F1, F2, ..., FN] (replace F_anchor_idx with Ft)
        backward_frames = frames.clone()
        backward_frames[:, anchor_idx] = forward_pred

        # New timestamps: target_time replaces anchor_times[anchor_idx]
        backward_times = anchor_times.clone()
        backward_times[:, anchor_idx] = target_time.squeeze(-1) if target_time.dim() > 1 else target_time

        # Target: reconstruct the original anchor
        reconstruct_time = anchor_times[:, anchor_idx]

        # Forward pass to reconstruct anchor (use no_grad to avoid memory overhead)
        with torch.no_grad():
            reconstructed, _ = self.forward(backward_frames, backward_times, reconstruct_time)

        return reconstructed

    def forward(
        self,
        frames: torch.Tensor,        # [B, N, 3, H, W]
        anchor_times: torch.Tensor,  # [B, N]
        target_time: torch.Tensor,   # [B] or [B, 1]
        compute_bidirectional: bool = False,  # TEMPO BEAST Phase 3
    ):
        """
        Synthesize frame at target_time from N temporal observations.

        Args:
            frames: Input anchor frames [B, N, 3, H, W]
            anchor_times: Anchor timestamps [B, N]
            target_time: Target timestamp [B] or [B, 1]
            compute_bidirectional: If True, compute backward synthesis for consistency loss

        Returns:
            output: [B, 3, H, W]
            aux: dict with weights, confidence, entropy, and optionally backward_anchor
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
        # 3. ENCODE ALL OBSERVATIONS
        # =====================================================================
        feats1, feats2, feats3, feats4 = [], [], [], []
        
        for i in range(N):
            f1, f2, f3, f4 = self.encoder(frames[:, i], rel_enc[:, i])
            feats1.append(f1)
            feats2.append(f2)
            feats3.append(f3)
            feats4.append(f4)
        
        # Stack: [B, N, C, H, W] per scale
        v1 = torch.stack(feats1, dim=1)
        v2 = torch.stack(feats2, dim=1)
        v3 = torch.stack(feats3, dim=1)
        v4 = torch.stack(feats4, dim=1)
        
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
        output, uncertainty_log_var = self.decoder(f1, f2, f3, f4, tgt_enc_with_speed)

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

        # =====================================================================
        # 8. BIDIRECTIONAL CONSISTENCY (TEMPO BEAST Phase 3)
        # =====================================================================
        if compute_bidirectional and self.training and N >= 2:
            # Choose anchor to reconstruct (nearest to target for stability)
            rel_time_abs = rel_time.abs()
            anchor_idx = rel_time_abs.argmin(dim=1)[0].item()  # Use same idx for whole batch

            # Compute backward synthesis
            backward_anchor = self.compute_backward_synthesis(
                output.detach(),  # Detach to avoid double backprop
                frames,
                anchor_times,
                target_time,
                anchor_idx=anchor_idx
            )

            aux["backward_anchor"] = backward_anchor
            aux["backward_anchor_idx"] = anchor_idx

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
