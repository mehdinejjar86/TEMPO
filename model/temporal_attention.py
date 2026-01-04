# temporal_attention.py
# High-Performance Temporal Cross-Attention for TEMPO
#
# Philosophy: 
#   - Input frames are temporal observations of a scene
#   - Deformable attention gathers relevant information (implicit motion)
#   - Time is a first-class citizen, deeply integrated
#   - Decoder synthesizes the output
#
# Key components:
#   - Deformable sampling: "Where to look in each observation"
#   - Temporal attention: "How much to trust each observation"
#   - Deep time integration: Time affects WHERE and HOW MUCH
#   - Multi-scale: Different motion magnitudes at different scales

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


# ==============================================================================
# Time Encoding
# ==============================================================================

class SinusoidalTimeEncoding(nn.Module):
    """
    Continuous time encoding with learnable projection.
    """
    def __init__(self, channels: int, max_period: float = 10000.0):
        super().__init__()
        self.channels = channels
        self.max_period = max_period
        self.proj = nn.Linear(channels, channels)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [...] timestamps (any shape)
        Returns:
            encoding: [..., channels]
        """
        half = self.channels // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        
        t_flat = t.reshape(-1, 1)
        args = t_flat * freqs
        enc = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        enc = enc.reshape(*t.shape, self.channels)
        
        return self.proj(enc)


# ==============================================================================
# Deformable Temporal Attention
# ==============================================================================

class DeformableTemporalAttention(nn.Module):
    """
    High-performance temporal attention with deformable sampling.
    
    For each query position:
      1. Predict WHERE to sample in each observation (deformable offsets)
      2. Predict HOW MUCH to weight each observation (attention)
      3. Both are conditioned on temporal distance
    
    This is "implicit motion" — the offsets learn to track content across time.
    """
    def __init__(
        self,
        channels: int,
        temporal_channels: int,
        num_heads: int = 8,
        num_points: int = 6,  # TEMPO BEAST: Increased from 4 to 6 (48 total samples)
        dropout: float = 0.0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        
        self.channels = channels
        self.temporal_channels = temporal_channels
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = channels // num_heads
        
        # =====================
        # Query pathway
        # =====================
        self.q_proj = nn.Conv2d(channels, channels, 1)
        
        # =====================
        # Offset prediction (WHERE to look)
        # Time-conditioned: offset depends on temporal distance
        # TEMPO BEAST: Added extra capacity for better motion estimation
        # =====================
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),  # Spatial context
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),  # Added: Extra intermediate layer
            nn.GELU(),
            nn.Conv2d(channels, channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(channels // 2, num_heads * num_points * 2, 1),
        )
        
        # Time modulates offsets (larger dt → larger search range)
        self.time_to_offset_scale = nn.Sequential(
            nn.Linear(temporal_channels, num_heads * num_points),
            nn.Softplus(),
        )
        
        # =====================
        # Attention prediction (HOW MUCH to weight)
        # TEMPO BEAST: Added spatial context layer for better attention
        # =====================
        self.attn_net = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels // 4, 3, padding=1),  # Added: Spatial context
            nn.GELU(),
            nn.Conv2d(channels // 4, num_heads * num_points, 1),
        )
        
        # Time bias for attention (closer observations → higher weight)
        self.time_to_attn_bias = nn.Linear(temporal_channels, num_heads)
        
        # =====================
        # Value pathway
        # =====================
        self.v_proj = nn.Conv2d(channels, channels, 1)
        
        # =====================
        # Output
        # =====================
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        
        # Gate for residual (learned, starts at 0)
        self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        # Offsets start at zero (identity sampling)
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)
        
        # Attention starts uniform
        nn.init.zeros_(self.attn_net[-1].weight)
        nn.init.zeros_(self.attn_net[-1].bias)
        
        # Output projection
        nn.init.xavier_uniform_(self.out_proj[0].weight)
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)
    
    def forward(
        self,
        query: torch.Tensor,          # [B, C, H, W] - target features
        values: torch.Tensor,         # [B, N, C, H, W] - observation features
        rel_time: torch.Tensor,       # [B, N] - relative time (normalized)
        time_enc: torch.Tensor,       # [B, N, Ct] - time encodings
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: [B, C, H, W] - fused features (residual-ready)
            confidence: [B, 1, H, W] - per-pixel confidence
            entropy: [B, 1, H, W] - attention entropy
        """
        B, C, H, W = query.shape
        N = values.shape[1]
        device = query.device
        dtype = query.dtype
        
        # Project query
        q = self.q_proj(query)  # [B, C, H, W]
        
        # Predict base offsets from query features
        base_offsets = self.offset_net(q)  # [B, heads*points*2, H, W]
        base_offsets = base_offsets.view(B, self.num_heads, self.num_points, 2, H, W)
        base_offsets = base_offsets.permute(0, 1, 2, 4, 5, 3)  # [B, heads, points, H, W, 2]
        
        # Predict base attention logits
        base_attn = self.attn_net(q)  # [B, heads*points, H, W]
        base_attn = base_attn.view(B, self.num_heads, self.num_points, H, W)
        base_attn = base_attn.permute(0, 1, 3, 4, 2)  # [B, heads, H, W, points]
        
        # Build sampling grid (normalized to [-1, 1])
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        
        # Process each observation
        all_samples = []  # Will be [B, N, heads, H, W, points, head_dim]
        all_attn_logits = []  # Will be [B, N, heads, H, W, points]
        
        for n in range(N):
            # Time encoding for this observation
            t_enc = time_enc[:, n]  # [B, Ct]
            t_rel = rel_time[:, n]  # [B]
            
            # Scale offsets by temporal distance (larger dt → look further)
            offset_scale = self.time_to_offset_scale(t_enc)  # [B, heads*points]
            offset_scale = offset_scale.view(B, self.num_heads, self.num_points, 1, 1, 1)
            offset_scale = 0.5 + offset_scale  # Base scale of 0.5, can grow
            
            # Scale offsets by temporal distance magnitude
            dt_magnitude = t_rel.abs().view(B, 1, 1, 1, 1, 1)
            scaled_offsets = base_offsets * offset_scale * (1.0 + dt_magnitude)
            
            # Clamp offsets to reasonable range
            scaled_offsets = scaled_offsets.clamp(-1.0, 1.0)
            
            # Attention bias from time (closer → higher attention)
            attn_bias = self.time_to_attn_bias(t_enc)  # [B, heads]
            attn_bias = attn_bias.view(B, self.num_heads, 1, 1, 1)
            
            # Distance penalty (fundamental, always prefer closer)
            dist_penalty = -t_rel.abs().view(B, 1, 1, 1, 1) * 2.0
            
            # Biased attention logits for this observation
            attn_n = base_attn + attn_bias + dist_penalty  # [B, heads, H, W, points]
            all_attn_logits.append(attn_n)
            
            # Project values for this observation
            v_n = self.v_proj(values[:, n])  # [B, C, H, W]
            v_n = v_n.view(B, self.num_heads, self.head_dim, H, W)
            
            # Sample at each point
            samples_n = []
            for p in range(self.num_points):
                # Sampling grid for this point
                offset_p = scaled_offsets[:, :, p]  # [B, heads, H, W, 2]
                
                # For each head, sample from values
                head_samples = []
                for h in range(self.num_heads):
                    grid_h = base_grid + offset_p[:, h]  # [B, H, W, 2]
                    grid_h = grid_h.clamp(-1, 1)
                    
                    # Sample
                    padding_mode = "zeros" if torch.backends.mps.is_available() else "border"
                    sampled = F.grid_sample(
                        v_n[:, h],  # [B, head_dim, H, W]
                        grid_h,
                        mode='bilinear',
                        padding_mode=padding_mode,
                        align_corners=True
                    )  # [B, head_dim, H, W]
                    head_samples.append(sampled)
                
                # Stack heads: [B, heads, head_dim, H, W]
                head_samples = torch.stack(head_samples, dim=1)
                samples_n.append(head_samples)
            
            # Stack points: [B, heads, points, head_dim, H, W]
            samples_n = torch.stack(samples_n, dim=2)
            samples_n = samples_n.permute(0, 1, 4, 5, 2, 3)  # [B, heads, H, W, points, head_dim]
            all_samples.append(samples_n)
        
        # Stack observations: [B, N, heads, H, W, points, head_dim]
        all_samples = torch.stack(all_samples, dim=1)
        
        # Stack attention: [B, N, heads, H, W, points]
        all_attn_logits = torch.stack(all_attn_logits, dim=1)
        
        # Softmax over (observations × points)
        attn_shape = all_attn_logits.shape
        all_attn_flat = all_attn_logits.view(B, N * self.num_heads, H, W, self.num_points)
        all_attn_flat = all_attn_flat.view(B, -1, H, W, self.num_points)
        
        # Reshape for softmax over N*points
        all_attn_logits = all_attn_logits.permute(0, 2, 3, 4, 1, 5)  # [B, heads, H, W, N, points]
        all_attn_logits = all_attn_logits.reshape(B, self.num_heads, H, W, N * self.num_points)
        attn_weights = F.softmax(all_attn_logits, dim=-1)  # [B, heads, H, W, N*points]
        attn_weights = self.dropout(attn_weights)
        
        # Compute entropy for confidence
        entropy = -(attn_weights * (attn_weights + 1e-8).log()).sum(dim=-1)  # [B, heads, H, W]
        entropy = entropy.mean(dim=1, keepdim=True).unsqueeze(1)  # [B, 1, 1, H, W] (intermediate)
        entropy = entropy.squeeze(2)  # [B, 1, H, W] (final shape)
        
        # Reshape attention weights back
        attn_weights = attn_weights.view(B, self.num_heads, H, W, N, self.num_points)
        attn_weights = attn_weights.permute(0, 4, 1, 2, 3, 5)  # [B, N, heads, H, W, points]
        
        # Weighted sum of samples
        # all_samples: [B, N, heads, H, W, points, head_dim]
        # attn_weights: [B, N, heads, H, W, points]
        output = (all_samples * attn_weights.unsqueeze(-1)).sum(dim=(1, 5))  # [B, heads, H, W, head_dim]
        output = output.permute(0, 1, 4, 2, 3)  # [B, heads, head_dim, H, W]
        output = output.reshape(B, C, H, W)
        
        # Output projection with gating
        output = self.out_proj(output) * torch.sigmoid(self.gate)
        
        # Confidence from entropy (low entropy = high confidence)
        max_entropy = math.log(N * self.num_points)
        confidence = 1.0 - (entropy / max_entropy).clamp(0, 1)
        
        return output, confidence, entropy


# ==============================================================================
# Temporal Fusion Block
# ==============================================================================

class TemporalFusionBlock(nn.Module):
    """
    Single-scale temporal fusion.
    
    Components:
        1. Deformable temporal attention
        2. Channel attention (squeeze-excite)
        3. FFN
    """
    def __init__(
        self,
        channels: int,
        temporal_channels: int,
        num_heads: int = 8,
        num_points: int = 4,
        ffn_expansion: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Temporal attention
        self.attn = DeformableTemporalAttention(
            channels=channels,
            temporal_channels=temporal_channels,
            num_heads=num_heads,
            num_points=num_points,
            dropout=dropout,
        )
        
        # Channel attention (squeeze-excite)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )
        
        # FFN
        hidden = int(channels * ffn_expansion)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),  # Depthwise
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        
        # Norms
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
        # FFN gate
        self.ffn_gate = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(
        self,
        query: torch.Tensor,
        values: torch.Tensor,
        rel_time: torch.Tensor,
        time_enc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Temporal attention with residual
        attn_out, conf, entropy = self.attn(self.norm1(query), values, rel_time, time_enc)
        x = query + attn_out
        
        # Channel attention
        ca = self.channel_attn(x)
        x = x * ca.view(x.shape[0], -1, 1, 1)
        
        # FFN with gated residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out * torch.sigmoid(self.ffn_gate)
        
        return x, conf, entropy


# ==============================================================================
# Multi-Scale Temporal Fusion
# ==============================================================================

class MultiScaleTemporalFusion(nn.Module):
    """
    Multi-scale temporal fusion with optional cross-scale refinement.
    
    Scales:
        - Scale 1: Full resolution (fine details, small motion)
        - Scale 2: 1/2× (medium details, medium motion)
        - Scale 3: 1/4× (coarse structure, large motion)
        - Scale 4: 1/8× (global context, very large motion)
    
    Cross-scale: Coarse predictions guide fine predictions.
    """
    def __init__(
        self,
        base_channels: int = 64,
        temporal_channels: int = 64,
        num_heads: int = 8,
        num_points: int = 4,
        use_cross_scale: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        C = base_channels
        Ct = temporal_channels
        
        self.use_cross_scale = use_cross_scale
        
        # Per-scale fusion (more heads at coarser scales)
        self.fuse4 = TemporalFusionBlock(C * 8, Ct, num_heads=num_heads, num_points=num_points, dropout=dropout)
        self.fuse3 = TemporalFusionBlock(C * 4, Ct, num_heads=num_heads, num_points=num_points, dropout=dropout)
        self.fuse2 = TemporalFusionBlock(C * 2, Ct, num_heads=num_heads, num_points=num_points, dropout=dropout)
        self.fuse1 = TemporalFusionBlock(C * 1, Ct, num_heads=num_heads, num_points=num_points, dropout=dropout)
        
        # Cross-scale refinement (coarse → fine)
        if use_cross_scale:
            self.cross_4to3 = CrossScaleRefinement(C * 8, C * 4)
            self.cross_3to2 = CrossScaleRefinement(C * 4, C * 2)
            self.cross_2to1 = CrossScaleRefinement(C * 2, C * 1)
        
        # Time encoding
        self.time_enc = SinusoidalTimeEncoding(Ct)
    
    def forward(
        self,
        queries: Tuple[torch.Tensor, ...],
        values: Tuple[torch.Tensor, ...],
        rel_time: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        
        q1, q2, q3, q4 = queries
        v1, v2, v3, v4 = values
        
        B = rel_time.shape[0]
        
        # Encode relative times
        time_enc = self.time_enc(rel_time)  # [B, N, Ct]
        
        # Coarse to fine fusion
        # Scale 4 (coarsest)
        f4, conf4, ent4 = self.fuse4(q4, v4, rel_time, time_enc)
        
        # Scale 3
        if self.use_cross_scale:
            q3 = self.cross_4to3(f4, q3)
        f3, conf3, ent3 = self.fuse3(q3, v3, rel_time, time_enc)
        
        # Scale 2
        if self.use_cross_scale:
            q2 = self.cross_3to2(f3, q2)
        f2, conf2, ent2 = self.fuse2(q2, v2, rel_time, time_enc)
        
        # Scale 1 (finest)
        if self.use_cross_scale:
            q1 = self.cross_2to1(f2, q1)
        f1, conf1, ent1 = self.fuse1(q1, v1, rel_time, time_enc)
        
        return (f1, f2, f3, f4), conf1, ent1


class CrossScaleRefinement(nn.Module):
    """
    Refine fine-scale query using coarse-scale prediction.
    
    MODIFIED: Replaced PixelShuffle with bilinear upsampling to eliminate
    vertical/horizontal stripe artifacts that occur when conv weights
    produce uneven values across sub-pixel positions.
    """
    def __init__(self, coarse_channels: int, fine_channels: int):
        super().__init__()
        
        # CHANGED: Bilinear + Conv instead of PixelShuffle to prevent stripe artifacts
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(coarse_channels, fine_channels, 3, padding=1),
            nn.GELU(),
        )
        
        self.gate = nn.Sequential(
            nn.Conv2d(fine_channels * 2, fine_channels, 3, padding=1),
            nn.Sigmoid(),
        )
        
        self.refine = nn.Conv2d(fine_channels * 2, fine_channels, 1)
    
    def forward(self, coarse: torch.Tensor, fine: torch.Tensor) -> torch.Tensor:
        # Upsample coarse to fine resolution
        coarse_up = self.upsample(coarse)
        
        # Handle size mismatch
        if coarse_up.shape[-2:] != fine.shape[-2:]:
            coarse_up = F.interpolate(coarse_up, size=fine.shape[-2:], mode='bilinear', align_corners=False)
        
        # Gated fusion
        combined = torch.cat([fine, coarse_up], dim=1)
        gate = self.gate(combined)
        
        refined = fine + gate * self.refine(combined)
        
        return refined


# ==============================================================================
# Factory
# ==============================================================================

def build_temporal_fusion(
    base_channels: int = 64,
    temporal_channels: int = 64,
    num_heads: int = 8,
    num_points: int = 4,
    use_cross_scale: bool = True,
    dropout: float = 0.0,
) -> MultiScaleTemporalFusion:
    return MultiScaleTemporalFusion(
        base_channels=base_channels,
        temporal_channels=temporal_channels,
        num_heads=num_heads,
        num_points=num_points,
        use_cross_scale=use_cross_scale,
        dropout=dropout,
    )


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    B, N, C, H, W = 2, 4, 64, 64, 64
    
    # Create dummy inputs
    q1 = torch.randn(B, C * 1, H, W, device=device)
    q2 = torch.randn(B, C * 2, H // 2, W // 2, device=device)
    q3 = torch.randn(B, C * 4, H // 4, W // 4, device=device)
    q4 = torch.randn(B, C * 8, H // 8, W // 8, device=device)
    
    v1 = torch.randn(B, N, C * 1, H, W, device=device)
    v2 = torch.randn(B, N, C * 2, H // 2, W // 2, device=device)
    v3 = torch.randn(B, N, C * 4, H // 4, W // 4, device=device)
    v4 = torch.randn(B, N, C * 8, H // 8, W // 8, device=device)
    
    rel_time = torch.tensor([[-0.5, -0.2, 0.3, 0.5], [-0.3, 0.0, 0.3, 0.7]], device=device)
    weights = F.softmax(-rel_time.abs(), dim=-1)
    
    # Build module
    fusion = build_temporal_fusion(
        base_channels=C,
        temporal_channels=64,
        num_heads=8,
        num_points=4,
        use_cross_scale=True,
    ).to(device)
    
    # Forward
    with torch.no_grad():
        (f1, f2, f3, f4), conf, entropy = fusion(
            queries=(q1, q2, q3, q4),
            values=(v1, v2, v3, v4),
            rel_time=rel_time,
            weights=weights,
        )
    
    print(f"\nInput shapes:")
    print(f"  Queries: {q1.shape}, {q2.shape}, {q3.shape}, {q4.shape}")
    print(f"  Values: {v1.shape}, {v2.shape}, {v3.shape}, {v4.shape}")
    
    print(f"\nOutput shapes:")
    print(f"  Fused: {f1.shape}, {f2.shape}, {f3.shape}, {f4.shape}")
    print(f"  Confidence: {conf.shape}, mean={conf.mean():.3f}")
    print(f"  Entropy: {entropy.shape}, mean={entropy.mean():.3f}")
    
    # Parameter count
    params = sum(p.numel() for p in fusion.parameters()) / 1e6
    print(f"\nParameters: {params:.2f}M")
    
    print("\n✓ High-performance fusion ready!")
