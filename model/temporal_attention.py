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

class UpdateBlock(nn.Module):
    """
    GRU-based update block for iterative offset refinement.
    Predicts delta_offset and delta_attn from current state.
    """
    def __init__(self, hidden_dim: int, input_dim: int, head_dim: int, num_points: int):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.head_dim = head_dim
        self.num_points = num_points
        
        # Output prediction heads
        self.to_delta_offset = nn.Linear(hidden_dim, num_points * 2)
        self.to_delta_attn = nn.Linear(hidden_dim, num_points)
        
        # Initialize
        nn.init.zeros_(self.to_delta_offset.weight)
        nn.init.zeros_(self.to_delta_offset.bias)
        nn.init.zeros_(self.to_delta_attn.weight)
        nn.init.zeros_(self.to_delta_attn.bias)

    def forward(self, net: torch.Tensor, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            net: [B*heads*H*W, hidden] hidden state
            inp: [B*heads*H*W, input_dim] input features (sampled values, correlation, etc)
        Returns:
            net: updated hidden state
            delta_offset: [B*heads, H, W, points, 2]
            delta_attn: [B*heads, H, W, points]
        """
        net = self.gru(inp, net)
        
        delta_offset = self.to_delta_offset(net)
        delta_attn = self.to_delta_attn(net)
        
        return net, delta_offset, delta_attn


class DeformableTemporalAttention(nn.Module):
    """
    High-performance temporal attention with iterative offset refinement.
    
    Features:
      1. Correlation Peak Initialization: Uses cost volume to find initial motion.
      2. Iterative Refinement: GRU-based loop refines offsets (3 iterations).
      3. Deformable Sampling: Samples features at refined locations.
    """
    def __init__(
        self,
        channels: int,
        temporal_channels: int,
        num_heads: int = 8,
        num_points: int = 4,
        dropout: float = 0.0,
        num_iters: int = 3,  # Number of refinement iterations
    ):
        super().__init__()
        assert channels % num_heads == 0
        
        self.channels = channels
        self.temporal_channels = temporal_channels
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = channels // num_heads
        self.num_iters = num_iters
        
        # =====================
        # Feature Extractors
        # =====================
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        
        # Context extraction for UpdateBlock
        self.cnet = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )
        
        # =====================
        # Iterative Update Components
        # =====================
        # Input to GRU: [Sampled Features + Correlation info + Motion features]
        # For simplicity, we feed: [Head Features (head_dim) + Current Offset (2)]
        input_dim = self.head_dim + 2
        hidden_dim = self.head_dim
        
        self.update_block = UpdateBlock(
            hidden_dim=hidden_dim, 
            input_dim=input_dim, 
            head_dim=self.head_dim, 
            num_points=num_points
        )
        
        # =====================
        # Time Modulation
        # =====================
        # Time bias for attention (closer observations → higher weight)
        self.time_to_attn_bias = nn.Linear(temporal_channels, num_heads)
        
        # =====================
        # Output
        # =====================
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        
        self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.out_proj[0].weight)
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def _compute_correlation(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute local correlation volume to initialize offsets.
        Result: Best matching offset among 5 local neighbors (0,0), (+1,0), (-1,0), etc.
        """
        B, heads, C, H, W = q.shape
        device = q.device
        
        # 5 search points: Center, Left, Right, Up, Down
        # Shifts in (dx, dy)
        shifts = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        offset_vals = torch.tensor(shifts, device=device, dtype=torch.float32) # [5, 2]
        
        # We compute dot product <q, v_shifted>
        # q: [B, heads, C, H, W]
        # v: [B, heads, C, H, W] assuming v is the "nearest neighbor" observation passed in context?
        # Actually _compute_correlation is called with v=? 
        # In current init, we don't call it yet. Let's fix usage in Forward too if needed.
        # But providing the capability is key.
        
        scores = []
        for dx, dy in shifts:
            # Shift v
            v_shifted = torch.roll(v, shifts=(dy, dx), dims=(-2, -1))
            
            # Zero out rolled-over parts (simple padding emulation)
            if dy > 0: v_shifted[..., :dy, :] = 0
            if dy < 0: v_shifted[..., dy:, :] = 0
            if dx > 0: v_shifted[..., :, :dx] = 0
            if dx < 0: v_shifted[..., :, dx:] = 0
            
            # Dot product
            score = (q * v_shifted).sum(dim=2) # [B, heads, H, W]
            scores.append(score)
            
        scores = torch.stack(scores, dim=-1) # [B, heads, H, W, 5]
        best_idx = scores.argmax(dim=-1) # [B, heads, H, W]
        
        # Convert index to offset vector
        # offset_vals: [5, 2]
        pad = max(H, W) # Normalize to [-1, 1] range? 
        # Offsets in grid_sample are normalized to [-1, 1]. 
        # 1 pixel = 2/H or 2/W
        scale_y = 2.0 / H
        scale_x = 2.0 / W
        
        # Gather best offsets
        # Use simple indexing since 5 is small
        out_offsets = torch.zeros(B, heads, H, W, 2, device=device)
        
        # This loop is slow but fine for logic demonstration. 
        # Vectorized gather is better:
        best_off = offset_vals[best_idx] # [B, heads, H, W, 2]
        
        # Scale to normalized coordinates
        out_offsets[..., 0] = best_off[..., 1] * scale_x # x is dim 1 in offsets
        out_offsets[..., 1] = best_off[..., 0] * scale_y # y is dim 0 in shifts
        
        return out_offsets

    def forward(
        self,
        query: torch.Tensor,          # [B, C, H, W]
        values: torch.Tensor,         # [B, N, C, H, W]
        rel_time: torch.Tensor,       # [B, N]
        time_enc: torch.Tensor,       # [B, N, Ct]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, C, H, W = query.shape
        N = values.shape[1]
        device = query.device
        
        # 1. Project features
        q = self.q_proj(query)             # [B, C, H, W]
        v = self.v_proj(values.view(-1, C, H, W)).view(B, N, C, H, W)
        
        # Reshape to heads: [B, heads, head_dim, H, W]
        q_heads = q.view(B, self.num_heads, self.head_dim, H, W)
        v_heads = v.view(B, N, self.num_heads, self.head_dim, H, W)
        
        # 2. Extract Context (Hidden State for GRU)
        net = self.cnet(query)             # [B, C, H, W]
        # Flatten for GRU: [B*heads*H*W, head_dim]
        net = net.view(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2).reshape(-1, self.head_dim)
        
        # 3. Initialization
        # Find nearest neighbor for initialization
        nearest_idx = rel_time.abs().argmin(dim=1) # [B]
        v_nearest = torch.stack([v_heads[b, nearest_idx[b]] for b in range(B)], dim=0) # [B, heads, head_dim, H, W]
        
        # Compute correlation-based offset
        # q_heads: [B, heads, head_dim, H, W]
        best_offset = self._compute_correlation(q_heads, v_nearest) # [B, heads, H, W, 2]
        
        # Initialize coords with specific correlation peak
        # Expand across points? For now, all points start at the correlation peak
        coords_init = best_offset.unsqueeze(4).repeat(1, 1, 1, 1, self.num_points, 1) # [B, heads, H, W, points, 2]
        
        attn_logits = torch.zeros(B, self.num_heads, H, W, N, self.num_points, device=device)
        
        # Base Grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1) # [H, W, 2]
        
        # 4. Iterative Loop
        curr_offsets = coords_init # [B, heads, H, W, points, 2]
        curr_attn = attn_logits    # [B, heads, H, W, N, points]
        
        for i in range(self.num_iters):
             # 4a. Sample features at current offsets
             # Use v_nearest (cached)
             # Sample from nearest neighbor at current point 0
             p0_offset = curr_offsets[..., 0, :] # [B, heads, H, W, 2]
             
             sample_grid = base_grid.view(1, 1, H, W, 2) + p0_offset
             sample_grid = sample_grid.view(B*self.num_heads, H, W, 2).clamp(-1, 1)
             
             # Flatten v_nearest for grid_sample
             v_flat = v_nearest.view(B*self.num_heads, self.head_dim, H, W)
             
             padding_mode = "zeros" if torch.torch.backends.mps.is_available() else "border"
             sampled_feat = F.grid_sample(v_flat, sample_grid, align_corners=True, padding_mode=padding_mode) # [B*h, C/h, H, W]
             
             # Input to GRU: [Sampled Features (C/h) + Current Offset (2)]
             sampled_feat = sampled_feat.permute(0, 2, 3, 1).reshape(-1, self.head_dim)
             offset_feat = p0_offset.view(-1, 2)
             
             gru_in = torch.cat([sampled_feat, offset_feat], dim=-1)
             
             # 4b. Update State
             net, delta_offset_flat, delta_attn_flat = self.update_block(net, gru_in)
             
             # 4c. Apply Updates
             # delta_offset: [B*heads*H*W, points*2] -> [B, heads, H, W, points, 2]
             d_offset = delta_offset_flat.view(B, self.num_heads, H, W, self.num_points, 2)
             curr_offsets = curr_offsets + d_offset
             
             # delta_attn: [B*heads*H*W, points] -> [B, heads, H, W, points] (shared across N)
             d_attn = delta_attn_flat.view(B, self.num_heads, H, W, self.num_points)
             # Expand to N (global bias update)
             curr_attn = curr_attn + d_attn.unsqueeze(4)

        # 5. Final Sampling & Aggregation
        # Normalize weights
        # Time bias
        # t_enc: [B, N, Ct]
        # bias: [B, N, heads]
        t_bias = self.time_to_attn_bias(time_enc).permute(0, 2, 1).view(B, self.num_heads, 1, 1, N, 1)
        
        final_attn = curr_attn + t_bias
        attn_weights = F.softmax(final_attn.view(B, self.num_heads, H, W, -1), dim=-1)
        attn_weights = attn_weights.view(B, self.num_heads, H, W, N, self.num_points)
        attn_weights = self.dropout(attn_weights)
        
        # Sample all
        all_samples = []
        for n in range(N):
            samples_n = []
            for p in range(self.num_points):
                off = curr_offsets[..., p, :] # [B, heads, H, W, 2]
                grid = base_grid.view(1, 1, H, W, 2) + off
                grid = grid.view(B*self.num_heads, H, W, 2).clamp(-1, 1)
                
                v_n = v_heads[:, n].reshape(B*self.num_heads, self.head_dim, H, W)
                s = F.grid_sample(v_n, grid, align_corners=True, padding_mode=padding_mode)
                samples_n.append(s.view(B, self.num_heads, self.head_dim, H, W))
            all_samples.append(torch.stack(samples_n, dim=-1)) # [B, h, d, H, W, p]

        # Stack: [B, N, h, d, H, W, p]
        all_samples = torch.stack(all_samples, dim=1)
        
        # Weighted sum
        # attn: [B, h, H, W, N, p]
        # samples: [B, N, h, d, H, W, p] -> permute to match
        all_samples = all_samples.permute(0, 2, 4, 5, 1, 6, 3) # [B, h, H, W, N, p, d]
        
        output = (all_samples * attn_weights.unsqueeze(-1)).sum(dim=(4, 5)) # [B, h, H, W, d]
        output = output.permute(0, 1, 4, 2, 3).reshape(B, C, H, W)
        
        output = self.out_proj(output) * torch.sigmoid(self.gate)
        
        # Real confidence statistics
        # Entropy of the attention distribution (over N*points)
        att_flat = attn_weights.view(B, self.num_heads, H, W, -1)
        entropy = -(att_flat * (att_flat + 1e-8).log()).sum(dim=-1) # [B, heads, H, W]
        entropy = entropy.mean(dim=1, keepdim=True) # [B, 1, H, W]
        
        # Confidence = 1 - normalized_entropy
        max_entropy = math.log(N * self.num_points)
        confidence = 1.0 - (entropy / (max_entropy + 1e-8)).clamp(0, 1)
        
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
    """
    def __init__(self, coarse_channels: int, fine_channels: int):
        super().__init__()
        
        self.upsample = nn.Sequential(
            nn.Conv2d(coarse_channels, fine_channels * 4, 1),
            nn.PixelShuffle(2),
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
