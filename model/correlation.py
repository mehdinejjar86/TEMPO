"""
Correlation Pyramid for Offset Initialization

This module implements multi-scale correlation volumes for initializing
deformable attention offsets based on feature similarity, similar to RAFT.

The key insight: Instead of random or learned offset initialization, we can
use feature correlation to find likely correspondences between query and
temporal observations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class CorrelationPyramid(nn.Module):
    """
    Multi-scale correlation volume for offset initialization.

    Computes correlation between query features and temporal observation features
    at multiple scales to provide a coarse-to-fine motion prior.

    This is inspired by RAFT (Teed & Deng, 2020) but adapted for temporal
    multi-view synthesis where we have N observations rather than just 2 frames.
    """

    def __init__(
        self,
        channels: int,
        num_levels: int = 4,
        search_radius: int = 4,
    ):
        """
        Args:
            channels: Number of input feature channels
            num_levels: Number of pyramid levels (default: 4)
            search_radius: Search radius in pixels at finest level (default: 4)
        """
        super().__init__()
        self.num_levels = num_levels
        self.search_radius = search_radius

        # Feature projection to reduce channels before correlation
        # This reduces computation and memory while preserving discriminative power
        self.feature_proj = nn.Conv2d(channels, channels // 2, 1)

    def forward(
        self,
        query_feat: torch.Tensor,
        value_feats: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Compute correlation pyramid between query and value features.

        Args:
            query_feat: [B, C, H, W] - query frame features
            value_feats: [B, N, C, H, W] - temporal observation features

        Returns:
            correlations: List of [B, N, H, W, (2*r+1)²] at each pyramid level
                         where r is the search radius at that level
        """
        B, N, C, H, W = value_feats.shape

        # Project features to reduce dimensionality
        q = self.feature_proj(query_feat)  # [B, C/2, H, W]
        v = self.feature_proj(value_feats.flatten(0, 1))  # [B*N, C/2, H, W]
        v = v.view(B, N, -1, H, W)  # [B, N, C/2, H, W]

        # Build pyramid
        correlations = []
        for level in range(self.num_levels):
            scale = 2 ** level
            radius = max(1, self.search_radius // scale)

            # Downsample if needed
            if scale > 1:
                q_level = F.avg_pool2d(q, scale, scale)
                v_level = F.avg_pool2d(v.flatten(0, 1), scale, scale)
                v_level = v_level.view(B, N, -1, *v_level.shape[-2:])
            else:
                q_level, v_level = q, v

            # Compute correlation within search radius
            corr = self._compute_correlation(q_level, v_level, radius)
            correlations.append(corr)

        return correlations

    def _compute_correlation(
        self,
        query: torch.Tensor,
        values: torch.Tensor,
        radius: int,
    ) -> torch.Tensor:
        """
        Compute local correlation within search radius.

        Args:
            query: [B, C, H, W]
            values: [B, N, C, H, W]
            radius: search radius in pixels

        Returns:
            correlation: [B, N, H, W, (2*r+1)²]
        """
        B, N, C, H, W = values.shape

        # Normalize features for cosine similarity
        query = F.normalize(query, dim=1)  # [B, C, H, W]
        values = F.normalize(values, dim=2)  # [B, N, C, H, W]

        # Flatten B and N for padding (pad doesn't support 5D with replicate mode)
        values_flat = values.reshape(B * N, C, H, W)

        # Compute correlation for each offset in search window
        correlations = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # Shift values by (dy, dx)
                if dy == 0 and dx == 0:
                    values_shifted = values_flat
                else:
                    # Use roll for shifting (handles boundaries with wraparound)
                    # This is simpler and avoids padding issues
                    values_shifted = torch.roll(values_flat, shifts=(dy, dx), dims=(-2, -1))

                # Reshape back
                values_shifted = values_shifted.view(B, N, C, H, W)

                # Dot product (cosine similarity since normalized)
                corr = (query.unsqueeze(1) * values_shifted).sum(dim=2)  # [B, N, H, W]
                correlations.append(corr)

        # Stack all offsets: [B, N, H, W, (2*r+1)²]
        correlation = torch.stack(correlations, dim=-1)

        return correlation


def extract_correlation_peak_offset(
    correlation: torch.Tensor,
    radius: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Extract offset to correlation peak with soft-argmax for differentiability.

    Instead of hard argmax, we use temperature-scaled softmax to get a
    soft peak location that's differentiable and more robust to noise.

    Args:
        correlation: [B, N, H, W, (2*r+1)²] - correlation volume
        radius: search radius used to compute correlation
        temperature: softmax temperature (lower = sharper peak, higher = smoother)

    Returns:
        offsets: [B, N, H, W, 2] - (dx, dy) to correlation peak in [-1, 1] range
    """
    B, N, H, W, num_offsets = correlation.shape
    grid_size = 2 * radius + 1

    assert num_offsets == grid_size ** 2, f"Expected {grid_size**2} offsets, got {num_offsets}"

    # Apply temperature-scaled softmax
    weights = F.softmax(correlation / temperature, dim=-1)  # [B, N, H, W, (2*r+1)²]

    # Create offset grid: (dy, dx) for each position in search window
    # Grid ranges from (-r, -r) to (r, r)
    dy_grid = torch.arange(-radius, radius + 1, device=correlation.device, dtype=torch.float32)
    dx_grid = torch.arange(-radius, radius + 1, device=correlation.device, dtype=torch.float32)

    # Meshgrid and flatten
    dy_grid, dx_grid = torch.meshgrid(dy_grid, dx_grid, indexing='ij')
    dy_grid = dy_grid.reshape(-1)  # [(2*r+1)²]
    dx_grid = dx_grid.reshape(-1)  # [(2*r+1)²]

    # Weighted sum to get soft peak location
    offset_dy = (weights * dy_grid.view(1, 1, 1, 1, -1)).sum(dim=-1)  # [B, N, H, W]
    offset_dx = (weights * dx_grid.view(1, 1, 1, 1, -1)).sum(dim=-1)  # [B, N, H, W]

    # Stack and normalize to [-1, 1] range (for grid_sample compatibility)
    # Note: radius pixels corresponds to the full search range
    offsets = torch.stack([offset_dx / radius, offset_dy / radius], dim=-1)
    offsets = offsets.clamp(-1.0, 1.0)  # [B, N, H, W, 2]

    return offsets


def aggregate_temporal_offsets(
    offsets: torch.Tensor,
    temporal_weights: torch.Tensor,
    num_heads: int,
    num_points: int,
) -> torch.Tensor:
    """
    Aggregate per-observation offsets into per-head-per-point offsets.

    The correlation gives us one offset per observation. We aggregate these
    across observations using temporal weights, then broadcast to all heads
    and points as initialization.

    Args:
        offsets: [B, N, H, W, 2] - offset per observation
        temporal_weights: [B, N] - importance weight per observation
        num_heads: number of attention heads
        num_points: number of sampling points per head

    Returns:
        init_offsets: [B, num_heads, num_points, H, W, 2]
    """
    B, N, H, W, _ = offsets.shape

    # Weight offsets by temporal importance
    # [B, N, H, W, 2] * [B, N, 1, 1, 1] -> [B, N, H, W, 2]
    weighted_offsets = offsets * temporal_weights.view(B, N, 1, 1, 1)

    # Sum across observations (normalized by sum of weights)
    base_offset = weighted_offsets.sum(dim=1)  # [B, H, W, 2]
    weight_sum = temporal_weights.sum(dim=1, keepdim=True)  # [B, 1]
    base_offset = base_offset / (weight_sum.view(B, 1, 1, 1) + 1e-8)

    # Broadcast to all heads and points
    # [B, H, W, 2] -> [B, num_heads, num_points, H, W, 2]
    init_offsets = base_offset.unsqueeze(1).unsqueeze(2).expand(
        B, num_heads, num_points, H, W, 2
    )

    return init_offsets
