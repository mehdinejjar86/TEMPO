"""
Tiling and Stitching Utilities for High-Resolution Inference

Enables tile-based inference for large images (e.g., 4K) that don't fit in GPU memory.
Uses overlapping tiles with weighted blending for smooth stitching.
"""
from typing import List, Tuple
import torch
import torch.nn as nn


def compute_tiles(H: int, W: int, tile_size: int = 512, overlap: int = 64) -> List[Tuple[int, int, int, int]]:
    """
    Compute tile coordinates for overlapping grid coverage.

    Tiles are placed with stride = tile_size - overlap, ensuring complete
    coverage with overlapping regions for smooth blending.

    Args:
        H: Image height
        W: Image width
        tile_size: Size of each square tile (default: 512)
        overlap: Overlap between adjacent tiles (default: 64)

    Returns:
        List of (y_start, y_end, x_start, x_end) tuples

    Example:
        For 2160×3840 image with 512×512 tiles and 64px overlap:
        - Stride = 512 - 64 = 448
        - Number of tiles ≈ (2160/448) × (3840/448) ≈ 5 × 9 = 45 tiles
    """
    stride = tile_size - overlap
    tiles = []

    # Iterate over grid with stride
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            # Clamp to image bounds
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)

            # Adjust start if we hit the boundary (keep tile_size constant)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tiles.append((y_start, y_end, x_start, x_end))

    return tiles


def create_blend_weight(tile_size: int, overlap: int) -> torch.Tensor:
    """
    Create 2D weight map with linear ramps at tile edges.

    Weight profile:
      - Center of tile: weight = 1.0
      - Overlap region: linear ramp from 0.0 (edge) to 1.0 (center)
      - This ensures smooth blending where tiles overlap

    Args:
        tile_size: Size of the square tile
        overlap: Width of the overlap region

    Returns:
        weight: [1, tile_size, tile_size] tensor with blend weights

    Example:
        For 512×512 tile with 64px overlap:
        - Top 64 rows: ramp from 0.0 to 1.0
        - Bottom 64 rows: ramp from 1.0 to 0.0
        - Left 64 cols: ramp from 0.0 to 1.0
        - Right 64 cols: ramp from 1.0 to 0.0
        - Corners: multiplicative blend of both ramps
    """
    weight = torch.ones(1, tile_size, tile_size)

    # Linear ramp from 0 to 1
    ramp = torch.linspace(0, 1, overlap)

    # Apply ramps to edges
    # Top edge
    weight[:, :overlap, :] *= ramp.view(1, -1, 1)
    # Bottom edge
    weight[:, -overlap:, :] *= ramp.flip(0).view(1, -1, 1)
    # Left edge
    weight[:, :, :overlap] *= ramp.view(1, 1, -1)
    # Right edge
    weight[:, :, -overlap:] *= ramp.flip(0).view(1, 1, -1)

    return weight


def stitch_tiles(
    tile_predictions: List[torch.Tensor],
    tile_coords: List[Tuple[int, int, int, int]],
    H: int,
    W: int,
    overlap: int = 64
) -> torch.Tensor:
    """
    Stitch overlapping tiles using weighted blending.

    Tiles are blended using weight maps that fade out at edges, ensuring
    smooth transitions in overlap regions and avoiding seam artifacts.

    Args:
        tile_predictions: List of [C, tile_size, tile_size] predictions
        tile_coords: List of (y1, y2, x1, x2) coordinates matching predictions
        H: Output image height
        W: Output image width
        overlap: Overlap width used for computing blend weights

    Returns:
        output: [C, H, W] stitched image

    Algorithm:
        1. Initialize output accumulator and weight map to zeros
        2. For each tile:
           a. Create blend weight (fade out at edges)
           b. Add weighted tile to output accumulator
           c. Add weight to weight map
        3. Normalize: output / weight_map
    """
    device = tile_predictions[0].device
    C = tile_predictions[0].shape[0]
    tile_size = tile_predictions[0].shape[-1]

    # Initialize accumulators
    output = torch.zeros(C, H, W, device=device)
    weight_map = torch.zeros(1, H, W, device=device)

    # Create blend weight for tile (reused for all tiles)
    blend_weight = create_blend_weight(tile_size, overlap).to(device)

    # Accumulate weighted tiles
    for pred, (y1, y2, x1, x2) in zip(tile_predictions, tile_coords):
        output[:, y1:y2, x1:x2] += pred * blend_weight
        weight_map[:, y1:y2, x1:x2] += blend_weight

    # Normalize by accumulated weights (avoid division by zero)
    output = output / weight_map.clamp(min=1e-6)

    return output


@torch.no_grad()
def infer_with_tiling(
    model: nn.Module,
    frames: torch.Tensor,
    anchor_times: torch.Tensor,
    target_time: torch.Tensor,
    tile_size: int = 512,
    overlap: int = 64,
    pad_size: int = None
) -> torch.Tensor:
    """
    Tile-based inference for high-resolution images with edge padding.

    Processes large images by:
    1. Pad image edges with reflection padding (eliminates edge artifacts)
    2. Divide into overlapping tiles
    3. Run model on each tile independently
    4. Stitch results with weighted blending
    5. Crop back to original size

    Args:
        model: TEMPO model (or any model with forward(frames, anchor_times, target_time))
        frames: [1, N, 3, H, W] anchor frames (batch_size MUST be 1)
        anchor_times: [1, N] anchor timestamps
        target_time: [1] target timestamp
        tile_size: Size of square tiles (default: 512)
        overlap: Overlap between tiles (default: 64)
        pad_size: Padding to add before tiling (default: overlap)
                 Eliminates edge artifacts by ensuring all pixels have tile coverage

    Returns:
        pred: [1, 3, H, W] stitched prediction at full resolution

    Memory:
        - Peak memory ∝ tile_size² (not full image size)
        - For 4K (2160×3840): ~60 tiles × 512² each
        - Much lower than direct 2160×3840 inference

    Speed:
        - ~2-3x slower than direct inference (tile overhead)
        - Necessary trade-off for memory constraints
    """
    B, N, C, H_orig, W_orig = frames.shape
    assert B == 1, "Tiled inference only supports batch_size=1 (single image at a time)"

    # Default padding is the overlap size
    if pad_size is None:
        pad_size = overlap

    # Pad input frames with reflection padding to avoid edge artifacts
    # PyTorch's reflection padding only supports up to 4D tensors
    # Reshape [B, N, C, H, W] -> [B*N, C, H, W] for padding
    frames_reshaped = frames.view(B * N, C, H_orig, W_orig)

    # pad: (left, right, top, bottom)
    frames_padded_4d = torch.nn.functional.pad(
        frames_reshaped,
        (pad_size, pad_size, pad_size, pad_size),
        mode='reflect'
    )

    # Reshape back to [B, N, C, H_pad, W_pad]
    _, _, H_pad, W_pad = frames_padded_4d.shape
    frames_padded = frames_padded_4d.view(B, N, C, H_pad, W_pad)

    # Compute tile coordinates on padded image
    tiles = compute_tiles(H_pad, W_pad, tile_size, overlap)

    # Process each tile
    tile_preds = []
    for y1, y2, x1, x2 in tiles:
        # Extract tile from all N anchor frames
        frames_tile = frames_padded[:, :, :, y1:y2, x1:x2]

        # Run model on tile
        pred_tile, _ = model(frames_tile, anchor_times, target_time)

        # Store prediction (remove batch dimension)
        tile_preds.append(pred_tile[0])  # [3, tile_size, tile_size]

    # Stitch tiles back to full resolution (padded size)
    pred_padded = stitch_tiles(tile_preds, tiles, H_pad, W_pad, overlap)

    # Crop back to original size
    pred_full = pred_padded[:, pad_size:pad_size+H_orig, pad_size:pad_size+W_orig]

    # Return with batch dimension
    return pred_full.unsqueeze(0)  # [1, 3, H, W]
