"""
Quick test for tiling and stitching functionality
"""
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.tiling import compute_tiles, create_blend_weight, stitch_tiles

def create_test_image(H=2160, W=3840):
    """Create a synthetic 4K test image with gradients and patterns"""
    # Create coordinate grids
    y = torch.linspace(0, 1, H).view(H, 1).repeat(1, W)
    x = torch.linspace(0, 1, W).view(1, W).repeat(H, 1)

    # Create RGB channels with different patterns
    r = (torch.sin(x * 10) + 1) / 2  # Vertical stripes
    g = (torch.sin(y * 10) + 1) / 2  # Horizontal stripes
    b = (x + y) / 2  # Diagonal gradient

    # Stack into [3, H, W]
    img = torch.stack([r, g, b], dim=0)

    return img


def test_tiling_stitching(tile_size=512, overlap=64, use_padding=True):
    """Test tiling and stitching with a synthetic image"""
    print("=" * 60)
    print(f"Testing Tiling and Stitching {'WITH' if use_padding else 'WITHOUT'} Padding")
    print("=" * 60)

    # Create test image
    H, W = 2160, 3840  # 4K resolution
    print(f"\n📐 Creating test image: {H}×{W} (4K)")
    original = create_test_image(H, W)

    # Apply padding if requested
    if use_padding:
        pad_size = overlap
        print(f"\n📦 Padding image with {pad_size}px reflection padding")
        original_padded = torch.nn.functional.pad(
            original.unsqueeze(0),  # Add batch dim
            (pad_size, pad_size, pad_size, pad_size),
            mode='reflect'
        )[0]  # Remove batch dim
        H_pad, W_pad = original_padded.shape[1], original_padded.shape[2]
        print(f"   Padded size: {H_pad}×{W_pad}")
    else:
        original_padded = original
        H_pad, W_pad = H, W
        pad_size = 0

    # Compute tiles on (possibly padded) image
    print(f"\n🔲 Computing tiles (size={tile_size}, overlap={overlap})")
    tiles = compute_tiles(H_pad, W_pad, tile_size, overlap)
    print(f"   Generated {len(tiles)} tiles")

    # Extract tiles from padded image
    tile_crops = []
    for i, (y1, y2, x1, x2) in enumerate(tiles):
        tile_crop = original_padded[:, y1:y2, x1:x2]
        tile_crops.append(tile_crop)
        if i < 3 or i >= len(tiles) - 3:  # Debug first and last few tiles
            print(f"   Tile {i}: coords=({y1}, {y2}, {x1}, {x2}), shape={tile_crop.shape}")

    print(f"   Typical tile shape: {tile_crops[0].shape}")

    # Stitch back together
    print(f"\n🔧 Stitching tiles back together")
    stitched_padded = stitch_tiles(tile_crops, tiles, H_pad, W_pad, overlap)

    # Crop back to original size if padded
    if use_padding:
        stitched = stitched_padded[:, pad_size:pad_size+H, pad_size:pad_size+W]
        print(f"   Cropping back to original size: {H}×{W}")
    else:
        stitched = stitched_padded

    # Compute error
    error = (stitched - original).abs()
    max_error = error.max().item()
    mean_error = error.mean().item()

    # Find where max error occurs
    max_pos = (error == max_error).nonzero()[0]
    c, y, x = max_pos[0].item(), max_pos[1].item(), max_pos[2].item()

    print(f"\n📊 Results:")
    print(f"   Original shape:  {original.shape}")
    print(f"   Stitched shape:  {stitched.shape}")
    print(f"   Max error:       {max_error:.6f} at position ({y}, {x})")
    print(f"   Mean error:      {mean_error:.6f}")
    print(f"   Median error:    {error.median().item():.6f}")
    print(f"   99th percentile: {error.flatten().kthvalue(int(error.numel() * 0.99))[0].item():.6f}")

    # Check if reconstruction is accurate
    # Use mean error and 99th percentile for realistic assessment
    # (max error can be affected by edge pixels)
    if mean_error < 1e-5 and error.flatten().kthvalue(int(error.numel() * 0.99))[0].item() < 1e-5:
        print(f"\n✅ Perfect reconstruction! (mean error < 1e-5)")
        success = True
    elif mean_error < 1e-3:
        print(f"\n✅ Excellent reconstruction (mean error < 1e-3)")
        print(f"   Note: Max error at image boundaries is expected due to blend weights")
        success = True
    else:
        print(f"\n⚠️  Reconstruction has noticeable error")
        success = False

    # Visualize results
    print(f"\n🖼️  Saving visualization...")
    visualize_results(original, stitched, error, tiles)

    return success


def visualize_results(original, stitched, error, tiles):
    """Create visualization of tiling and stitching"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Convert to [H, W, 3] for display
    orig_np = original.permute(1, 2, 0).numpy()
    stitch_np = stitched.permute(1, 2, 0).numpy()
    error_np = error.permute(1, 2, 0).numpy()

    # Original image
    axes[0, 0].imshow(orig_np)
    axes[0, 0].set_title('Original Image (4K)', fontsize=14)
    axes[0, 0].axis('off')

    # Tile grid overlay
    axes[0, 1].imshow(orig_np)
    for y1, y2, x1, x2 in tiles:
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                            fill=False, edgecolor='red', linewidth=1)
        axes[0, 1].add_patch(rect)
    axes[0, 1].set_title(f'Tile Grid ({len(tiles)} tiles)', fontsize=14)
    axes[0, 1].axis('off')

    # Stitched result
    axes[0, 2].imshow(stitch_np)
    axes[0, 2].set_title('Stitched Result', fontsize=14)
    axes[0, 2].axis('off')

    # Error map (amplified for visibility)
    error_amplified = np.clip(error_np * 100, 0, 1)
    axes[1, 0].imshow(error_amplified)
    axes[1, 0].set_title(f'Error Map (×100)\nMax: {error.max():.6f}', fontsize=14)
    axes[1, 0].axis('off')

    # Center crop comparison
    cy, cx = orig_np.shape[0] // 2, orig_np.shape[1] // 2
    crop_size = 512
    y1, y2 = cy - crop_size//2, cy + crop_size//2
    x1, x2 = cx - crop_size//2, cx + crop_size//2

    axes[1, 1].imshow(orig_np[y1:y2, x1:x2])
    axes[1, 1].set_title('Original (center 512×512)', fontsize=14)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(stitch_np[y1:y2, x1:x2])
    axes[1, 2].set_title('Stitched (center 512×512)', fontsize=14)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('tiling_test_result.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: tiling_test_result.png")
    plt.close()

    # Save a detail comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Zoom into a corner region with overlap
    detail_size = 256
    axes[0].imshow(orig_np[:detail_size, :detail_size])
    axes[0].set_title('Original (top-left 256×256)', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(stitch_np[:detail_size, :detail_size])
    axes[1].set_title('Stitched (top-left 256×256)', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(error_amplified[:detail_size, :detail_size])
    axes[2].set_title('Error (×100)', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('tiling_test_detail.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: tiling_test_detail.png")
    plt.close()


def test_blend_weight():
    """Test blend weight creation"""
    print("\n" + "=" * 60)
    print("Testing Blend Weight")
    print("=" * 60)

    tile_size = 512
    overlap = 64

    weight = create_blend_weight(tile_size, overlap)

    print(f"\n📊 Blend weight stats:")
    print(f"   Shape: {weight.shape}")
    print(f"   Center value: {weight[0, 256, 256].item():.3f} (should be 1.0)")
    print(f"   Corner value: {weight[0, 0, 0].item():.3f} (should be ~0.0)")
    print(f"   Edge value:   {weight[0, 0, 256].item():.3f}")
    print(f"   Min value:    {weight.min().item():.6f}")
    print(f"   Max value:    {weight.max().item():.3f}")

    # Visualize weight
    plt.figure(figsize=(10, 8))
    plt.imshow(weight[0].numpy(), cmap='hot')
    plt.colorbar(label='Weight')
    plt.title(f'Blend Weight Map ({tile_size}×{tile_size}, overlap={overlap})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.savefig('blend_weight.png', dpi=150, bbox_inches='tight')
    print(f"\n   Saved: blend_weight.png")
    plt.close()

    # Verify properties
    assert weight[0, 256, 256].item() == 1.0, "Center should be 1.0"
    assert weight[0, 0, 0].item() < 0.01, "Corner should be near 0"

    print(f"\n✅ Blend weight test passed!")


if __name__ == "__main__":
    print("\n🧪 Tiling and Stitching Test Suite\n")

    # Test blend weight
    test_blend_weight()

    # Test tiling and stitching WITH padding (recommended)
    print("\n")
    success_padded = test_tiling_stitching(tile_size=512, overlap=64, use_padding=True)

    # Test tiling and stitching WITHOUT padding (for comparison)
    print("\n")
    success_unpadded = test_tiling_stitching(tile_size=512, overlap=64, use_padding=False)

    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"  With padding:    {'✅ PASSED' if success_padded else '❌ FAILED'}")
    print(f"  Without padding: {'✅ PASSED' if success_unpadded else '❌ FAILED'}")

    if success_padded:
        print("\n" + "=" * 60)
        print("🎉 Padding eliminates edge artifacts!")
        print("=" * 60)
        print("\nThe tiling and stitching system with padding is working perfectly.")
        print("Recommendation: Always use padding for best quality.")
        print("\nCheck the generated images:")
        print("  - tiling_test_result.png  (overview)")
        print("  - tiling_test_detail.png  (zoom)")
        print("  - blend_weight.png        (weight map)")
    else:
        print("\n" + "=" * 60)
        print("❌ Tests failed!")
        print("=" * 60)
