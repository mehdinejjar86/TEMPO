"""
Quick tests for the new validation system
"""
import sys
sys.path.insert(0, '/Users/nightstalker/Projects/ICML/TEMPO_no_runs')

import torch
from utils.tiling import compute_tiles, create_blend_weight, stitch_tiles


def test_tiling_functions():
    """Test tiling and stitching utilities"""
    print("\n" + "=" * 60)
    print("Testing Tiling Functions")
    print("=" * 60)

    # Test compute_tiles
    H, W = 2160, 3840  # 4K resolution
    tiles = compute_tiles(H, W, tile_size=512, overlap=64)

    print(f"\n✅ compute_tiles():")
    print(f"   4K image (2160×3840)")
    print(f"   Tile size: 512×512, Overlap: 64px")
    print(f"   Number of tiles: {len(tiles)}")
    print(f"   First tile: {tiles[0]}")
    print(f"   Last tile: {tiles[-1]}")

    # Verify no gaps (all pixels covered)
    covered = set()
    for y1, y2, x1, x2 in tiles:
        for y in range(y1, y2):
            for x in range(x1, x2):
                covered.add((y, x))

    total_pixels = H * W
    print(f"   Coverage: {len(covered)}/{total_pixels} pixels ({100*len(covered)/total_pixels:.1f}%)")

    assert len(covered) == total_pixels, "Not all pixels covered!"

    # Test create_blend_weight
    weight = create_blend_weight(512, 64)
    print(f"\n✅ create_blend_weight():")
    print(f"   Shape: {weight.shape}")
    print(f"   Center weight: {weight[0, 256, 256].item():.3f} (should be 1.0)")
    print(f"   Edge weight: {weight[0, 0, 0].item():.3f} (should be ~0.0)")
    print(f"   Edge weight: {weight[0, 0, 511].item():.3f} (should be ~0.0)")

    assert weight[0, 256, 256].item() == 1.0, "Center weight should be 1.0"
    assert weight[0, 0, 0].item() < 0.1, "Corner weight should be near 0"

    # Test stitch_tiles (simple case)
    print(f"\n✅ stitch_tiles():")

    # Create dummy tiles (all ones)
    tile_preds = [torch.ones(3, 512, 512) for _ in tiles]

    # Stitch
    stitched = stitch_tiles(tile_preds, tiles, H, W, overlap=64)

    print(f"   Stitched shape: {stitched.shape}")
    print(f"   Mean value: {stitched.mean().item():.3f} (should be 1.0)")
    print(f"   Min value: {stitched.min().item():.3f}")
    print(f"   Max value: {stitched.max().item():.3f}")

    assert stitched.shape == (3, H, W), f"Wrong output shape: {stitched.shape}"
    assert torch.allclose(stitched, torch.ones(3, H, W), atol=1e-5), "Stitching failed for uniform tiles"

    print(f"\n✅ All tiling functions passed!")


def test_x4k_test_dataset():
    """Test X4K test dataset loader"""
    print("\n" + "=" * 60)
    print("Testing X4K Test Dataset")
    print("=" * 60)

    try:
        from data.data_x4k_test import X4K1000TestDataset

        # Try to load dataset (will fail if no data, but should not crash)
        dataset = X4K1000TestDataset(
            root="/Users/nightstalker/Projects/datasets",
            step=1,
            n_frames=4,
        )

        print(f"\n✅ Dataset initialization successful")
        print(f"   Total sequences: {len(dataset.sequences)}")
        print(f"   Total samples: {len(dataset)}")

        if len(dataset) > 0:
            # Test loading first sample
            frames, anchor_times, target_time, target = dataset[0]

            print(f"\n✅ Sample loading successful:")
            print(f"   frames.shape: {frames.shape}")
            print(f"   anchor_times: {anchor_times}")
            print(f"   target_time: {target_time.item():.3f}")
            print(f"   target.shape: {target.shape}")

            # Verify shapes
            assert frames.shape[0] == 4, f"Expected N=4 frames, got {frames.shape[0]}"
            assert frames.shape[1] == 3, "Expected RGB frames"
            assert anchor_times.shape[0] == 4, "Expected 4 anchor times"
            assert target.shape[0] == 3, "Expected RGB target"

            # Verify times
            expected_times = [0.0, 1.0/3, 2.0/3, 1.0]
            for i, (actual, expected) in enumerate(zip(anchor_times, expected_times)):
                assert abs(actual - expected) < 1e-5, f"Anchor time {i} mismatch"

            # Verify target time is between 0 and 1
            assert 0 < target_time < 1, f"Target time should be in (0,1), got {target_time}"

            print(f"\n✅ All X4K test dataset checks passed!")
        else:
            print(f"\n⚠️  No test sequences found (this is OK if test data not downloaded)")

    except FileNotFoundError as e:
        print(f"\n⚠️  X4K test data not found: {e}")
        print(f"   This is expected if test dataset hasn't been downloaded")
    except Exception as e:
        print(f"\n❌ Error testing X4K dataset: {e}")
        raise


def test_tiling_inference():
    """Test end-to-end tiling inference with dummy model"""
    print("\n" + "=" * 60)
    print("Testing Tiling Inference")
    print("=" * 60)

    from utils.tiling import infer_with_tiling

    # Create dummy model that just averages input frames
    class DummyModel(torch.nn.Module):
        def forward(self, frames, anchor_times, target_time):
            # Just average all frames
            B, N, C, H, W = frames.shape
            avg = frames.mean(dim=1)  # [B, 3, H, W]

            # Dummy aux output
            aux = {
                'confidence': torch.ones(B, 1, H, W),
                'weights': torch.ones(B, N) / N,
            }

            return avg, aux

    model = DummyModel()

    # Create dummy 4K input
    H, W = 1080, 1920  # Use smaller size for speed (half 4K)
    frames = torch.randn(1, 4, 3, H, W)
    anchor_times = torch.tensor([[0.0, 0.33, 0.67, 1.0]])
    target_time = torch.tensor([0.5])

    print(f"\n  Input: {frames.shape}")
    print(f"  Tile size: 512×512, Overlap: 64px")

    # Run tiling inference
    pred = infer_with_tiling(model, frames, anchor_times, target_time,
                             tile_size=512, overlap=64)

    print(f"  Output: {pred.shape}")

    assert pred.shape == (1, 3, H, W), f"Wrong output shape: {pred.shape}"

    print(f"\n✅ Tiling inference test passed!")


if __name__ == "__main__":
    print("\n🧪 Running Validation System Tests")

    test_tiling_functions()
    test_x4k_test_dataset()
    test_tiling_inference()

    print("\n" + "=" * 60)
    print("🎉 All Tests Passed!")
    print("=" * 60)
    print("\n✅ Validation system is ready to use")
    print("✅ Run training with: python train_tempo_mixed.py [args]")
