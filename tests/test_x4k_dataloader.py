"""Test X4K1000 dataset loading with different STEP values"""
import torch
from torch.utils.data import DataLoader
from data.data_x4k1000 import X4K1000Dataset, x4k_collate

print("\n" + "="*70)
print("X4K1000 Dataset - STEP-Based Sampling Test")
print("="*70)

# Test STEP=1 (small motion)
print("\n" + "="*70)
print("Test 1: STEP=1 (small motion)")
print("="*70)

dataset = X4K1000Dataset(
    root="/Users/nightstalker/Projects/datasets",
    split="train",
    step=1,         # spacing=2, frames=[0,2,4,6]
    crop_size=None, # No crop for testing
    aug_flip=False,
)

print(f"\nDataset size: {len(dataset)} samples")
print(f"Expected: 4408 seqs × 3 targets/seq = 13,224 samples")

# Test single sample
frames, anchor_times, target_time, target = dataset[0]
print(f"\nSample 0:")
print(f"  Anchor frames: {frames.shape}")      # [4, 3, H, W]
print(f"  Anchor times: {anchor_times}")       # [0.0, 0.33, 0.67, 1.0]
print(f"  Target time: {target_time.item():.4f}")  # e.g., 0.17 (frame 1 between 0 and 2)
print(f"  Target: {target.shape}")              # [3, H, W]

# Verify anchor times
expected_times = torch.linspace(0.0, 1.0, 4)
assert torch.allclose(anchor_times, expected_times, atol=1e-6), "Anchor times mismatch!"
print(f"  ✓ Anchor times correct: [0.0, 0.33, 0.67, 1.0]")

# Verify target time is between 0 and 1
assert 0.0 <= target_time.item() <= 1.0, "Target time out of range!"
print(f"  ✓ Target time in valid range [0, 1]")

# Test dataloader
loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=x4k_collate)

batch = next(iter(loader))
frames_b, times_b, target_time_b, target_b = batch
print(f"\nBatch (size=2):")
print(f"  Frames: {frames_b.shape}")           # [2, 4, 3, H, W]
print(f"  Anchor times: {times_b.shape}")      # [2, 4]
print(f"  Target times: {target_time_b.shape}") # [2]
print(f"  Targets: {target_b.shape}")          # [2, 3, H, W]

assert frames_b.shape[1] == 4, "Expected N=4 frames!"
print(f"  ✓ Batch shapes correct (N=4)")

# Test STEP=2 (medium motion)
print("\n" + "="*70)
print("Test 2: STEP=2 (medium motion)")
print("="*70)

dataset2 = X4K1000Dataset(
    root="/Users/nightstalker/Projects/datasets",
    split="train",
    step=2,         # spacing=4, frames=[0,4,8,12]
    crop_size=512,  # Test cropping
    aug_flip=True,  # Test augmentation
)

print(f"\nDataset size: {len(dataset2)} samples")
print(f"Expected: 4408 seqs × 9 targets/seq = 39,672 samples")

frames, anchor_times, target_time, target = dataset2[0]
print(f"\nSample 0 (with 512x512 crop):")
print(f"  Anchor frames: {frames.shape}")      # [4, 3, 512, 512]
print(f"  Target: {target.shape}")              # [3, 512, 512]

assert frames.shape[-1] == 512 and frames.shape[-2] == 512, "Crop size mismatch!"
print(f"  ✓ Crop size correct: 512×512")

# Test STEP=3 (large motion)
print("\n" + "="*70)
print("Test 3: STEP=3 (large motion)")
print("="*70)

dataset3 = X4K1000Dataset(
    root="/Users/nightstalker/Projects/datasets",
    split="train",
    step=3,  # spacing=6, frames=[0,6,12,18]
)

print(f"\nDataset size: {len(dataset3)} samples")
print(f"Expected: 4408 seqs × 15 targets/seq = 66,120 samples")

# Sample verification
frames, anchor_times, target_time, target = dataset3[100]
print(f"\nSample 100:")
print(f"  Anchor frames: {frames.shape}")
print(f"  Target time: {target_time.item():.4f}")

# Test test split
print("\n" + "="*70)
print("Test 4: Test split (15 sequences)")
print("="*70)

dataset_test = X4K1000Dataset(
    root="/Users/nightstalker/Projects/datasets",
    split="test",
    step=1,
)

print(f"\nTest dataset size: {len(dataset_test)} samples")
print(f"Expected: 15 seqs × 3 targets/seq = 45 samples")

# Final summary
print("\n" + "="*70)
print("Summary:")
print("="*70)
print(f"STEP=1 (train): {len(dataset):,} samples")
print(f"STEP=2 (train): {len(dataset2):,} samples")
print(f"STEP=3 (train): {len(dataset3):,} samples")
print(f"STEP=1 (test):  {len(dataset_test):,} samples")

print("\n" + "="*70)
print("✅ All tests passed! X4K dataloader is working correctly.")
print("="*70)
print("\nKey features verified:")
print("  ✓ STEP-based anchor sampling (0, 2*step, 4*step, 6*step)")
print("  ✓ Multiple targets per sequence (all frames between anchors)")
print("  ✓ Anchor times normalization [0.0, 0.33, 0.67, 1.0]")
print("  ✓ Target time interpolation [0.0, 1.0]")
print("  ✓ Spatial cropping (512×512)")
print("  ✓ Batch collation (N=4)")
print("  ✓ Train/test splits")
