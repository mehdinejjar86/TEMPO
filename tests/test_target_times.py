"""Verify target time computation for different STEP values"""
import torch
from data.data_x4k1000 import X4K1000Dataset

print("\n" + "="*70)
print("Target Time Verification")
print("="*70)

# Test STEP=1
print("\n" + "="*70)
print("STEP=1: Anchors [0, 2, 4, 6]")
print("="*70)

dataset1 = X4K1000Dataset(
    root="/Users/nightstalker/Projects/datasets",
    split="train",
    step=1,
    crop_size=None,
    aug_flip=False,
)

print(f"\nTotal samples: {len(dataset1)}")
print("\nTarget time for each possible target frame:")
print("(Remember: target time should be BETWEEN 0.0 and 1.0, not AT the endpoints)")
print()

# Get samples from first sequence (they should have different target frames)
for i in range(min(3, len(dataset1))):
    frames, anchor_times, target_time, target = dataset1[i]
    seq_idx, target_frame, anchors = dataset1.samples[i]

    # Manually compute expected target time
    expected_time = (target_frame - anchors[0]) / (anchors[-1] - anchors[0])

    print(f"Sample {i}:")
    print(f"  Anchors: {anchors}")
    print(f"  Target frame: {target_frame}")
    print(f"  Target time: {target_time.item():.4f}")
    print(f"  Expected: ({target_frame}-{anchors[0]})/({anchors[-1]}-{anchors[0]}) = {expected_time:.4f}")
    print(f"  ✓ Match!" if abs(target_time.item() - expected_time) < 1e-6 else f"  ✗ Mismatch!")
    print()

# Test STEP=2
print("="*70)
print("STEP=2: Anchors [0, 4, 8, 12]")
print("="*70)

dataset2 = X4K1000Dataset(
    root="/Users/nightstalker/Projects/datasets",
    split="train",
    step=2,
    crop_size=None,
    aug_flip=False,
)

print(f"\nTotal samples: {len(dataset2)}")
print("\nShowing first 9 samples (all from same sequence, different targets):")
print()

for i in range(min(9, len(dataset2))):
    frames, anchor_times, target_time, target = dataset2[i]
    seq_idx, target_frame, anchors = dataset2.samples[i]

    expected_time = (target_frame - anchors[0]) / (anchors[-1] - anchors[0])

    print(f"Sample {i}: Target frame {target_frame} → time {target_time.item():.4f} (expected {expected_time:.4f})")

# Test STEP=3
print("\n" + "="*70)
print("STEP=3: Anchors [0, 6, 12, 18]")
print("="*70)

dataset3 = X4K1000Dataset(
    root="/Users/nightstalker/Projects/datasets",
    split="train",
    step=3,
    crop_size=None,
    aug_flip=False,
)

print(f"\nTotal samples: {len(dataset3)}")
print("\nShowing first 15 samples (all from same sequence):")
print()

for i in range(min(15, len(dataset3))):
    frames, anchor_times, target_time, target = dataset3[i]
    seq_idx, target_frame, anchors = dataset3.samples[i]

    expected_time = (target_frame - anchors[0]) / (anchors[-1] - anchors[0])

    print(f"Sample {i:2d}: Target frame {target_frame:2d} → time {target_time.item():.4f} (expected {expected_time:.4f})")

# Summary
print("\n" + "="*70)
print("Summary: Target Time Distribution")
print("="*70)

for step_num, dataset, step_name in [(1, dataset1, "STEP=1"), (2, dataset2, "STEP=2"), (3, dataset3, "STEP=3")]:
    target_times = []
    for i in range(min(100, len(dataset))):
        _, _, target_time, _ = dataset[i]
        target_times.append(target_time.item())

    print(f"\n{step_name}:")
    print(f"  Min target_time: {min(target_times):.4f}")
    print(f"  Max target_time: {max(target_times):.4f}")
    print(f"  Range: ({min(target_times):.4f}, {max(target_times):.4f})")

    # Check if any are exactly 0.0 or 1.0 (should not be!)
    at_endpoints = [t for t in target_times if t == 0.0 or t == 1.0]
    if at_endpoints:
        print(f"  ⚠️  WARNING: Found {len(at_endpoints)} samples at endpoints 0.0 or 1.0!")
    else:
        print(f"  ✓ All target times strictly between 0.0 and 1.0 (interpolation only)")

print("\n" + "="*70)
print("✅ Target time verification complete!")
print("="*70)
