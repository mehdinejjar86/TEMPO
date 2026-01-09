"""
X4K1000 Test Dataset Loader

Handles X4K test sequences (33 frames, 0000.png to 0032.png) with STEP=1 sampling
for consistent validation.

Directory structure:
    root/test/Type1/TEST01_003_f0433/0000.png ... 0032.png
"""
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class X4K1000TestDataset(data.Dataset):
    """
    X4K test dataset (33-frame sequences, full 4K resolution).

    Test structure differs from training:
      - Training: 65 frames in root/train/002/occ008.320_f2881/
      - Test: 33 frames in root/test/Type1/TEST01_003_f0433/

    Validation uses STEP=1 for consistent evaluation:
      - Anchors: [0, 2, 4, 6] (N=4, evenly spaced)
      - Targets: [1, 3, 5] (frames between anchors)
      - Each sequence generates 3 validation samples

    Args:
        root: Dataset root (contains test/ directory)
        step: Frame spacing (default: 1, recommended for validation)
        n_frames: Number of anchor frames (default: 4)

    Returns:
        frames: [N, 3, H, W] anchor frames (full 4K, no crops)
        anchor_times: [N] normalized timestamps
        target_time: scalar, normalized target time
        target: [3, H, W] ground truth target frame
    """

    def __init__(
        self,
        root: str = "/Users/nightstalker/Projects/datasets",
        step: int = 1,
        n_frames: int = 4,
    ):
        self.root = Path(root)
        self.step = step
        self.n_frames = n_frames

        # Scan test sequences (33 frames each)
        self.sequences = self._scan_sequences()
        if len(self.sequences) == 0:
            raise RuntimeError(
                f"No 33-frame test sequences found in {self.root / 'test'}. "
                f"Expected structure: test/Type*/TEST*/*.png"
            )

        print(f"[X4K Test] Found {len(self.sequences)} sequences")

        # Generate (sequence_idx, target_frame, anchors) tuples
        self.samples = self._generate_samples()
        print(f"[X4K Test] Generated {len(self.samples)} samples with STEP={step}")

    def _scan_sequences(self) -> List[Path]:
        """
        Scan test directory for 33-frame sequences.

        Expected structure:
            root/test/Type1/TEST01_003_f0433/0000.png ... 0032.png
            root/test/Type2/TEST02_045_f0465/0000.png ... 0032.png

        Returns:
            List of sequence directory paths
        """
        sequences = []
        test_dir = self.root / "test"

        if not test_dir.exists():
            print(f"⚠️  Warning: Test directory not found: {test_dir}")
            return sequences

        # Iterate through Type* directories
        for type_dir in sorted(test_dir.iterdir()):
            if not type_dir.is_dir():
                continue

            # Iterate through TEST* sequence directories
            for seq_dir in sorted(type_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue

                # Count frames
                frames = sorted(seq_dir.glob("*.png"))
                if len(frames) == 33:
                    sequences.append(seq_dir)
                elif len(frames) > 0:
                    print(f"⚠️  Skipping {seq_dir.name}: has {len(frames)} frames (expected 33)")

        return sequences

    def _generate_samples(self) -> List[Tuple[int, int, List[int]]]:
        """
        Generate all (sequence, target, anchors) tuples for validation.

        With STEP=1 (recommended for validation):
          - spacing = 2 * 1 = 2
          - anchors = [0, 2, 4, 6]
          - valid_targets = [1, 3, 5] (frames between anchors)
          - Each sequence → 3 samples

        Returns:
            List of (seq_idx, target_frame, anchors) tuples
        """
        spacing = 2 * self.step
        anchors = [i * spacing for i in range(self.n_frames)]

        # Check if anchors fit in 33 frames (indices 0-32)
        if anchors[-1] >= 33:
            raise ValueError(
                f"STEP={self.step} (spacing={spacing}) produces anchors={anchors}, "
                f"but test sequences only have 33 frames (indices 0-32). "
                f"Max STEP for N={self.n_frames} is {32 // (2 * (self.n_frames - 1))}"
            )

        # All frames between first and last anchor (excluding anchors)
        valid_targets = []
        for i in range(anchors[0] + 1, anchors[-1]):
            if i not in anchors:
                valid_targets.append(i)

        if len(valid_targets) == 0:
            raise ValueError(
                f"No valid target frames between anchors {anchors}. "
                f"Anchors are too close together (STEP={self.step})."
            )

        # Generate all (seq_idx, target_frame, anchors) pairs
        samples = []
        for seq_idx in range(len(self.sequences)):
            for target_frame in valid_targets:
                samples.append((seq_idx, target_frame, anchors))

        return samples

    def _load_frames(self, seq_path: Path, frame_indices: List[int]) -> List[torch.Tensor]:
        """
        Load frames at specified indices from sequence.

        Args:
            seq_path: Path to sequence directory
            frame_indices: List of frame indices to load

        Returns:
            List of [3, H, W] tensors in range [0, 1]
        """
        frames = []
        for idx in frame_indices:
            frame_path = seq_path / f"{idx:04d}.png"

            if not frame_path.exists():
                raise FileNotFoundError(
                    f"Frame not found: {frame_path}\n"
                    f"Sequence: {seq_path}\n"
                    f"Expected frames 0000.png to 0032.png"
                )

            # Load and convert to tensor
            img = Image.open(frame_path).convert("RGB")
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            frames.append(img)

        return frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get validation sample.

        Returns:
            frames: [N, 3, H, W] anchor frames (full 4K resolution)
            anchor_times: [N] normalized times [0.0, 0.33, 0.67, 1.0]
            target_time: scalar, normalized position of target
            target: [3, H, W] ground truth target frame

        Example (STEP=1):
            - Anchors: [0, 2, 4, 6]
            - Target: 3
            - anchor_times: [0.0, 0.33, 0.67, 1.0]
            - target_time: 0.5 (middle)
        """
        seq_idx, target_frame, anchors = self.samples[idx]
        seq_path = self.sequences[seq_idx]

        # Load all frames (anchors + target)
        all_indices = anchors + [target_frame]
        all_frames = self._load_frames(seq_path, all_indices)

        # Split into anchors and target
        anchor_frames = all_frames[:self.n_frames]  # First N frames
        target_frame_data = all_frames[-1]           # Last frame

        # Stack anchors into [N, 3, H, W]
        anchor_frames = torch.stack(anchor_frames, dim=0)

        # Compute normalized times
        anchor_times = torch.linspace(0.0, 1.0, self.n_frames)

        # Target time (interpolation between first and last anchor)
        target_time = (target_frame - anchors[0]) / (anchors[-1] - anchors[0])
        target_time = torch.tensor(target_time, dtype=torch.float32)

        return anchor_frames, anchor_times, target_time, target_frame_data


def x4k_test_collate(batch):
    """
    Collate function for X4K test batches.

    Input: List of tuples (frames[N,3,H,W], times[N], target_time, target[3,H,W])
    Output: Batched tensors

    Note: For tiled inference, batch_size should be 1 (process one 4K image at a time).
    """
    frames       = torch.stack([b[0] for b in batch], dim=0)  # [B, N, 3, H, W]
    anchor_times = torch.stack([b[1] for b in batch], dim=0)  # [B, N]
    target_time  = torch.stack([b[2] for b in batch], dim=0)  # [B]
    target       = torch.stack([b[3] for b in batch], dim=0)  # [B, 3, H, W]

    return frames, anchor_times, target_time, target
