"""
X4K1000 Dataset Loader with STEP-based Sampling

STEP-based sampling allows systematic control of motion magnitude:
  - step=1: spacing=2 (frames 0,2,4,6)     → small motion
  - step=2: spacing=4 (frames 0,4,8,12)    → medium motion
  - step=3: spacing=6 (frames 0,6,12,18)   → large motion

Each 65-frame sequence generates multiple training samples (all frames between
anchors become targets). Number of targets depends on both STEP and n_frames.
Example with default n_frames=4: step=1→3 targets, step=2→9 targets, step=3→15 targets.
"""

import random
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class X4K1000Dataset(data.Dataset):
    """
    X4K1000 dataset loader with STEP-based N=4 frame sampling.

    Dataset structure:
        root/split/parent_dir/seq_name/0000.png ... 0064.png

    Example:
        /Users/nightstalker/Projects/datasets/train/002/occ008.320_f2881/0000.png

    STEP controls motion magnitude:
      - step=1: spacing=2 (frames 0,2,4,6)     → small motion
      - step=2: spacing=4 (frames 0,4,8,12)    → medium motion
      - step=3: spacing=6 (frames 0,6,12,18)   → large motion

    Each sequence (65 frames) generates multiple samples (all valid targets).

    Args:
        root: Dataset root (default: /Users/nightstalker/Projects/datasets)
        split: 'train' or 'test'
        step: Frame spacing parameter (1, 2, 3, ...)
        crop_size: Spatial crop for 4K (512 or 768), None for no crop
        aug_flip: Horizontal flip augmentation
        n_frames: Number of anchor frames (default: 4)
    """

    def __init__(
        self,
        root: str = "/Users/nightstalker/Projects/datasets",
        split: str = "train",
        step: int = 1,
        crop_size: Optional[int] = None,
        aug_flip: bool = False,
        n_frames: int = 4,
    ):
        self.root = Path(root)
        self.split = split
        self.step = step
        self.crop_size = crop_size
        self.aug_flip = aug_flip
        self.n_frames = n_frames

        # Scan all sequences
        self.sequences = self._scan_sequences()
        print(f"Found {len(self.sequences)} sequences in {split} split")

        # Generate (sequence_idx, target_idx) pairs for all valid samples
        self.samples = self._generate_samples()
        print(f"Generated {len(self.samples)} samples with step={step}")

    def _scan_sequences(self) -> List[str]:
        """Scan nested directory structure for 65-frame sequences"""
        sequences = []
        split_dir = self.root / self.split  # /datasets/train or /datasets/test

        # Iterate through parent dirs (002, 003, ...)
        for parent_dir in sorted(split_dir.iterdir()):
            if not parent_dir.is_dir():
                continue

            # Iterate through sequence dirs (occ*.*)
            for seq_dir in sorted(parent_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue

                # Verify 65 frames
                frames = sorted(seq_dir.glob("*.png"))
                if len(frames) == 65:
                    # Store relative path from split_dir
                    rel_path = seq_dir.relative_to(split_dir)
                    sequences.append(str(rel_path))  # e.g., "002/occ008.320_f2881"

        return sequences

    def _generate_samples(self) -> List[Tuple[int, int, List[int]]]:
        """
        Generate all (sequence, target) pairs for the given STEP.

        Each sequence can generate multiple training samples.

        Returns:
            List of (seq_idx, target_frame, anchors) tuples
        """
        spacing = 2 * self.step  # step=1→2, step=2→4, step=3→6
        anchors = [i * spacing for i in range(self.n_frames)]  # [0, spacing, 2*spacing, 3*spacing]

        # Check if anchors fit in 65 frames (indices 0-64)
        if anchors[-1] >= 65:
            raise ValueError(
                f"step={self.step} (spacing={spacing}) produces anchors={anchors}, "
                f"but sequence only has 65 frames (indices 0-64). Max step is {64 // (2*(self.n_frames-1))}"
            )

        # All target frames between first and last anchor (excluding anchors)
        valid_targets = []
        for i in range(anchors[0] + 1, anchors[-1]):
            if i not in anchors:
                valid_targets.append(i)

        # Generate all (seq_idx, target_frame) pairs
        samples = []
        for seq_idx in range(len(self.sequences)):
            for target_frame in valid_targets:
                samples.append((seq_idx, target_frame, anchors))

        return samples  # [(seq_idx, target_frame, [anchor frames]), ...]

    def _load_frames(self, seq_path: str, frame_indices: List[int]) -> List[torch.Tensor]:
        """Load frames at specified indices"""
        frames = []
        seq_dir = self.root / self.split / seq_path

        for idx in frame_indices:
            frame_path = seq_dir / f"{idx:04d}.png"
            img = Image.open(frame_path).convert("RGB")
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            frames.append(img)

        return frames  # List of [3,H,W]

    def _random_crop_all(self, frames: List[torch.Tensor], crop_size: Optional[int]) -> List[torch.Tensor]:
        """Apply same random crop to all frames"""
        if not frames:
            return frames

        _, H, W = frames[0].shape

        if crop_size is not None and (H > crop_size or W > crop_size):
            # Random crop parameters
            top = random.randint(0, max(0, H - crop_size))
            left = random.randint(0, max(0, W - crop_size))

            # Apply to all frames
            frames = [f[:, top:top+crop_size, left:left+crop_size] for f in frames]

        return frames

    def _flip_all(self, frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply same horizontal flip to all frames"""
        if self.aug_flip and random.random() < 0.5:
            frames = [torch.flip(f, dims=[2]) for f in frames]
        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            frames: [N,3,H,W] anchor frames
            anchor_times: [N] normalized times [0.0, 0.33, 0.67, 1.0]
            target_time: scalar, normalized position of target
            target: [3,H,W] target frame
        """
        seq_idx, target_frame, anchors = self.samples[idx]
        seq_path = self.sequences[seq_idx]

        # Load anchors + target
        all_indices = anchors + [target_frame]
        all_frames = self._load_frames(seq_path, all_indices)

        # Augmentations (consistent across all frames)
        all_frames = self._random_crop_all(all_frames, self.crop_size)
        all_frames = self._flip_all(all_frames)

        # Split anchors and target
        anchor_frames = all_frames[:self.n_frames]  # First N frames
        target_frame_data = all_frames[-1]           # Last frame

        # Stack anchors into [N,3,H,W]
        anchor_frames = torch.stack(anchor_frames, dim=0)

        # Compute normalized times
        anchor_times = torch.linspace(0.0, 1.0, self.n_frames)  # [0.0, 0.33, 0.67, 1.0]

        # Target time (interpolation between anchors)
        # Map target_frame to [0, 1] based on its position between first and last anchor
        target_time = (target_frame - anchors[0]) / (anchors[-1] - anchors[0])
        target_time = torch.tensor(target_time, dtype=torch.float32)

        return anchor_frames, anchor_times, target_time, target_frame_data


def x4k_collate(batch):
    """
    Collate for X4K batches (N=4).

    Input: List of tuples (frames[N,3,H,W], times[N], target_time, target[3,H,W])
    Output: Batched tensors
    """
    frames       = torch.stack([b[0] for b in batch], dim=0)  # [B,N,3,H,W]
    anchor_times = torch.stack([b[1] for b in batch], dim=0)  # [B,N]
    target_time  = torch.stack([b[2] for b in batch], dim=0)  # [B]
    target       = torch.stack([b[3] for b in batch], dim=0)  # [B,3,H,W]

    return frames, anchor_times, target_time, target
