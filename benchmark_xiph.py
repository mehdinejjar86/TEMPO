"""
Xiph/Netflix 4K Benchmark for TEMPO

Evaluates frame interpolation on 8 Xiph 4K sequences (4096x2160).
Uses N=4 anchor frames with tiled inference for memory efficiency.

Setup:
    Run xiph.py first to download the dataset to ./netflix/

Usage:
    python benchmark_xiph.py --checkpoint path/to/checkpoint.pt
    python benchmark_xiph.py --checkpoint ckpt.pt --save_frames ./output/xiph

Frame arrangement:
    - 100 frames per sequence (001.png to 100.png)
    - Odd frames (1, 3, 5, ..., 99) are anchors
    - Even frames (2, 4, 6, ..., 98) are targets to interpolate
    - For each target t, use anchors: t-3, t-1, t+1, t+3
"""
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import json

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from model.tempo import build_tempo
from utils.tiling import infer_with_tiling
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


# Xiph/Netflix sequences
SEQUENCES = [
    "BoxingPractice",
    "Crosswalk",
    "DrivingPOV",
    "FoodMarket",
    "FoodMarket2",
    "RitualDance",
    "SquareAndTimelapse",
    "Tango",
]


def load_frame(path: Path) -> torch.Tensor:
    """Load a single frame as [3, H, W] tensor in [0, 1] range."""
    img = Image.open(path).convert("RGB")
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def save_frame(tensor: torch.Tensor, path: Path):
    """Save a [3, H, W] tensor as PNG image."""
    img = (tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(path)


def get_frame_path(data_dir: Path, seq_name: str, frame_idx: int) -> Path:
    """Get path to a specific frame (1-indexed)."""
    return data_dir / f"{seq_name}-{frame_idx:03d}.png"


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between two images."""
    mse = F.mse_loss(pred, target)
    if mse < 1e-10:
        return 100.0
    return -10 * torch.log10(mse).item()


def build_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Build and load TEMPO model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Build model with default config
    model = build_tempo(
        base_channels=64,
        temporal_channels=64,
        encoder_depths=[3, 3, 12, 3],
        decoder_depths=[3, 3, 3, 3],
        num_heads=8,
        num_points=4,
        use_cross_scale=True,
    ).to(device)

    # Handle state dict prefixes (DDP module., torch.compile _orig_mod.)
    state_dict = checkpoint['model_state']

    # Remove module. prefix if present
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Remove _orig_mod. prefix if present
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    print(f"  Loaded from step {checkpoint.get('step', 'unknown')}")
    if 'best_metric' in checkpoint:
        print(f"  Best validation PSNR: {checkpoint['best_metric']:.2f} dB")

    return model


def get_anchor_indices(target_frame: int, n_frames: int = 4) -> List[int]:
    """
    Get anchor frame indices for a target frame.

    For target t (even), selects 4 consecutive odd frames that surround t.
    Uses a sliding window approach to handle boundaries properly.

    Args:
        target_frame: Target frame index (even, 2-98)
        n_frames: Number of anchor frames (default: 4)

    Returns:
        List of 4 distinct anchor frame indices (odd frames)

    Examples:
        target=2  -> anchors=[1, 3, 5, 7]   (shifted right, target at 1/6)
        target=4  -> anchors=[1, 3, 5, 7]   (shifted right, target at 3/6)
        target=50 -> anchors=[47, 49, 51, 53] (centered, target at 3/6)
        target=96 -> anchors=[93, 95, 97, 99] (shifted left, target at 3/6)
        target=98 -> anchors=[93, 95, 97, 99] (shifted left, target at 5/6)
    """
    # All odd frames available: 1, 3, 5, ..., 99
    # For N=4, we need 4 consecutive odd frames spanning 6 frame indices
    # (e.g., [1,3,5,7] spans indices 1-7)

    # Ideal center position: target should be between anchor[1] and anchor[2]
    # So first anchor ideally at target - 3
    ideal_first = target_frame - 3

    # Clamp to valid range: first anchor in [1, 93] to fit 4 odd frames
    # (93, 95, 97, 99 is the last valid window)
    first_anchor = max(1, min(93, ideal_first))

    # Ensure first_anchor is odd
    if first_anchor % 2 == 0:
        first_anchor -= 1

    # Generate 4 consecutive odd frames
    anchors = [first_anchor + 2 * i for i in range(n_frames)]

    return anchors


def benchmark_sequence(
    model: torch.nn.Module,
    data_dir: Path,
    seq_name: str,
    device: torch.device,
    ssim_metric: SSIM,
    tile_size: int = 512,
    overlap: int = 64,
    save_dir: Path = None,
) -> Dict[str, float]:
    """
    Benchmark a single sequence.

    Args:
        save_dir: If provided, save predicted frames to this directory

    Returns:
        Dict with 'psnr' and 'ssim' metrics
    """
    psnr_values = []
    ssim_values = []

    # Create save directory for this sequence
    if save_dir is not None:
        seq_save_dir = save_dir / seq_name
        seq_save_dir.mkdir(parents=True, exist_ok=True)

    # Target frames: 2, 4, 6, ..., 98 (even frames)
    target_frames = list(range(2, 99, 2))  # 2 to 98 inclusive

    pbar = tqdm(target_frames, desc=f"  {seq_name}", leave=False, ncols=100)

    for target_idx in pbar:
        # Get anchor indices
        anchor_indices = get_anchor_indices(target_idx)

        # Load anchor frames
        anchor_frames = []
        for idx in anchor_indices:
            frame_path = get_frame_path(data_dir, seq_name, idx)
            if not frame_path.exists():
                raise FileNotFoundError(f"Frame not found: {frame_path}")
            anchor_frames.append(load_frame(frame_path))

        # Load target frame (ground truth)
        target_path = get_frame_path(data_dir, seq_name, target_idx)
        target = load_frame(target_path).to(device)

        # Stack anchors: [N, 3, H, W]
        frames = torch.stack(anchor_frames, dim=0).unsqueeze(0).to(device)  # [1, N, 3, H, W]

        # Compute normalized times
        # Anchor times based on their positions relative to the window
        min_anchor = min(anchor_indices)
        max_anchor = max(anchor_indices)
        anchor_times = torch.tensor(
            [(idx - min_anchor) / (max_anchor - min_anchor) for idx in anchor_indices],
            dtype=torch.float32, device=device
        ).unsqueeze(0)  # [1, N]

        # Target time (normalized position within anchor window)
        target_time = torch.tensor(
            [(target_idx - min_anchor) / (max_anchor - min_anchor)],
            dtype=torch.float32, device=device
        )  # [1]

        # Run inference with tiling
        with torch.no_grad():
            pred = infer_with_tiling(
                model, frames, anchor_times, target_time,
                tile_size=tile_size, overlap=overlap
            )

        pred = pred.squeeze(0).clamp(0, 1)  # [3, H, W]

        # Compute metrics
        psnr = compute_psnr(pred, target)
        psnr_values.append(psnr)

        ssim_metric.reset()
        ssim_val = ssim_metric(pred.unsqueeze(0), target.unsqueeze(0)).item()
        ssim_values.append(ssim_val)

        # Save predicted frame
        if save_dir is not None:
            save_path = seq_save_dir / f"{target_idx:03d}.png"
            save_frame(pred, save_path)

        pbar.set_postfix({'psnr': f'{psnr:.2f}', 'ssim': f'{ssim_val:.4f}'})

    return {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'psnr_std': np.std(psnr_values),
        'ssim_std': np.std(ssim_values),
        'n_frames': len(target_frames),
    }


def main():
    parser = argparse.ArgumentParser(description="Xiph 4K Benchmark for TEMPO")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="./netflix",
                        help="Path to Xiph/Netflix frames (default: ./netflix)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    parser.add_argument("--tile_size", type=int, default=512,
                        help="Tile size for tiled inference (default: 512)")
    parser.add_argument("--overlap", type=int, default=64,
                        help="Overlap between tiles (default: 64)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--save_frames", type=str, default=None,
                        help="Directory to save predicted frames (default: None)")
    parser.add_argument("--sequences", type=str, nargs='+', default=None,
                        help="Specific sequences to benchmark (default: all)")
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Run xiph.py first to download the dataset.")
        return

    # Check sequences exist
    sequences = args.sequences if args.sequences else SEQUENCES
    for seq in sequences:
        first_frame = get_frame_path(data_dir, seq, 1)
        if not first_frame.exists():
            print(f"Sequence '{seq}' not found. Run xiph.py to download.")
            return

    print(f"\n{'='*60}")
    print("Xiph 4K Benchmark")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Sequences: {len(sequences)}")
    print(f"Tile size: {args.tile_size}, Overlap: {args.overlap}")
    if args.save_frames:
        print(f"Saving frames to: {args.save_frames}")
    print(f"{'='*60}\n")

    # Setup save directory
    save_dir = Path(args.save_frames) if args.save_frames else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = build_model(args.checkpoint, device)

    # SSIM metric
    ssim_metric = SSIM(data_range=1.0).to(device)

    # Benchmark each sequence
    results = {}
    all_psnr = []
    all_ssim = []

    print("\nBenchmarking sequences:")
    for seq_name in sequences:
        print(f"\n{seq_name}:")

        seq_results = benchmark_sequence(
            model, data_dir, seq_name, device, ssim_metric,
            tile_size=args.tile_size, overlap=args.overlap,
            save_dir=save_dir
        )

        results[seq_name] = seq_results
        all_psnr.append(seq_results['psnr'])
        all_ssim.append(seq_results['ssim'])

        print(f"  PSNR: {seq_results['psnr']:.2f} dB (std: {seq_results['psnr_std']:.2f})")
        print(f"  SSIM: {seq_results['ssim']:.4f} (std: {seq_results['ssim_std']:.4f})")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Average PSNR: {np.mean(all_psnr):.2f} dB")
    print(f"Average SSIM: {np.mean(all_ssim):.4f}")
    print(f"{'='*60}\n")

    # Per-sequence table
    print("Per-sequence results:")
    print(f"{'Sequence':<25} {'PSNR (dB)':>12} {'SSIM':>12}")
    print("-" * 50)
    for seq_name in sequences:
        r = results[seq_name]
        print(f"{seq_name:<25} {r['psnr']:>12.2f} {r['ssim']:>12.4f}")
    print("-" * 50)
    print(f"{'Average':<25} {np.mean(all_psnr):>12.2f} {np.mean(all_ssim):>12.4f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        results['_summary'] = {
            'avg_psnr': float(np.mean(all_psnr)),
            'avg_ssim': float(np.mean(all_ssim)),
            'checkpoint': args.checkpoint,
            'n_sequences': len(sequences),
        }
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
