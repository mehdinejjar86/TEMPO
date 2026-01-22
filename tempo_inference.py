#!/usr/bin/env python3
"""
TEMPO Evaluation Script - VFIMamba Exact Match
==============================================

This script evaluates TEMPO using the EXACT same methodology as VFIMamba
to ensure fair comparison and identify any remaining discrepancies.

Key Features:
- Matches VFIMamba's PSNR/SSIM calculation exactly
- Supports test-time augmentation (horizontal flip)
- Handles both Vimeo-90K and X4K datasets
- Provides detailed per-sample statistics
- Saves predictions for visual inspection

Usage:
    # Basic evaluation
    python tempo_eval_vfimamba_match.py \
        --model_path runs/best_model.pth \
        --data_path datasets/vimeo_triplet

    # With test-time augmentation (TTA)
    python tempo_eval_vfimamba_match.py \
        --model_path runs/best_model.pth \
        --data_path datasets/vimeo_triplet \
        --use_tta

    # Save predictions for inspection
    python tempo_eval_vfimamba_match.py \
        --model_path runs/best_model.pth \
        --data_path datasets/vimeo_triplet \
        --save_predictions output/preds
"""

import cv2
import math
import sys
import torch
import numpy as np
import argparse
import os
import warnings
from pathlib import Path
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

sys.path.append('.')
from model.tempo import build_tempo

# Import VFIMamba's SSIM implementation for exact match
VFIMAMBA_ROOT = "/home/groups/ChangLab/govindsa/confocal_project/datasets/benchmarking/VFIMamba"
sys.path.insert(0, VFIMAMBA_ROOT)
from benchmark.utils.pytorch_msssim import ssim_matlab


def parse_args():
    parser = argparse.ArgumentParser(description="TEMPO Evaluation - VFIMamba Match")
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to TEMPO checkpoint file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset (should contain tri_testlist.txt and sequences/)')
    
    # Model architecture (should match training config)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--temporal_channels', type=int, default=64)
    parser.add_argument('--encoder_depths', type=int, nargs='+', default=[3, 3, 12, 3])
    parser.add_argument('--decoder_depths', type=int, nargs='+', default=[3, 3, 3, 3])
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_points', type=int, default=4)
    parser.add_argument('--use_cross_scale', action='store_true', default=True)
    
    # Evaluation settings
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--use_tta', action='store_true',
                       help='Use test-time augmentation (horizontal flip averaging)')
    parser.add_argument('--save_predictions', type=str, default=None,
                       help='Directory to save predictions (for visual inspection)')
    parser.add_argument('--save_every_n', type=int, default=100,
                       help='Save every Nth prediction (to avoid filling disk)')
    
    # Analysis options
    parser.add_argument('--verbose', action='store_true',
                       help='Print per-sample metrics')
    parser.add_argument('--save_stats', type=str, default=None,
                       help='Save detailed statistics to JSON file')
    
    return parser.parse_args()


class VFIMambaMetrics:
    """
    Metric calculation that EXACTLY matches VFIMamba's implementation.
    
    Key differences from naive implementations:
    1. PSNR: Pure MSE-based, no epsilon added, no clamping before calculation
    2. SSIM: Uses ssim_matlab from VFIMamba (handles edge cases differently)
    3. Both metrics calculated on UNCLAMPED predictions (only GT is in [0,1])
    """
    
    @staticmethod
    def calculate_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
        """
        Calculate PSNR exactly as VFIMamba does.
        
        Args:
            pred: [3, H, W] prediction tensor (unclamped, may have values outside [0,1])
            gt: [3, H, W] ground truth tensor (in [0,1])
            
        Returns:
            PSNR in dB (can be negative if prediction is very bad)
        """
        # Convert to numpy for exact match with VFIMamba
        pred_np = pred.detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        gt_np = gt.detach().cpu().numpy().transpose(1, 2, 0)      # [H, W, 3]
        
        # CRITICAL: No clamping on prediction
        # VFIMamba calculates MSE on raw predictions
        mse = ((pred_np - gt_np) ** 2).mean()
        
        # Handle edge case: identical images
        if mse == 0:
            return 100.0  # Perfect match
        
        # Standard PSNR formula: -10 * log10(MSE)
        # Note: Can be negative if MSE > 1.0 (very bad predictions)
        psnr = -10 * math.log10(mse)
        
        return psnr
    
    @staticmethod
    def calculate_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
        """
        Calculate SSIM exactly as VFIMamba does.
        
        Args:
            pred: [3, H, W] prediction tensor (unclamped)
            gt: [3, H, W] ground truth tensor (in [0,1])
            
        Returns:
            SSIM score (typically in [0,1], but can be negative for very bad predictions)
        """
        # Add batch dimension for ssim_matlab: [1, 3, H, W]
        pred_batch = pred.unsqueeze(0)
        gt_batch = gt.unsqueeze(0)
        
        # CRITICAL: No clamping before SSIM calculation
        # VFIMamba's ssim_matlab handles unclamped inputs
        ssim_score = ssim_matlab(gt_batch, pred_batch).detach().cpu().numpy()
        
        return float(ssim_score)


class TempoEvaluator:
    """Evaluator that matches VFIMamba's methodology exactly."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.metrics = VFIMambaMetrics()
        
        # Build and load model
        self.model = self._load_model()
        
        # Setup output directory if saving predictions
        if args.save_predictions:
            self.pred_dir = Path(args.save_predictions)
            self.pred_dir.mkdir(parents=True, exist_ok=True)
            print(f"💾 Predictions will be saved to: {self.pred_dir}")
        else:
            self.pred_dir = None
    
    def _load_model(self):
        """Load TEMPO model from checkpoint."""
        print("🏗️ Building TEMPO model...")
        model = build_tempo(
            base_channels=self.args.base_channels,
            temporal_channels=self.args.temporal_channels,
            encoder_depths=self.args.encoder_depths,
            decoder_depths=self.args.decoder_depths,
            num_heads=self.args.num_heads,
            num_points=self.args.num_points,
            use_cross_scale=self.args.use_cross_scale,
            use_checkpointing=False,
        ).to(self.device)
        
        print(f"📂 Loading checkpoint: {self.args.model_path}")
        checkpoint = torch.load(self.args.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        state_dict = checkpoint.get("model_state", checkpoint)
        
        # Remove DDP/compile prefixes
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        if any("_orig_mod" in k for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️  Missing keys: {missing}")
        if unexpected:
            print(f"⚠️  Unexpected keys: {unexpected}")
        
        model.eval()
        print(f"✅ Model loaded successfully")
        
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"📊 Model: {total_params:.2f}M parameters")
        
        return model
    
    def _infer_single(self, frames: torch.Tensor, anchor_times: torch.Tensor, 
                     target_time: torch.Tensor) -> torch.Tensor:
        """
        Run inference on a single sample.
        
        Args:
            frames: [1, 2, 3, H, W] anchor frames
            anchor_times: [1, 2] anchor timestamps
            target_time: [1] target timestamp
            
        Returns:
            pred: [3, H, W] predicted frame (UNCLAMPED)
        """
        with torch.no_grad():
            pred, aux = self.model(frames, anchor_times, target_time)
        
        # Return unclamped prediction (match VFIMamba evaluation)
        return pred[0]  # [3, H, W]
    
    def _infer_with_tta(self, frames: torch.Tensor, anchor_times: torch.Tensor,
                       target_time: torch.Tensor) -> torch.Tensor:
        """
        Run inference with test-time augmentation (horizontal flip averaging).
        
        Args:
            frames: [1, 2, 3, H, W] anchor frames
            anchor_times: [1, 2] anchor timestamps
            target_time: [1] target timestamp
            
        Returns:
            pred: [3, H, W] averaged prediction (UNCLAMPED)
        """
        # Original prediction
        pred_orig = self._infer_single(frames, anchor_times, target_time)
        
        # Flipped prediction
        frames_flip = torch.flip(frames, dims=[4])  # Flip width dimension
        pred_flip = self._infer_single(frames_flip, anchor_times, target_time)
        pred_flip = torch.flip(pred_flip, dims=[2])  # Flip back
        
        # Average (unclamped)
        pred_avg = (pred_orig + pred_flip) / 2.0
        
        return pred_avg
    
    def _save_prediction(self, pred: torch.Tensor, gt: torch.Tensor, 
                        seq_name: str, idx: int):
        """Save prediction for visual inspection."""
        if self.pred_dir is None:
            return
        
        # Only save every Nth sample
        if idx % self.args.save_every_n != 0:
            return
        
        # Clamp for visualization only
        pred_vis = pred.clamp(0, 1)
        
        # Convert to numpy [H, W, 3]
        pred_np = (pred_vis * 255).byte().permute(1, 2, 0).cpu().numpy()
        gt_np = (gt * 255).byte().permute(1, 2, 0).cpu().numpy()
        
        # Create comparison image
        comparison = np.hstack([gt_np, pred_np])
        
        # Save
        save_path = self.pred_dir / f"{seq_name.replace('/', '_')}_{idx:04d}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    def evaluate_vimeo(self):
        """
        Evaluate on Vimeo-90K dataset using VFIMamba's exact methodology.
        
        Returns:
            results: Dict with avg_psnr, avg_ssim, and detailed statistics
        """
        path = self.args.data_path
        tri_testlist = os.path.join(path, 'tri_testlist.txt')
        
        if not os.path.exists(tri_testlist):
            raise FileNotFoundError(f"tri_testlist.txt not found at {tri_testlist}")
        
        with open(tri_testlist, 'r') as f:
            test_sequences = [line.strip() for line in f if line.strip()]
        
        print(f"\n{'='*80}")
        print(f"TEMPO Evaluation - VFIMamba Match Mode")
        print(f"{'='*80}")
        print(f"Dataset: {path}")
        print(f"Sequences: {len(test_sequences)}")
        print(f"Test-time augmentation: {'ON' if self.args.use_tta else 'OFF'}")
        print(f"{'='*80}\n")
        
        psnr_list = []
        ssim_list = []
        sample_stats = []
        
        for idx, seq_name in enumerate(tqdm(test_sequences, desc="Evaluating")):
            if len(seq_name) <= 1:
                continue
            
            # Load images
            im1_path = os.path.join(path, 'sequences', seq_name, 'im1.png')
            im2_path = os.path.join(path, 'sequences', seq_name, 'im2.png')
            im3_path = os.path.join(path, 'sequences', seq_name, 'im3.png')
            
            if not all(os.path.exists(p) for p in [im1_path, im2_path, im3_path]):
                print(f"⚠️  Skipping {seq_name} - missing files")
                continue
            
            I0 = cv2.imread(im1_path)  # First frame
            I1 = cv2.imread(im2_path)  # Ground truth (middle)
            I2 = cv2.imread(im3_path)  # Last frame
            
            if I0 is None or I1 is None or I2 is None:
                print(f"⚠️  Skipping {seq_name} - failed to load images")
                continue
            
            # Convert to tensors [1, 2, 3, H, W]
            frames = torch.stack([
                torch.tensor(I0.transpose(2, 0, 1)).float() / 255.0,
                torch.tensor(I2.transpose(2, 0, 1)).float() / 255.0,
            ], dim=0).unsqueeze(0).to(self.device)
            
            anchor_times = torch.tensor([[0.0, 1.0]]).to(self.device)
            target_time = torch.tensor([0.5]).to(self.device)
            
            # Run inference (with or without TTA)
            if self.args.use_tta:
                pred = self._infer_with_tta(frames, anchor_times, target_time)
            else:
                pred = self._infer_single(frames, anchor_times, target_time)
            
            # Ground truth tensor [3, H, W] in [0, 1]
            gt = torch.tensor(I1.transpose(2, 0, 1)).float().to(self.device) / 255.0
            
            # Calculate metrics (VFIMamba exact match - unclamped predictions)
            psnr = self.metrics.calculate_psnr(pred, gt)
            ssim = self.metrics.calculate_ssim(pred, gt)
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            # Store per-sample stats
            sample_stats.append({
                'sequence': seq_name,
                'psnr': float(psnr),
                'ssim': float(ssim)
            })
            
            # Verbose output
            if self.args.verbose and idx % 100 == 0:
                print(f"\n[{idx}/{len(test_sequences)}] {seq_name}")
                print(f"  PSNR: {psnr:.4f} dB, SSIM: {ssim:.4f}")
                print(f"  Running avg - PSNR: {np.mean(psnr_list):.4f}, SSIM: {np.mean(ssim_list):.4f}")
            
            # Save predictions
            self._save_prediction(pred, gt, seq_name, idx)
        
        # Final statistics
        results = {
            'avg_psnr': float(np.mean(psnr_list)),
            'avg_ssim': float(np.mean(ssim_list)),
            'std_psnr': float(np.std(psnr_list)),
            'std_ssim': float(np.std(ssim_list)),
            'min_psnr': float(np.min(psnr_list)),
            'max_psnr': float(np.max(psnr_list)),
            'min_ssim': float(np.min(ssim_list)),
            'max_ssim': float(np.max(ssim_list)),
            'num_samples': len(psnr_list),
            'use_tta': self.args.use_tta,
            'sample_stats': sample_stats
        }
        
        # Save detailed statistics if requested
        if self.args.save_stats:
            stats_path = Path(self.args.save_stats)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Detailed statistics saved to: {stats_path}")
        
        return results


def print_results(results: dict):
    """Pretty print evaluation results."""
    print(f"\n{'='*80}")
    print("FINAL RESULTS (VFIMamba Match Mode)")
    print(f"{'='*80}")
    print(f"Total sequences processed: {results['num_samples']}")
    print(f"Test-time augmentation: {'ON' if results['use_tta'] else 'OFF'}")
    print(f"\nAverage Metrics:")
    print(f"  PSNR: {results['avg_psnr']:.4f} ± {results['std_psnr']:.4f} dB")
    print(f"  SSIM: {results['avg_ssim']:.6f} ± {results['std_ssim']:.6f}")
    print(f"\nRange:")
    print(f"  PSNR: [{results['min_psnr']:.4f}, {results['max_psnr']:.4f}] dB")
    print(f"  SSIM: [{results['min_ssim']:.6f}, {results['max_ssim']:.6f}]")
    print(f"{'='*80}\n")
    
    # Find worst-performing samples
    sample_stats = results['sample_stats']
    worst_psnr = sorted(sample_stats, key=lambda x: x['psnr'])[:5]
    worst_ssim = sorted(sample_stats, key=lambda x: x['ssim'])[:5]
    
    print("Worst PSNR samples:")
    for i, s in enumerate(worst_psnr, 1):
        print(f"  {i}. {s['sequence']}: {s['psnr']:.4f} dB (SSIM: {s['ssim']:.4f})")
    
    print("\nWorst SSIM samples:")
    for i, s in enumerate(worst_ssim, 1):
        print(f"  {i}. {s['sequence']}: {s['ssim']:.4f} (PSNR: {s['psnr']:.4f} dB)")
    
    print()


def main():
    args = parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")
    
    # Run evaluation
    evaluator = TempoEvaluator(args)
    results = evaluator.evaluate_vimeo()
    
    # Print results
    print_results(results)
    
    # Comparison hints
    print("💡 Comparison Tips:")
    print("  1. Check if VFIMamba uses TTA (try --use_tta)")
    print("  2. Verify VFIMamba's reported metrics are on the SAME test set")
    print("  3. Check for any preprocessing differences (resize, normalization)")
    print("  4. Inspect worst-performing samples for patterns")
    print("  5. If still discrepant, save predictions and compare pixel-by-pixel\n")


if __name__ == "__main__":
    main()