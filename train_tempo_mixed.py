"""
TEMPO Mixed Training System
============================
Temporal Multi-View Frame Synthesis with Mixed Vimeo (N=2) + X4K (N=4) Training

Features:
- Mixed-dataset training with pure batches (no N mixing within batches)
- Separate validation for Vimeo and X4K datasets
- Configurable dataset ratios
- STEP-based X4K sampling for controllable motion magnitude
- All other features from standard TEMPO training
"""
import argparse
from pathlib import Path
from typing import Dict
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler, autocast_mode

import numpy as np
from PIL import Image
from tqdm import tqdm
import os

# Optional but recommended
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ WandB not installed. Install with: pip install wandb")

# Model imports
from model.tempo import build_tempo
from model.loss.tempo_loss import build_tempo_loss, LossScheduler, MetricTracker
from data.data_vimeo_triplet import Vimeo90KTriplet, vimeo_collate
from data.data_x4k1000 import X4K1000Dataset, x4k_collate
from data.data_x4k_test import X4K1000TestDataset, x4k_test_collate
from data.samplers import PureBatchSampler, DistributedPureBatchSampler
from utils.tiling import infer_with_tiling
from config.default import TrainingConfig
from config.manager import RunManager
from config.dpp import setup_distributed_training, cleanup_distributed_training, is_main_process
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


class MixedTrainer:
    """Training orchestrator for mixed Vimeo+X4K datasets"""

    def __init__(self, config: TrainingConfig, x4k_config: Dict):
        self.config = config
        self.x4k_config = x4k_config

        # DDP Setup
        self.is_distributed = setup_distributed_training()
        self.is_main_process = is_main_process()

        # Set device based on local rank
        if self.is_distributed:
            self.device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
            os.environ['NCCL_TIMEOUT'] = '600'
        else:
            self.device = torch.device(config.device)

        self.ssim_metric = SSIM(data_range=1.0).to(self.device)

        # AMP setup
        self.amp_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.amp_dtype = (
            torch.bfloat16 if config.amp_dtype.lower() == "bf16"
            else torch.float16 if config.amp_dtype.lower() == "fp16"
            else torch.float32
        )
        self.use_autocast = (
            config.use_amp and autocast_mode.is_autocast_available(self.amp_device_type)
        )

        # Scaler only for CUDA + fp16
        if self.use_autocast and self.amp_device_type == "cuda" and self.amp_dtype == torch.float16:
            self.scaler = GradScaler(device_type="cuda")
        else:
            self.scaler = None

        # Run manager (only main process)
        if self.is_main_process:
            self.run_manager = RunManager(config)
        else:
            self.run_manager = None

        print(f"🧪 AMP: use_autocast={self.use_autocast}, device={self.amp_device_type}, "
              f"dtype={self.amp_dtype}, scaler={'yes' if self.scaler else 'no'}")

        # Build model
        print("🏗️ Building model...")
        self.model = build_tempo(
            base_channels=config.base_channels,
            temporal_channels=config.temporal_channels,
            encoder_depths=config.encoder_depths,
            decoder_depths=config.decoder_depths,
            num_heads=config.num_heads,
            num_points=config.num_points,
            use_cross_scale=config.use_cross_scale,
            use_checkpointing=config.use_checkpointing,
        ).to(self.device)

        # Wrap for DDP
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.device.index], find_unused_parameters=True)

        if config.compile_model and hasattr(torch, 'compile'):
            print("  ⚡ Compiling model with PyTorch 2.0 (max-autotune)...")
            self.model = torch.compile(
                self.model,
                mode="max-autotune",  # Try all CUDA kernels, pick fastest
                fullgraph=True,       # Compile entire forward pass
            )

        # Loss
        self.loss_fn = build_tempo_loss(config.loss_config).to(self.device)
        self.loss_scheduler = LossScheduler(self.loss_fn)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )

        # Data
        self._setup_data()

        # LR scheduler
        self.lr_scheduler = self._build_lr_scheduler()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_psnr = 0.0

        # Resume if specified
        if config.resume:
            self._load_checkpoint(config.resume)

        # Print info
        if self.is_main_process:
            self._print_model_info()

    def _build_lr_scheduler(self):
        if self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs * len(self.train_loader),
                eta_min=1e-6
            )
        elif self.config.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.5
            )
        return None

    def _setup_data(self):
        print("📊 Loading datasets...")

        # ===== Vimeo Training Dataset =====
        vimeo_train = Vimeo90KTriplet(
            root=self.config.data_root,
            split="train",
            mode="mix",
            crop_size=self.config.crop_size,
            aug_flip=False,
        )

        # ===== X4K Training Dataset =====
        x4k_train = X4K1000Dataset(
            root=self.x4k_config['root'],
            split="train",
            steps=self.x4k_config['steps'],  # Can be single int or list
            crop_size=self.x4k_config['crop_size'],
            aug_flip=True,
            n_frames=4,
        )

        # ===== Concatenate Datasets =====
        train_dataset = ConcatDataset([vimeo_train, x4k_train])

        dataset_sizes = [len(vimeo_train), len(x4k_train)]

        # Calculate optimal ratio for full dataset coverage if requested
        requested_ratio = self.x4k_config['vimeo_ratio']
        if requested_ratio < 0:  # Use -1 to trigger auto calculation
            # Calculate batch counts
            vimeo_batches = len(vimeo_train) // self.config.batch_size
            x4k_batches = len(x4k_train) // self.config.batch_size
            total_batches = vimeo_batches + x4k_batches

            # Set ratio proportional to dataset sizes
            vimeo_ratio = vimeo_batches / total_batches
            x4k_ratio = x4k_batches / total_batches

            print(f"  Vimeo: {len(vimeo_train):,} samples (N=2) → {vimeo_batches:,} batches")
            print(f"  X4K:   {len(x4k_train):,} samples (N=4, STEPs={self.x4k_config['steps']}) → {x4k_batches:,} batches")
            print(f"  Auto ratio: {vimeo_ratio:.3f} Vimeo / {x4k_ratio:.3f} X4K (uses ALL samples per epoch)")
        else:
            vimeo_ratio = requested_ratio
            x4k_ratio = 1.0 - vimeo_ratio

            print(f"  Vimeo: {len(vimeo_train):,} samples (N=2)")
            print(f"  X4K:   {len(x4k_train):,} samples (N=4, STEPs={self.x4k_config['steps']})")
            print(f"  Batch ratio: {vimeo_ratio:.0%} Vimeo / {x4k_ratio:.0%} X4K (may skip samples)")

        # ===== Pure Batch Sampler =====
        if self.is_distributed:
            self.train_sampler = DistributedPureBatchSampler(
                dataset_sizes=dataset_sizes,
                batch_size=self.config.batch_size,
                ratios=[vimeo_ratio, x4k_ratio],
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                drop_last=True,
            )
        else:
            self.train_sampler = PureBatchSampler(
                dataset_sizes=dataset_sizes,
                batch_size=self.config.batch_size,
                ratios=[vimeo_ratio, x4k_ratio],
                shuffle=True,
                drop_last=True,
            )

        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=self.train_sampler,
            num_workers=self.config.num_workers,
            collate_fn=self._mixed_collate,
            pin_memory=True,
        )

        # ===== Validation Dataset (Vimeo Only) =====
        vimeo_val = Vimeo90KTriplet(
            root=self.config.data_root,
            split="test",
            mode="interp",
            crop_size=None,
            center_crop_eval=False
        )

        vimeo_sampler = DistributedSampler(vimeo_val, shuffle=False) if self.is_distributed else None

        self.val_loader = DataLoader(
            vimeo_val,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            sampler=vimeo_sampler,
            collate_fn=vimeo_collate,
            pin_memory=True
        )

        print(f"  Vimeo Validation: {len(vimeo_val):,} samples")

        # ===== X4K Test Dataset =====
        x4k_test = X4K1000TestDataset(
            root=self.x4k_config['root'],
            step=1,  # Always use STEP=1 for validation (consistent evaluation)
            n_frames=4,
        )

        x4k_sampler = DistributedSampler(x4k_test, shuffle=False) if self.is_distributed else None

        self.x4k_test_loader = DataLoader(
            x4k_test,
            batch_size=1,  # Process one 4K image at a time (for tiled inference)
            shuffle=False,
            num_workers=4,
            sampler=x4k_sampler,
            collate_fn=x4k_test_collate,
            pin_memory=True
        )

        print(f"  X4K Test: {len(x4k_test):,} samples (N=4, STEP=1)")

    def _mixed_collate(self, batch):
        """Auto-detect dataset type and route to correct collate function"""
        # Check N from first sample
        N = batch[0][0].shape[0]  # frames: [N,3,H,W]

        if N == 2:
            return vimeo_collate(batch)
        elif N == 4:
            return x4k_collate(batch)
        else:
            raise ValueError(f"Unexpected N={N} frames in batch")

    def _save_image_grid(self, images: Dict[str, torch.Tensor], path: Path):
        """
        Save validation samples as 2x3 grid.

        Grid layout:
          Row 1: frame0  | frame1  | target
          Row 2: pred    | error   | conf

        Args:
            images: Dict with keys ['frame0', 'frame1', 'target', 'pred', 'error', 'conf']
            path: Save path for PNG file
        """
        keys = ['frame0', 'frame1', 'target', 'pred', 'error', 'conf']
        imgs = []

        for k in keys:
            if k in images:
                img = images[k].clamp(0, 1)
                # Convert to numpy uint8 [H, W, 3]
                img = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
                imgs.append(img)

        if imgs:
            h, w = imgs[0].shape[:2]
            # Create 2x3 grid
            grid = np.zeros((h*2, w*3, 3), dtype=np.uint8)
            for i, img in enumerate(imgs):
                row, col = i // 3, i % 3
                grid[row*h:(row+1)*h, col*w:(col+1)*w] = img

            Image.fromarray(grid).save(path)

    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Get component breakdown
        model_ref = self.model.module if self.is_distributed else self.model
        enc_params = sum(p.numel() for p in model_ref.encoder.parameters())
        dec_params = sum(p.numel() for p in model_ref.decoder.parameters())
        fus_params = sum(p.numel() for p in model_ref.fusion.parameters())

        print(f"\n📊 Model Statistics:")
        print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable: {trainable_params:,}")
        print(f"\n  Component breakdown:")
        print(f"    Encoder (ConvNeXt):      {enc_params/1e6:.2f}M")
        print(f"    Fusion (CrossAttention): {fus_params/1e6:.2f}M")
        print(f"    Decoder (NAFNet):        {dec_params/1e6:.2f}M")

    def train_epoch(self) -> Dict[str, Dict[str, float]]:
        self.model.train()

        # Separate trackers for each dataset
        vimeo_tracker = MetricTracker()
        x4k_tracker = MetricTracker()

        if self.is_distributed:
            dist.barrier()

        iterable = self.train_loader
        pbar = None
        if self.is_main_process:
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {self.epoch+1}/{self.config.epochs}",
                unit="batch",
                ncols=160,
                bar_format="{l_bar}{bar:30}{r_bar}",
                colour="cyan"
            )
            iterable = pbar

        for batch_idx, (frames, anchor_times, target_time, target) in enumerate(iterable):
            frames = frames.to(self.device, non_blocking=True)
            anchor_times = anchor_times.to(self.device, non_blocking=True)
            target_time = target_time.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # Warmup
            if self.global_step < self.config.warmup_steps:
                lr_scale = (self.global_step + 1) / self.config.warmup_steps
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.config.learning_rate * lr_scale

            self.loss_scheduler.update(self.global_step)

            # Forward
            if self.use_autocast:
                with autocast(self.amp_device_type, dtype=self.amp_dtype):
                    pred, aux = self.model(frames, anchor_times, target_time)
                    loss, metrics = self.loss_fn(pred, target, frames, anchor_times, target_time, aux)
            else:
                pred, aux = self.model(frames, anchor_times, target_time)
                loss, metrics = self.loss_fn(pred, target, frames, anchor_times, target_time, aux)

            # Backward
            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            if self.lr_scheduler and self.global_step >= self.config.warmup_steps:
                self.lr_scheduler.step()

            # Detect batch type and update corresponding tracker
            N = frames.shape[1]
            batch_type = "vimeo" if N == 2 else "x4k"

            if batch_type == "vimeo":
                vimeo_tracker.update(metrics)
            else:
                x4k_tracker.update(metrics)

            metrics['lr'] = self.optimizer.param_groups[0]['lr']

            if self.is_main_process and pbar is not None:
                pbar.set_postfix({
                    'type': batch_type,
                    'loss': f"{metrics['total']:.4f}",
                    'l1': f"{metrics.get('l1', 0):.3f}",
                    'psnr': f"{metrics.get('psnr', 0):.2f}",
                    'lr': f"{metrics['lr']:.1e}"
                })

            # Logging - separate metrics for each dataset
            if self.global_step % self.config.log_interval == 0 and self.is_main_process:
                if self.run_manager:
                    # Log Vimeo metrics
                    vimeo_avg = vimeo_tracker.get_averages()
                    if vimeo_avg:  # Only log if we have Vimeo samples
                        vimeo_prefixed = {f"vimeo/{k}": v for k, v in vimeo_avg.items()}
                        self.run_manager.log_metrics(vimeo_prefixed, self.global_step, "train")

                    # Log X4K metrics
                    x4k_avg = x4k_tracker.get_averages()
                    if x4k_avg:  # Only log if we have X4K samples
                        x4k_prefixed = {f"x4k/{k}": v for k, v in x4k_avg.items()}
                        self.run_manager.log_metrics(x4k_prefixed, self.global_step, "train")

                vimeo_tracker.reset()
                x4k_tracker.reset()

            # Validation
            if self.global_step % self.config.val_interval == 0 and self.global_step > 0:
                if self.is_distributed:
                    dist.barrier()

                # Validate on both datasets
                vimeo_metrics = self.validate_vimeo()
                x4k_metrics = self.validate_x4k()

                if self.is_main_process and self.run_manager:
                    # Log separately with prefixes
                    vimeo_prefixed = {f"vimeo/{k}": v for k, v in vimeo_metrics.items()}
                    x4k_prefixed = {f"x4k/{k}": v for k, v in x4k_metrics.items()}

                    self.run_manager.log_metrics(vimeo_prefixed, self.global_step, "val")
                    self.run_manager.log_metrics(x4k_prefixed, self.global_step, "val")

                    # Update best PSNR (use Vimeo as primary metric)
                    if vimeo_metrics.get('psnr', 0) > self.best_psnr:
                        self.best_psnr = vimeo_metrics['psnr']

                self.model.train()

                if self.is_distributed:
                    dist.barrier()

            # Checkpointing
            if self.global_step % self.config.save_interval == 0 and self.global_step > 0 and self.is_main_process:
                if self.run_manager:
                    self.run_manager.save_checkpoint(
                        self.model, self.optimizer, self.lr_scheduler,
                        self.global_step, self.epoch, self.best_psnr,
                        is_best=False,
                        is_main_process=self.is_main_process,
                        is_distributed=self.is_distributed,
                    )

            self.global_step += 1

        # Return separate metrics for each dataset
        return {
            'vimeo': vimeo_tracker.get_averages(),
            'x4k': x4k_tracker.get_averages(),
        }

    @torch.no_grad()
    def validate_vimeo(self) -> Dict[str, float]:
        """Validate on Vimeo dataset (N=2, full resolution)"""
        self.model.eval()
        self.ssim_metric.reset()

        total_psnr = 0.0
        total_ssim_sum = 0.0
        num_samples = 0

        model_to_eval = self.model.module if self.is_distributed else self.model

        iterable = self.val_loader
        pbar = None
        if self.is_main_process:
            pbar = tqdm(self.val_loader, desc="Validating Vimeo", unit="sample",
                       ncols=120, leave=False, colour="green")
            iterable = pbar

        # Compute sample indices for saving
        num_val_batches = len(self.val_loader)
        sample_indices = np.linspace(0, num_val_batches - 1,
                                    min(self.config.n_val_samples, num_val_batches),
                                    dtype=int)

        for idx, (frames, anchor_times, target_time, target) in enumerate(iterable):
            frames = frames.to(self.device, non_blocking=True)
            anchor_times = anchor_times.to(self.device, non_blocking=True)
            target_time = target_time.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            pred, aux = model_to_eval(frames, anchor_times, target_time)
            pred = pred.clamp(0, 1)

            # Metrics
            mse = F.mse_loss(pred, target)
            psnr = -10 * torch.log10(mse + 1e-8)
            total_psnr += psnr.item() * frames.size(0)
            num_samples += frames.size(0)

            self.ssim_metric.update(pred, target)
            batch_ssim = self.ssim_metric(pred, target).item()
            total_ssim_sum += batch_ssim * frames.size(0)

            # Save samples (main process only)
            if idx in sample_indices and self.is_main_process and self.run_manager:
                viz_dict = {
                    'frame0': frames[0, 0],
                    'frame1': frames[0, -1],
                    'target': target[0],
                    'pred': pred[0],
                    'error': (pred[0] - target[0]).abs().mean(0, True).repeat(3,1,1),
                    'conf': aux['confidence'][0].repeat(3,1,1) if 'confidence' in aux else torch.zeros_like(pred[0])
                }
                save_path = self.run_manager.sample_dir / f"vimeo_epoch_{self.epoch:03d}_sample_{idx:03d}.png"
                self._save_image_grid(viz_dict, save_path)
                self.run_manager.log_images(viz_dict, self.global_step, "val/vimeo")

            if self.is_main_process and pbar is not None:
                pbar.set_postfix({
                    'psnr': f"{psnr.item():.2f}",
                    'ssim': f"{batch_ssim:.4f}",
                    'avg_psnr': f"{total_psnr/num_samples:.2f}"
                })

        # Aggregate across GPUs
        if self.is_distributed:
            local_results = torch.tensor([total_psnr, total_ssim_sum, float(num_samples)],
                                        dtype=torch.float64, device=self.device)
            dist.all_reduce(local_results, op=dist.ReduceOp.SUM)
            total_psnr = local_results[0].item()
            total_ssim_sum = local_results[1].item()
            num_samples = int(local_results[2].item())

        avg_psnr = total_psnr / max(1, num_samples)
        avg_ssim = self.ssim_metric.compute().item()

        if self.is_main_process:
            print(f"\n  📈 Vimeo Validation: PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}")

        return {'psnr': avg_psnr, 'ssim': avg_ssim}

    @torch.no_grad()
    def validate_x4k(self) -> Dict[str, float]:
        """
        Validate on X4K test dataset (N=4, 4K resolution with tiling).

        Process:
        1. Tile-based inference: 512×512 tiles with 64px overlap
        2. Stitch tiles back to full 4K resolution with weighted blending
        3. Compute PSNR/SSIM on FULL STITCHED 4K image (not crops)
        4. Save 512×512 center crops as preview samples (visualization only)
        """
        self.model.eval()
        self.ssim_metric.reset()

        total_psnr = 0.0
        total_ssim_sum = 0.0
        num_samples = 0

        model_to_eval = self.model.module if self.is_distributed else self.model

        iterable = self.x4k_test_loader
        pbar = None
        if self.is_main_process:
            pbar = tqdm(self.x4k_test_loader, desc="Validating X4K", unit="sample",
                       ncols=120, leave=False, colour="blue")
            iterable = pbar

        # Compute sample indices for saving
        num_val_batches = len(self.x4k_test_loader)
        sample_indices = np.linspace(0, num_val_batches - 1,
                                    min(self.config.n_val_samples, num_val_batches),
                                    dtype=int)

        for idx, (frames, anchor_times, target_time, target) in enumerate(iterable):
            frames = frames.to(self.device, non_blocking=True)
            anchor_times = anchor_times.to(self.device, non_blocking=True)
            target_time = target_time.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # ===== STEP 1: Tiled inference + stitching =====
            # Processes 4K image in 512×512 tiles, stitches back to full resolution
            pred_full = infer_with_tiling(
                model_to_eval, frames, anchor_times, target_time,
                tile_size=512, overlap=64
            )
            pred_full = pred_full.clamp(0, 1)

            # ===== STEP 2: Compute metrics on FULL STITCHED 4K image =====
            mse = F.mse_loss(pred_full, target)
            psnr = -10 * torch.log10(mse + 1e-8)
            total_psnr += psnr.item()
            num_samples += 1

            # SSIM also computed on full 4K resolution
            self.ssim_metric.update(pred_full, target)
            batch_ssim = self.ssim_metric(pred_full, target).item()
            total_ssim_sum += batch_ssim

            # ===== STEP 3: Save center crop samples (512×512) for visualization =====
            # Note: Metrics are computed on FULL 4K, but we save crops for easier inspection
            if idx in sample_indices and self.is_main_process and self.run_manager:
                _, _, H, W = pred_full.shape
                crop_size = 512

                # Center crop all frames for visualization
                cy, cx = H // 2, W // 2
                y1, y2 = cy - crop_size // 2, cy + crop_size // 2
                x1, x2 = cx - crop_size // 2, cx + crop_size // 2

                viz_dict = {
                    'frame0': frames[0, 0, :, y1:y2, x1:x2],
                    'frame1': frames[0, -1, :, y1:y2, x1:x2],
                    'target': target[0, :, y1:y2, x1:x2],
                    'pred': pred_full[0, :, y1:y2, x1:x2],
                    'error': (pred_full[0, :, y1:y2, x1:x2] - target[0, :, y1:y2, x1:x2]).abs().mean(0, True).repeat(3,1,1),
                    'conf': torch.zeros(3, crop_size, crop_size, device=self.device)  # Placeholder
                }

                save_path = self.run_manager.sample_dir / f"x4k_epoch_{self.epoch:03d}_sample_{idx:03d}.png"
                self._save_image_grid(viz_dict, save_path)
                self.run_manager.log_images(viz_dict, self.global_step, "val/x4k")

            if self.is_main_process and pbar is not None:
                pbar.set_postfix({
                    'psnr': f"{psnr.item():.2f}",
                    'ssim': f"{batch_ssim:.4f}",
                    'avg_psnr': f"{total_psnr/num_samples:.2f}"
                })

        # Aggregate across GPUs
        if self.is_distributed:
            local_results = torch.tensor([total_psnr, total_ssim_sum, float(num_samples)],
                                        dtype=torch.float64, device=self.device)
            dist.all_reduce(local_results, op=dist.ReduceOp.SUM)
            total_psnr = local_results[0].item()
            total_ssim_sum = local_results[1].item()
            num_samples = int(local_results[2].item())

        avg_psnr = total_psnr / max(1, num_samples)
        avg_ssim = self.ssim_metric.compute().item()

        if self.is_main_process:
            print(f"\n  📈 X4K Validation: PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}")

        return {'psnr': avg_psnr, 'ssim': avg_ssim}

    def _load_checkpoint(self, path: str):
        print(f"📂 Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        state_dict = checkpoint['model_state']

        # Handle DDP module. prefix
        if self.is_distributed:
            if not any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
        else:
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Handle torch.compile _orig_mod. prefix
        model_is_compiled = any(k.startswith('_orig_mod.') for k in self.model.state_dict().keys())
        checkpoint_is_compiled = any(k.startswith('_orig_mod.') for k in state_dict.keys())

        if model_is_compiled and not checkpoint_is_compiled:
            # Model compiled but checkpoint is not - add prefix
            state_dict = {'_orig_mod.' + k: v for k, v in state_dict.items()}
        elif not model_is_compiled and checkpoint_is_compiled:
            # Checkpoint compiled but model is not - remove prefix
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        if checkpoint.get('scheduler_state') and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state'])

        self.global_step = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_psnr = checkpoint.get('best_metric', 0)

        print(f"  Resumed from step {self.global_step}, epoch {self.epoch}")

    def train(self):
        if self.is_main_process:
            print("\n🚀 Starting mixed training (Vimeo + X4K)...\n")

        try:
            for epoch in range(self.epoch, self.config.epochs):
                self.epoch = epoch

                if self.is_distributed and hasattr(self.train_sampler, 'set_epoch'):
                    self.train_sampler.set_epoch(epoch)

                epoch_metrics = self.train_epoch()

                if self.is_distributed:
                    dist.barrier()

                # Validate on both datasets
                vimeo_metrics = self.validate_vimeo()
                x4k_metrics = self.validate_x4k()

                if self.is_distributed:
                    dist.barrier()

                if self.is_main_process:
                    # Determine if best (use Vimeo PSNR as primary)
                    is_best = vimeo_metrics['psnr'] > self.best_psnr
                    if is_best:
                        self.best_psnr = vimeo_metrics['psnr']

                    # Log to WandB/TensorBoard
                    if self.run_manager:
                        vimeo_prefixed = {f"vimeo/{k}": v for k, v in vimeo_metrics.items()}
                        x4k_prefixed = {f"x4k/{k}": v for k, v in x4k_metrics.items()}
                        self.run_manager.log_metrics(vimeo_prefixed, self.global_step, "val")
                        self.run_manager.log_metrics(x4k_prefixed, self.global_step, "val")

                    print(f"\n📊 Epoch {epoch+1} Summary:")
                    print(f"  Train Loss (Vimeo): {epoch_metrics['vimeo'].get('total', 0):.4f}")
                    print(f"  Train Loss (X4K):   {epoch_metrics['x4k'].get('total', 0):.4f}")
                    print(f"  Val PSNR (Vimeo):   {vimeo_metrics['psnr']:.2f} dB, SSIM: {vimeo_metrics['ssim']:.4f}")
                    print(f"  Val PSNR (X4K):     {x4k_metrics['psnr']:.2f} dB, SSIM: {x4k_metrics['ssim']:.4f}")
                    print(f"  Best PSNR (Vimeo):  {self.best_psnr:.2f} dB")

                    if self.run_manager:
                        self.run_manager.save_checkpoint(
                            self.model, self.optimizer, self.lr_scheduler,
                            self.global_step, epoch, self.best_psnr,
                            is_best=is_best,
                            is_main_process=self.is_main_process,
                            is_distributed=self.is_distributed,
                        )

        except KeyboardInterrupt:
            if self.is_main_process:
                print("\n\n⚠️ Training interrupted by user")
        except Exception as e:
            if self.is_main_process:
                print(f"\n\n❌ Training failed: {e}")
            raise
        finally:
            if self.is_main_process:
                print("\n💾 Saving final checkpoint...")
                if self.run_manager:
                    self.run_manager.save_checkpoint(
                        self.model, self.optimizer, self.lr_scheduler,
                        self.global_step, self.epoch, self.best_psnr,
                        is_best=False,
                        is_main_process=self.is_main_process,
                        is_distributed=self.is_distributed,
                    )
                    self.run_manager.close()
                    print(f"\n✅ Training complete! Results saved to: {self.run_manager.run_dir}")

            if self.is_distributed:
                cleanup_distributed_training()


# ===========================
# Entry Point
# ===========================

def main():
    from dataclasses import replace

    # Create default config (single source of truth)
    default_config = TrainingConfig()

    parser = argparse.ArgumentParser(description="TEMPO Mixed Training (Vimeo + X4K)")

    # Basic settings (no defaults - use TrainingConfig defaults)
    parser.add_argument("--data_root", type=str,
                       help="Path to Vimeo90K dataset")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float, dest='learning_rate')

    # X4K dataset settings (not part of TrainingConfig, so provide defaults here)
    parser.add_argument("--x4k_root", type=str, default="/Users/nightstalker/Projects/datasets",
                       help="Path to X4K1000 dataset")
    parser.add_argument("--x4k_step", type=int, nargs='+', default=[1],
                       help="STEP parameter(s) for X4K. Can be single (e.g., 3) or multiple (e.g., 1 3). "
                            "1=small motion, 2=medium, 3=large. Multiple steps increase motion diversity.")
    parser.add_argument("--x4k_crop", type=int, default=512,
                       help="Crop size for X4K (512 or 768)")
    parser.add_argument("--vimeo_ratio", type=float, default=0.5,
                       help="Fraction of batches from Vimeo (0.0-1.0). "
                            "Use -1 for automatic calculation to see ALL samples from both datasets per epoch.")

    # Model architecture
    parser.add_argument("--base_channels", type=int)
    parser.add_argument("--temporal_channels", type=int)
    parser.add_argument("--encoder_depths", type=int, nargs='+')
    parser.add_argument("--decoder_depths", type=int, nargs='+')

    # Experiment
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--notes", type=str)
    parser.add_argument("--resume", type=str)

    # Hardware
    parser.add_argument("--device", type=str)
    parser.add_argument("--compile", action="store_true", dest='compile_model',
                       help="Use PyTorch 2.0 compile")
    parser.add_argument("--amp", action="store_true", dest='use_amp')
    parser.add_argument("--amp_dtype", type=str, choices=["fp16", "bf16", "fp32"])

    # Loss configuration
    parser.add_argument("--loss_preset", type=str, choices=["psnr", "balanced", "ssim"],
                       default="psnr",
                       help="Loss configuration preset: 'psnr' (default, max PSNR), "
                            "'balanced' (balance PSNR+SSIM), 'ssim' (prioritize SSIM)")

    # Logging
    parser.add_argument("--use_wandb", action="store_true")

    # Distributed
    parser.add_argument("--distributed", action="store_true")

    args = parser.parse_args()

    # Separate X4K-specific args (not part of TrainingConfig)
    x4k_args = {
        'x4k_root', 'x4k_step', 'x4k_crop', 'vimeo_ratio', 'loss_preset'
    }

    # Build override dict from explicitly provided TrainingConfig args
    overrides = {}
    for key, value in vars(args).items():
        if key in x4k_args:
            continue  # Skip X4K-specific args

        if value is not None:
            # For store_true flags, only override if True (explicitly set)
            if isinstance(value, bool):
                if value:  # Only add if True (user provided the flag)
                    overrides[key] = value
            else:
                overrides[key] = value

    # Apply default exp_name and notes if not provided
    if 'exp_name' not in overrides:
        overrides['exp_name'] = f"mixed_vimeo_x4k_step{args.x4k_step}"
    if 'notes' not in overrides:
        overrides['notes'] = f"Mixed training: Vimeo (N=2) + X4K (N=4, STEP={args.x4k_step})"

    # Apply overrides to default config
    config = replace(default_config, **overrides)

    # Apply loss configuration preset
    if args.loss_preset == "balanced":
        from config.loss_balanced import BALANCED_LOSS_CONFIG
        config = replace(config, loss_config=BALANCED_LOSS_CONFIG)
        print(f"🎯 Using BALANCED loss configuration (optimizes both PSNR and SSIM)")
    elif args.loss_preset == "ssim":
        from config.loss_balanced import SSIM_FOCUSED_CONFIG
        config = replace(config, loss_config=SSIM_FOCUSED_CONFIG)
        print(f"🎯 Using SSIM-FOCUSED loss configuration (prioritizes SSIM)")
    else:
        # Default PSNR-optimized (no config override needed)
        print(f"🎯 Using PSNR-OPTIMIZED loss configuration (default)")

    # Build X4K config dict
    x4k_config = {
        'root': args.x4k_root,
        'steps': args.x4k_step,
        'crop_size': args.x4k_crop,
        'vimeo_ratio': args.vimeo_ratio,
    }

    trainer = MixedTrainer(config, x4k_config)
    trainer.train()


if __name__ == "__main__":
    main()
