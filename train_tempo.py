"""
TEMPO v2 Training System - Fixed Version
Features:
- WandB + TensorBoard logging
- Organized run management
- Best model checkpointing
- Validation sample generation
- Comprehensive metrics tracking
- Resume capability
"""
import argparse
from pathlib import Path
from typing import Dict
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

# Your model imports
from model.tempo import build_tempo
from model.loss.tempo_loss import build_tempo_loss, LossScheduler, MetricTracker
from data.data_vimeo_triplet import Vimeo90KTriplet, vimeo_collate
from config.default import TrainingConfig
from config.manager import RunManager
from config.dpp import setup_distributed_training, cleanup_distributed_training, is_main_process
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

class Trainer:
    """Main training orchestrator"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        # --- DDP Setup ---
        self.is_distributed = setup_distributed_training()
        self.is_main_process = is_main_process()
        
        # Set device based on local rank
        if self.is_distributed:
            self.device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
            # Increase NCCL timeout for validation
            os.environ['NCCL_TIMEOUT'] = '600'  # 10 minutes
        else:
            self.device = torch.device(config.device)

        self.ssim_metric = SSIM(data_range=1.0).to(self.device)

        # AMP setup (unified API)
        self.amp_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.amp_dtype = torch.bfloat16 if config.amp_dtype.lower() == "bf16" else torch.float16 if config.amp_dtype.lower() == "fp16" else torch.float32
        self.use_autocast = (
            config.use_amp and autocast_mode.is_autocast_available(self.amp_device_type)
        )

        # Scaler only for CUDA + fp16 (bf16/CPU don't need scaling)
        if self.use_autocast and self.amp_device_type == "cuda" and self.amp_dtype == torch.float16:
            self.scaler = GradScaler(device_type="cuda")
        else:
            self.scaler = None
        
        # Setup run manager (only on main process)
        if self.is_main_process:
            self.run_manager = RunManager(config)
        else:
            self.run_manager = None
            
        print(f"🧪 AMP: use_autocast={self.use_autocast}, device={self.amp_device_type}, dtype={self.amp_dtype}, scaler={'yes' if self.scaler else 'no'}")
        
        # Build model
        print("🏗️ Building model...")
        self.model = build_tempo(
            base_channels=config.base_channels,
            temporal_channels=config.temporal_channels,
            attn_heads=config.attn_heads,
            attn_points=config.attn_points
        ).to(self.device)

        # --- Wrap model for DDP if enabled ---
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.device.index], find_unused_parameters=True)
        
        if config.compile_model and hasattr(torch, 'compile'):
            print("  ⚡ Compiling model with PyTorch 2.0...")
            self.model = torch.compile(self.model)
            
        # Loss function with scheduler
        self.loss_fn = build_tempo_loss(config.loss_config).to(self.device)
        self.loss_scheduler = LossScheduler(self.loss_fn)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Data loaders
        self._setup_data()
        
        # Learning rate scheduler
        self.lr_scheduler = self._build_lr_scheduler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_psnr = 0.0
        
        # Resume if specified
        if config.resume:
            self._load_checkpoint(config.resume)
            
        # Print model info
        if self.is_main_process:
            self._print_model_info()
        
    def _build_lr_scheduler(self):
        """Build learning rate scheduler"""
        if self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs * len(self.train_loader),
                eta_min=1e-6
            )
        elif self.config.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.5
            )
        else:
            return None
            
    def _setup_data(self):
        """Setup data loaders"""
        print("📊 Loading datasets...")
        
        # Training
        train_dataset = Vimeo90KTriplet(
            root=self.config.data_root,
            split="train",
            mode="mix",
            crop_size=self.config.crop_size,
            aug_flip=False,
        )

        self.train_sampler = None
        if self.is_distributed:
            self.train_sampler = DistributedSampler(train_dataset)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=self.config.num_workers,
            collate_fn=vimeo_collate,
            pin_memory=True,
            drop_last=True
        )
        
        # Validation
        val_dataset = Vimeo90KTriplet(
            root=self.config.data_root,
            split="test",
            mode="interp",
            crop_size=None,  # Full resolution
            center_crop_eval=False
        )

        self.val_sampler = None
        if self.is_distributed:
            self.val_sampler = DistributedSampler(val_dataset, shuffle=False)
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Full resolution validation
            shuffle=False,
            num_workers=4,
            sampler=self.val_sampler,
            collate_fn=vimeo_collate,
            pin_memory=True
        )
        
        print(f"  Training samples: {len(train_dataset):,}")
        print(f"  Validation samples: {len(val_dataset):,}")
        
    def _print_model_info(self):
        """Print model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n📊 Model Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024**2:.2f} MB (fp32)")
        
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        metric_tracker = MetricTracker()
        
        # Synchronize before starting epoch
        if self.is_distributed:
            dist.barrier()
        
        iterable = self.train_loader
        # Beautiful progress bar only on main process
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
            # Move to device
            frames = frames.to(self.device, non_blocking=True)
            anchor_times = anchor_times.to(self.device, non_blocking=True)
            target_time = target_time.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            # Learning rate warmup
            if self.global_step < self.config.warmup_steps:
                lr_scale = (self.global_step + 1) / self.config.warmup_steps
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.config.learning_rate * lr_scale
                    
            # Update loss schedule for this step
            self.loss_scheduler.update(self.global_step)

            # Forward pass with unified autocast
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
                
            # Update schedulers
            if self.lr_scheduler and self.global_step >= self.config.warmup_steps:
                self.lr_scheduler.step()
            self.loss_scheduler.update(self.global_step)
            
            # Track metrics
            metric_tracker.update(metrics)
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
            
            # Update progress bar (only main process)
            if self.is_main_process and pbar is not None:
                pbar.set_postfix({
                    'loss': f"{metrics['total']:.4f}",
                    'l1': f"{metrics.get('l1', 0):.3f}",
                    'ssim': f"{metrics.get('ssim', 0):.3f}",
                    'psnr': f"{metrics.get('psnr', 0):.2f}",
                    'lr': f"{metrics['lr']:.1e}"
                })
            
            # Logging (only main process)
            if self.global_step % self.config.log_interval == 0 and self.is_main_process:
                avg_metrics = metric_tracker.get_averages()
                if self.run_manager:
                    self.run_manager.log_metrics(avg_metrics, self.global_step, "train")
                metric_tracker.reset()
            
            # CRITICAL FIX: ALL processes must participate in validation
            if self.global_step % self.config.val_interval == 0 and self.global_step > 0:
                # Ensure all processes are ready for validation
                if self.is_distributed:
                    dist.barrier()
                    
                val_metrics = self.validate()  # ALL processes run this
                
                # Only main process logs results
                if self.is_main_process and self.run_manager:
                    self.run_manager.log_metrics(val_metrics, self.global_step, "val")
                    
                    # Check if best
                    current_psnr = val_metrics.get('psnr', 0)
                    is_best = current_psnr > self.best_psnr
                    if is_best:
                        self.best_psnr = current_psnr
                
                # ALL processes resume training mode
                self.model.train()
                
                # Sync after validation
                if self.is_distributed:
                    dist.barrier()
            
            # Checkpointing (only main process saves)
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
        
        return metric_tracker.get_averages()
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation with sample generation (distributed-safe with robust SSIM)."""
        self.model.eval()
        
        # Reset the metric state at the beginning of validation
        self.ssim_metric.reset()
        
        total_psnr = 0.0
        total_ssim_sum = 0.0  # Track SSIM sum for manual averaging
        num_samples = 0

        model_to_eval = self.model.module if self.is_distributed else self.model
        
        # ALL processes iterate through validation
        iterable = self.val_loader
        pbar = None
        if self.is_main_process:
            pbar = tqdm(self.val_loader, desc="Validating", unit="sample",
                       ncols=120, leave=False, colour="green")
            iterable = pbar
        
        for idx, (frames, anchor_times, target_time, target) in enumerate(iterable):
            frames = frames.to(self.device, non_blocking=True)
            anchor_times = anchor_times.to(self.device, non_blocking=True)
            target_time = target_time.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            pred, aux = model_to_eval(frames, anchor_times, target_time)
            pred = pred.clamp(0, 1)
            
            # Compute PSNR locally
            mse = F.mse_loss(pred, target)
            psnr = -10 * torch.log10(mse + 1e-8)  # Add epsilon for stability
            total_psnr += psnr.item() * frames.size(0)
            num_samples += frames.size(0)
            
            # Update the SSIM metric
            self.ssim_metric.update(pred, target)
            
            # Also compute SSIM per-batch for display
            batch_ssim = self.ssim_metric(pred, target).item()
            total_ssim_sum += batch_ssim * frames.size(0)
            
            # Sample saving only on main process
            if self.is_main_process:
                # Update progress bar with both PSNR and SSIM
                if pbar is not None:
                    pbar.set_postfix({
                        'psnr': f"{psnr.item():.2f}",
                        'ssim': f"{batch_ssim:.4f}",
                        'avg_psnr': f"{total_psnr/num_samples:.2f}",
                        'avg_ssim': f"{total_ssim_sum/num_samples:.4f}"
                    })
                
                # Save visualization samples
                samples_per_gpu = len(self.val_loader)
                sample_indices = np.linspace(0, samples_per_gpu - 1, 
                                            min(self.config.n_val_samples, samples_per_gpu), 
                                            dtype=int)
                
                if idx in sample_indices:
                    viz_dict = {
                        'frame0': frames[0, 0], 
                        'frame1': frames[0, 1], 
                        'target': target[0],
                        'pred': pred[0], 
                        'error': (pred[0] - target[0]).abs().mean(0, True).repeat(3,1,1),
                        'conf': aux['conf_map'][0].repeat(3,1,1) if 'conf_map' in aux else torch.zeros_like(pred[0])
                    }
                    if self.run_manager:
                        save_path = self.run_manager.sample_dir / f"step_{self.global_step:06d}_sample_{idx:03d}.png"
                        self._save_image_grid(viz_dict, save_path)
                        self.run_manager.log_images(viz_dict, self.global_step, "val")

        # Aggregate results from all GPUs
        if self.is_distributed:
            # Gather metrics from all processes
            local_results = torch.tensor([total_psnr, total_ssim_sum, float(num_samples)], 
                                        dtype=torch.float64, device=self.device)
            dist.all_reduce(local_results, op=dist.ReduceOp.SUM)
            total_psnr = local_results[0].item()
            total_ssim_sum = local_results[1].item()
            num_samples = int(local_results[2].item())
        
        # Compute final metrics
        avg_psnr = total_psnr / max(1, num_samples)
        avg_ssim_manual = total_ssim_sum / max(1, num_samples)
        
        # Get the torchmetrics SSIM (should be similar to manual)
        avg_ssim = self.ssim_metric.compute().item()
        
        if self.is_main_process:
            print(f"\n  📈 Validation Results:")
            print(f"     PSNR: {avg_psnr:.2f} dB")
            print(f"     SSIM: {avg_ssim:.4f} (manual: {avg_ssim_manual:.4f})")
        
        return {'psnr': avg_psnr, 'ssim': avg_ssim}
    
    def _save_image_grid(self, images: Dict[str, torch.Tensor], path: Path):
        """Save image grid for visualization"""
        # Arrange in 2x3 grid
        keys = ['frame0', 'frame1', 'target', 'pred', 'error', 'conf']
        imgs = []
        for k in keys:
            if k in images:
                img = images[k].clamp(0, 1)
                img = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
                imgs.append(img)
                
        if imgs:
            # Create grid
            h, w = imgs[0].shape[:2]
            grid = np.zeros((h*2, w*3, 3), dtype=np.uint8)
            for i, img in enumerate(imgs):
                row, col = i // 3, i % 3
                grid[row*h:(row+1)*h, col*w:(col+1)*w] = img
                
            Image.fromarray(grid).save(path)
            
    def _load_checkpoint(self, path: str):
        """Load checkpoint for resuming"""
        print(f"📂 Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle both DDP and non-DDP checkpoints
        state_dict = checkpoint['model_state']
        if self.is_distributed:
            # If loading a non-DDP checkpoint into DDP model
            if not any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
        else:
            # If loading a DDP checkpoint into non-DDP model
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if checkpoint.get('scheduler_state') and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state'])
            
        self.global_step = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_psnr = checkpoint.get('best_metric', 0)
        
        print(f"  Resumed from step {self.global_step}, epoch {self.epoch}")
        
    def train(self):
        """Main training loop"""
        if self.is_main_process:
            print("\n🚀 Starting training...\n")
        
        try:
            for epoch in range(self.epoch, self.config.epochs):
                self.epoch = epoch
                
                # Set sampler epoch for proper shuffling
                if self.is_distributed and self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)
                
                # Train epoch
                epoch_metrics = self.train_epoch()
                
                # Ensure all processes are ready for validation
                if self.is_distributed:
                    dist.barrier()
                
                # End of epoch validation - ALL processes participate
                val_metrics = self.validate()
                
                # Sync after validation
                if self.is_distributed:
                    dist.barrier()
                
                # Only main process logs and saves
                if self.is_main_process:
                    # Log epoch summary
                    print(f"\n📊 Epoch {epoch+1} Summary:")
                    print(f"  Train Loss: {epoch_metrics.get('total', 0):.4f}")
                    print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB")
                    print(f"  Val SSIM: {val_metrics['ssim']:.4f}")
                    print(f"  Best PSNR: {self.best_psnr:.2f} dB")
                    
                    # Save epoch checkpoint
                    is_best = val_metrics['psnr'] > self.best_psnr
                    if is_best:
                        self.best_psnr = val_metrics['psnr']
                    
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
            # Final save (only main process)
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
                    
                    # Cleanup resources
                    self.run_manager.close()
                    print(f"\n✅ Training complete! Results saved to: {self.run_manager.run_dir}")
            
            # Clean up distributed training AFTER everything else
            if self.is_distributed:
                cleanup_distributed_training()


# ===========================
# Entry Point
# ===========================

def main():
    parser = argparse.ArgumentParser(description="TEMPO v2 Training")
    
    # Basic settings
    parser.add_argument("--data_root", type=str, default="datasets/vimeo_triplet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # Model
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--temporal_channels", type=int, default=64)
    
    # Experiment
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--resume", type=str, default=None)
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compile", action="store_true", help="Use PyTorch 2.0 compile")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true", default=False)

    # Distributed training
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed data parallel training.")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        base_channels=args.base_channels,
        temporal_channels=args.temporal_channels,
        exp_name=args.exp_name,
        use_wandb=args.use_wandb,
        notes=args.notes,
        resume=args.resume,
        device=args.device,
        compile_model=args.compile,
        use_amp=args.amp,
        amp_dtype=args.amp_dtype,
        distributed=args.distributed,
    )
    
    # Start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()