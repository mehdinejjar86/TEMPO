# train_tempo.py
"""
TEMPO v2 Training System
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
        else:
            self.device = torch.device(config.device)

        # AMP setup (unified API)
        self.amp_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.amp_dtype = torch.bfloat16 if config.amp_dtype.lower() == "bf16" else torch.float16 if config.amp_dtype.lower() == "fp16" else torch.float32
        self.use_autocast = (
            config.use_amp and autocast_mode.is_autocast_available(self.amp_device_type)
        )

        # Scaler only for CUDA + fp16 (bf16/CPU don’t need scaling)
        if self.use_autocast and self.amp_device_type == "cuda" and self.amp_dtype == torch.float16:
            self.scaler = GradScaler(device_type="cuda")
        else:
            self.scaler = None
        # Setup run manager
        self.run_manager = RunManager(config)
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
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_psnr = 0.0
        
        # Resume if specified
        if config.resume:
            self._load_checkpoint(config.resume)
            
        # Print model info
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
            mode="interp",
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
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Full resolution validation
            shuffle=False,
            num_workers=4,
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
        
        # Beautiful progress bar
        if self.is_main_process:
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {self.epoch+1}/{self.config.epochs}",
                unit="batch",
                ncols=160,
                bar_format="{l_bar}{bar:30}{r_bar}",
                colour="cyan"
            )
        else:
            pbar = self.train_loader
        
        for batch_idx, (frames, anchor_times, target_time, target) in enumerate(pbar):
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
                    
            # Forward pass
            # Update loss schedule for *this* step
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
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['total']:.4f}",
                'l1': f"{metrics.get('l1', 0):.3f}",
                'ssim': f"{metrics.get('ssim', 0):.3f}",
                'psnr': f"{metrics.get('psnr', 0):.2f}",
                'lr': f"{metrics['lr']:.1e}"
            })
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                avg_metrics = metric_tracker.get_averages()
                self.run_manager.log_metrics(avg_metrics, self.global_step, "train")
                metric_tracker.reset()
                
            # Validation
            if self.global_step % self.config.val_interval == 0 and self.global_step > 0:
                val_metrics = self.validate()
                self.run_manager.log_metrics(val_metrics, self.global_step, "val")
                
                # Check if best
                current_psnr = val_metrics.get('psnr', 0)
                is_best = current_psnr > self.best_psnr
                if is_best:
                    self.best_psnr = current_psnr
                    
                # Resume training mode
                self.model.train()
                
            # Checkpointing
            if self.global_step % self.config.save_interval == 0 and self.global_step > 0:
                self.run_manager.save_checkpoint(
                    self.model, self.optimizer, self.lr_scheduler,
                    self.global_step, self.epoch, self.best_psnr,
                    is_best=False,
                    is_main_process=self.is_main_process

                )
                
            self.global_step += 1
            
        return metric_tracker.get_averages()
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation with sample generation"""
        self.model.eval()
        
        total_psnr = 0.0
        total_ssim = 0.0
        num_samples = 0
        
        # Select samples to visualize
        sample_indices = np.linspace(0, len(self.val_loader)-1, 
                                    self.config.n_val_samples).astype(int)
        samples_to_save = []
        
        # Progress bar for validation
        val_pbar = tqdm(
            enumerate(self.val_loader),
            total=min(100, len(self.val_loader)),  # Cap at 100 for speed
            desc="Validating",
            unit="sample",
            ncols=100,
            leave=False,
            colour="green"
        )
        
        for idx, (frames, anchor_times, target_time, target) in val_pbar:
            if idx >= 100:  # Limit validation samples
                break
                
            frames = frames.to(self.device, non_blocking=True)
            anchor_times = anchor_times.to(self.device, non_blocking=True)
            target_time = target_time.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            # Forward pass
            pred, aux = self.model(frames, anchor_times, target_time)
            pred = pred.clamp(0, 1)
            
            # Compute metrics
            mse = F.mse_loss(pred, target)
            psnr = -10 * torch.log10(mse)
            
            # Simple SSIM approximation (for speed)
            mu_p, mu_t = pred.mean(dim=[2,3]), target.mean(dim=[2,3])
            var_p = ((pred - mu_p.view(-1,3,1,1))**2).mean(dim=[2,3])
            var_t = ((target - mu_t.view(-1,3,1,1))**2).mean(dim=[2,3])
            cov = ((pred - mu_p.view(-1,3,1,1)) * (target - mu_t.view(-1,3,1,1))).mean(dim=[2,3])
            
            C1, C2 = 0.01**2, 0.03**2
            ssim = ((2*mu_p*mu_t + C1) * (2*cov + C2)) / ((mu_p**2 + mu_t**2 + C1) * (var_p + var_t + C2))
            ssim = ssim.mean()
            
            total_psnr += psnr.item()
            total_ssim += ssim.item()
            num_samples += 1
            
            # Save samples
            if idx in sample_indices:
                # Create visualization grid
                viz_dict = {
                    'frame0': frames[0, 0],
                    'frame1': frames[0, 1],
                    'target': target[0],
                    'pred': pred[0],
                    'error': (pred[0] - target[0]).abs().mean(dim=0, keepdim=True).repeat(3,1,1),
                    'conf': aux['conf_map'][0].repeat(3,1,1)
                }
                
                # Log to tensorboard/wandb
                self.run_manager.log_images(viz_dict, self.global_step, "val")
                
                # Save to disk
                save_path = self.run_manager.sample_dir / f"step_{self.global_step:06d}_sample_{idx:03d}.png"
                self._save_image_grid(viz_dict, save_path)
                
            val_pbar.set_postfix({'psnr': f"{psnr.item():.2f}", 'ssim': f"{ssim.item():.4f}"})
            
        avg_psnr = total_psnr / max(1, num_samples)
        avg_ssim = total_ssim / max(1, num_samples)
        
        print(f"\n  📈 Validation: PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.4f}")
        
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
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if checkpoint.get('scheduler_state') and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state'])
            
        self.global_step = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_psnr = checkpoint.get('best_metric', 0)
        
        print(f"  Resumed from step {self.global_step}, epoch {self.epoch}")
        
    def train(self):
        """Main training loop"""
        print("\n🚀 Starting training...\n")
        
        try:
            for epoch in range(self.epoch, self.config.epochs):
                self.epoch = epoch
                # --- Set sampler epoch for proper shuffling ---
                if self.is_distributed:
                    self.train_loader.sampler.set_epoch(epoch)
                # Train epoch
                epoch_metrics = self.train_epoch()
                
                # End of epoch validation
                # --- Only main process validates and saves ---
                if self.is_main_process:
                    val_metrics = self.validate()
                    
                    # Log epoch summary
                    print(f"\n📊 Epoch {epoch+1} Summary:")
                    print(f"  Train Loss: {epoch_metrics.get('total', 0):.4f}")
                    print(f"  Val PSNR: {val_metrics['psnr']:.2f}dB")
                    print(f"  Val SSIM: {val_metrics['ssim']:.4f}")
                    print(f"  Best PSNR: {self.best_psnr:.2f}dB")
                    
                    # Save epoch checkpoint
                    is_best = val_metrics['psnr'] > self.best_psnr
                    if is_best:
                        self.best_psnr = val_metrics['psnr']
                        
                    self.run_manager.save_checkpoint(
                        self.model, self.optimizer, self.lr_scheduler,
                        self.global_step, epoch, self.best_psnr,
                        is_best=is_best
                    )

                cleanup_distributed_training() # Clean up at the end of training
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Training interrupted by user")
            
        except Exception as e:
            print(f"\n\n❌ Training failed: {e}")
            raise
            
        finally:
            # Final save
            print("\n💾 Saving final checkpoint...")
            self.run_manager.save_checkpoint(
                self.model, self.optimizer, self.lr_scheduler,
                self.global_step, self.epoch, self.best_psnr,
                is_best=False
            )
            
            # Cleanup
            self.run_manager.close()
            print(f"\n✅ Training complete! Results saved to: {self.run_manager.run_dir}")


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