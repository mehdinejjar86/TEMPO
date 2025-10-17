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
from dataclasses import asdict

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
    # Only print on the main process to avoid clutter
    if os.environ.get('RANK', '0') == '0':
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
        # Setup run manager only on the main process
        self.run_manager = RunManager(config) if self.is_main_process else None
        
        if self.is_main_process:
            print(f"🧪 AMP: use_autocast={self.use_autocast}, device={self.amp_device_type}, dtype={self.amp_dtype}, scaler={'yes' if self.scaler else 'no'}")
        
        # Build model
        if self.is_main_process: print("🏗️ Building model...")
        self.model = build_tempo(
            base_channels=config.base_channels,
            temporal_channels=config.temporal_channels,
            attn_heads=config.attn_heads,
            attn_points=config.attn_points
        ).to(self.device)

        # --- Wrap model for DDP if enabled ---
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.device.index])
        
        if config.compile_model and hasattr(torch, 'compile'):
            if self.is_main_process: print("  ⚡ Compiling model with PyTorch 2.0...")
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
        if self.is_main_process: print("📊 Loading datasets...")
        
        # Training
        train_dataset = Vimeo90KTriplet(
            root=self.config.data_root,
            split="train",
            mode="interp",
            crop_size=self.config.crop_size,
            aug_flip=True,
        )

        self.train_sampler = None
        if self.is_distributed:
            self.train_sampler = DistributedSampler(train_dataset)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            # **[FIXED]** Shuffle is False when a sampler is used
            shuffle=(self.train_sampler is None),
            num_workers=self.config.num_workers,
            collate_fn=vimeo_collate,
            pin_memory=True,
            drop_last=True,
            # **[FIXED]** The sampler must be passed to the DataLoader
            sampler=self.train_sampler
        )
        
        # Validation
        val_dataset = Vimeo90KTriplet(
            root=self.config.data_root,
            split="test",
            mode="interp",
            crop_size=None,
            center_crop_eval=False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=vimeo_collate,
            pin_memory=True
        )
        
        if self.is_main_process:
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
        
        # **[FIXED]** Correctly handle progress bar for all processes
        iterable = self.train_loader
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
            
            if self.global_step < self.config.warmup_steps:
                lr_scale = (self.global_step + 1) / self.config.warmup_steps
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.config.learning_rate * lr_scale
                    
            self.loss_scheduler.update(self.global_step)

            with autocast(self.amp_device_type, dtype=self.amp_dtype, enabled=self.use_autocast):
                pred, aux = self.model(frames, anchor_times, target_time)
                loss, metrics = self.loss_fn(pred, target, frames, anchor_times, target_time, aux)

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                
            if self.lr_scheduler and self.global_step >= self.config.warmup_steps:
                self.lr_scheduler.step()
            
            metric_tracker.update(metrics)
            
            if self.is_main_process:
                metrics['lr'] = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{metrics['total']:.4f}",
                    'psnr': f"{metrics.get('psnr', 0):.2f}",
                    'lr': f"{metrics['lr']:.1e}"
                })
            
                if self.global_step % self.config.log_interval == 0:
                    avg_metrics = metric_tracker.get_averages()
                    if self.run_manager: self.run_manager.log_metrics(avg_metrics, self.global_step, "train")
                    metric_tracker.reset()
                
                if self.global_step > 0 and self.global_step % self.config.val_interval == 0:
                    val_metrics = self.validate()
                    if self.run_manager: self.run_manager.log_metrics(val_metrics, self.global_step, "val")
                    
                    current_psnr = val_metrics.get('psnr', 0)
                    is_best = current_psnr > self.best_psnr
                    if is_best: self.best_psnr = current_psnr
                    self.model.train()
                
                if self.global_step > 0 and self.global_step % self.config.save_interval == 0:
                    self._save_checkpoint(is_best=False)
                
            self.global_step += 1
            
        return metric_tracker.get_averages()
    
    def _save_checkpoint(self, is_best: bool):
        if not self.is_main_process: return
        
        # Unwrap the model from DDP to save the original state dict
        model_state = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        
        if self.run_manager:
            self.run_manager.save_checkpoint(
                model_state, self.optimizer, self.lr_scheduler,
                self.global_step, self.epoch, self.best_psnr,
                is_best=is_best
            )

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        # ... (This method is only called by the main process, so no changes needed)
        self.model.eval()
        total_psnr, total_ssim, num_samples = 0.0, 0.0, 0
        
        sample_indices = np.linspace(0, len(self.val_loader)-1, self.config.n_val_samples, dtype=int)
        
        # In DDP, self.model is a wrapper, so we access the actual model via .module
        model_to_eval = self.model.module if self.is_distributed else self.model

        val_pbar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            desc="Validating", unit="sample", ncols=100, leave=False, colour="green"
        )
        
        for idx, (frames, anchor_times, target_time, target) in val_pbar:
            frames, anchor_times, target_time, target = (
                frames.to(self.device), anchor_times.to(self.device), 
                target_time.to(self.device), target.to(self.device)
            )
            
            pred, aux = model_to_eval(frames, anchor_times, target_time)
            pred = pred.clamp(0, 1)
            
            mse = F.mse_loss(pred, target)
            psnr = -10 * torch.log10(mse)
            total_psnr += psnr.item()
            num_samples += 1
            
            if self.run_manager and idx in sample_indices:
                viz_dict = {
                    'frame0': frames[0, 0], 'frame1': frames[0, 1], 'target': target[0],
                    'pred': pred[0], 'error': (pred[0] - target[0]).abs().mean(0, True).repeat(3,1,1),
                    'conf': aux['conf_map'][0].repeat(3,1,1)
                }
                self.run_manager.log_images(viz_dict, self.global_step, "val")
                
            val_pbar.set_postfix({'psnr': f"{psnr.item():.2f}"})
            
        avg_psnr = total_psnr / max(1, num_samples)
        print(f"\n  📈 Validation: PSNR={avg_psnr:.2f}dB")
        
        return {'psnr': avg_psnr}

    def train(self):
        """Main training loop"""
        if self.is_main_process: print("\n🚀 Starting training...\n")
        
        try:
            for epoch in range(self.epoch, self.config.epochs):
                self.epoch = epoch
                if self.is_distributed:
                    self.train_loader.sampler.set_epoch(epoch)
                
                epoch_metrics = self.train_epoch()
                
                if self.is_main_process:
                    val_metrics = self.validate()
                    print(f"\n📊 Epoch {epoch+1} Summary: Train Loss={epoch_metrics.get('total', 0):.4f}, Val PSNR={val_metrics['psnr']:.2f}dB, Best PSNR={self.best_psnr:.2f}dB")
                    
                    is_best = val_metrics['psnr'] > self.best_psnr
                    if is_best: self.best_psnr = val_metrics['psnr']
                    self._save_checkpoint(is_best=is_best)
                        
        except KeyboardInterrupt:
            if self.is_main_process: print("\n\n⚠️ Training interrupted by user")
        except Exception as e:
            if self.is_main_process: print(f"\n\n❌ Training failed: {e}")
            raise
        finally:
            # **[FIXED]** Cleanup should happen only once at the very end
            if self.is_main_process:
                print("\n💾 Saving final checkpoint...")
                self._save_checkpoint(is_best=False)
                if self.run_manager: self.run_manager.close()
                print(f"\n✅ Training complete! Results saved to: {self.run_manager.run_dir}")
            
            cleanup_distributed_training()

# ===========================
# Entry Point
# ===========================

def main():
    parser = argparse.ArgumentParser(description="TEMPO v2 Training")
    # ... (same as your original main function)
    args = parser.parse_args()
    config = TrainingConfig(...)
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()