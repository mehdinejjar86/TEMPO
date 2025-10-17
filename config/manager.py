# config/manager.py
import json
from datetime import datetime
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from config.default import TrainingConfig

# ===========================
# Training Infrastructure
# ===========================

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class RunManager:
    """Manages experiment directories and logging"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Create run directory
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_name = timestamp
        if config.exp_name:
            run_name = f"{timestamp}_{config.exp_name}"
            
        self.run_dir = Path("runs") / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.sample_dir = self.run_dir / "samples"
        self.log_dir = self.run_dir / "logs"
        
        for d in [self.checkpoint_dir, self.sample_dir, self.log_dir]:
            d.mkdir(exist_ok=True)
            
        # Save config
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)
            
        # Initialize loggers
        self.writer = SummaryWriter(self.log_dir)
        self.wandb_run = None
        
        # Lazy opt-in for wandb
        if self.config.use_wandb:
            try:
                import wandb
                self._wandb = wandb  # keep module on the instance
                self.wandb_run = self._wandb.init(
                    project=self.config.project_name,
                    name=str(self.run_dir.name),
                    config=self.config.to_dict(),
                    notes=self.config.notes,
                    dir=str(self.run_dir),
                )
            except ImportError:
                print("⚠️ wandb not installed; continuing without it.")
                self._wandb = None
                self.wandb_run = None
        else:
            self._wandb = None
            
        print(f"📁 Run directory: {self.run_dir}")
        
    def log_metrics(self, metrics: Dict, step: int, prefix: str = "train"):
        """Log metrics to all active loggers"""
        # TensorBoard
        for k, v in metrics.items():
            self.writer.add_scalar(f"{prefix}/{k}", v, step)
            
        # WandB
        if self.wandb_run:
            wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            wandb_metrics["step"] = step
            wandb.log(wandb_metrics)
            
    def log_images(self, images: Dict[str, torch.Tensor], step: int, prefix: str = "val"):
        """Log images to TensorBoard and WandB"""
        for name, img in images.items():
            # TensorBoard expects [N,C,H,W] or [C,H,W]
            if img.dim() == 4:
                img = img[0]  # Take first in batch
            self.writer.add_image(f"{prefix}/{name}", img.clamp(0, 1), step)
            
            # WandB
            if self.wandb_run:
                # Convert to PIL for WandB
                img_np = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
                wandb.log({f"{prefix}/{name}": wandb.Image(img_np)}, step=step)
                
    def save_checkpoint(self, model: nn.Module, optimizer, scheduler, 
                       step: int, epoch: int, best_metric: float, is_best: bool = False, is_main_process: bool = True):
        """Save training checkpoint"""

        if not is_main_process:
            return
        
        model_state = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()

        checkpoint = {
            'model_state': model_state,
            'step': step,
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'best_metric': best_metric,
            'config': self.config.to_dict()
        }
        
        # Regular checkpoint
        path = self.checkpoint_dir / f"checkpoint_step_{step:06d}.pth"
        torch.save(checkpoint, path)
        
        # Best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model (metric: {best_metric:.4f})")
            
        # Keep only last 5 regular checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pth"))
        for old_ckpt in checkpoints[:-5]:
            old_ckpt.unlink()
            
    def close(self):
        """Clean up loggers"""
        self.writer.close()
        if self.wandb_run:
            wandb.finish()