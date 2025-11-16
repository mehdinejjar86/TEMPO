# config/manager.py
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Set
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

class CSVLogger:
    """Simple CSV logger for metrics with dynamic field support"""
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.file = None
        self.writer = None
        self.fieldnames = ['step', 'prefix']  # Start with base fields
        self.seen_fields: Set[str] = set()
        self.rows_buffer = []  # Buffer rows until we know all fields
        self.initialized = False
        
    def log(self, metrics: Dict, step: int, prefix: str = "train"):
        """Log metrics to CSV"""
        # Add step and prefix to metrics
        row = {"step": step, "prefix": prefix}
        row.update({f"{k}": v for k, v in metrics.items()})
        
        # Track all fields we've seen
        current_fields = set(row.keys())
        new_fields = current_fields - self.seen_fields
        
        if new_fields:
            self.seen_fields.update(current_fields)
            # If we haven't initialized yet, just buffer
            if not self.initialized:
                self.rows_buffer.append(row)
                return
            else:
                # Need to reinitialize with new fields
                self._reinitialize()
        
        # Write row
        if self.initialized:
            # Fill in missing fields with empty string
            row_with_all_fields = {field: row.get(field, '') for field in self.fieldnames}
            self.writer.writerow(row_with_all_fields)
            self.file.flush()
        else:
            self.rows_buffer.append(row)
    
    def _initialize(self):
        """Initialize the CSV file with all known fields"""
        self.fieldnames = sorted(list(self.seen_fields))
        # Ensure step and prefix are first
        if 'step' in self.fieldnames:
            self.fieldnames.remove('step')
        if 'prefix' in self.fieldnames:
            self.fieldnames.remove('prefix')
        self.fieldnames = ['step', 'prefix'] + sorted([f for f in self.fieldnames if f not in ['step', 'prefix']])
        
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        
        # Write buffered rows
        for row in self.rows_buffer:
            row_with_all_fields = {field: row.get(field, '') for field in self.fieldnames}
            self.writer.writerow(row_with_all_fields)
        
        self.file.flush()
        self.rows_buffer.clear()
        self.initialized = True
    
    def _reinitialize(self):
        """Reinitialize CSV with new fields (read existing, rewrite with new columns)"""
        # Close current file
        if self.file:
            self.file.close()
        
        # Read existing data
        existing_rows = []
        if self.filepath.exists():
            with open(self.filepath, 'r', newline='') as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
        
        # Update fieldnames
        self.fieldnames = sorted(list(self.seen_fields))
        if 'step' in self.fieldnames:
            self.fieldnames.remove('step')
        if 'prefix' in self.fieldnames:
            self.fieldnames.remove('prefix')
        self.fieldnames = ['step', 'prefix'] + sorted([f for f in self.fieldnames if f not in ['step', 'prefix']])
        
        # Reopen and write all data
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        
        # Write existing rows with new fields
        for row in existing_rows:
            row_with_all_fields = {field: row.get(field, '') for field in self.fieldnames}
            self.writer.writerow(row_with_all_fields)
        
        self.file.flush()
    
    def finalize(self):
        """Finalize the CSV (write buffered data if not initialized)"""
        if not self.initialized and self.rows_buffer:
            self._initialize()
    
    def close(self):
        """Close the CSV file"""
        self.finalize()
        if self.file is not None:
            self.file.close()


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
        
        # CSV loggers for train and validation
        self.csv_train = CSVLogger(self.log_dir / "train_metrics.csv")
        self.csv_val = CSVLogger(self.log_dir / "val_metrics.csv")
        
        # Combined CSV with all metrics
        self.csv_all = CSVLogger(self.log_dir / "all_metrics.csv")
        
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
        print(f"📊 CSV logs:")
        print(f"   - Training: {self.log_dir / 'train_metrics.csv'}")
        print(f"   - Validation: {self.log_dir / 'val_metrics.csv'}")
        print(f"   - All metrics: {self.log_dir / 'all_metrics.csv'}")
        
    def log_metrics(self, metrics: Dict, step: int, prefix: str = "train"):
        """Log metrics to all active loggers"""
        # TensorBoard
        for k, v in metrics.items():
            self.writer.add_scalar(f"{prefix}/{k}", v, step)
        
        # CSV logging
        if prefix == "train":
            self.csv_train.log(metrics, step, prefix)
        elif prefix == "val":
            self.csv_val.log(metrics, step, prefix)
        
        # Log to combined CSV
        self.csv_all.log(metrics, step, prefix)
            
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
                       step: int, epoch: int, best_metric: float, is_best: bool = False, is_main_process: bool = True, is_distributed: bool = False):
        """Save training checkpoint"""

        if not is_main_process:
            return
        
        model_state = model.module.state_dict() if is_distributed else model.state_dict()

        checkpoint = {
            'model_state': model_state,
            'step': step,
            'epoch': epoch,
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
        self.csv_train.close()
        self.csv_val.close()
        self.csv_all.close()
        if self.wandb_run:
            wandb.finish()
