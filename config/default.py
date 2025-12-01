# config/default.py
from dataclasses import dataclass, asdict
from typing import Optional, Dict

@dataclass
class TrainingConfig:
    # Model
    base_channels: int = 64
    temporal_channels: int = 64
    attn_heads: int = 4
    attn_points: int = 4

    # Data
    data_root: str = "datasets/vimeo_triplet"
    batch_size: int = 4
    num_workers: int = 8
    crop_size: Optional[int] = None

    # Training
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Mixed precision
    use_amp: bool = False
    amp_dtype: str = "fp32"  # "fp16" or "bf16"

    # Scheduling
    lr_scheduler: str = "cosine"
    warmup_steps: int = 1000

    # Logging
    use_wandb: bool = False
    log_interval: int = 50
    val_interval: int = 1000
    save_interval: int = 5000
    n_val_samples: int = 8

    # Loss weights (override defaults)
    loss_config: Optional[Dict] = None

    # Resume
    resume: Optional[str] = None

    # Hardware
    device: str = "cuda"
    compile_model: bool = False

    # Experiment
    exp_name: Optional[str] = None
    project_name: str = "TEMPO-v2"
    notes: str = ""

    # Distributed Training
    distributed: bool = False

    def to_dict(self):
        d = asdict(self)
        if d['loss_config'] is None:
            d['loss_config'] = {}
        return d
