# config/default.py
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List


@dataclass
class TrainingConfig:
    # ==========================
    # Model Architecture
    # ==========================
    base_channels: int = 64
    temporal_channels: int = 64
    
    # Encoder (ConvNeXt)
    encoder_depths: List[int] = field(default_factory=lambda: [3, 3, 12, 3])
    
    # Decoder (NAFNet)
    decoder_depths: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    
    # Fusion (Deformable Temporal Attention)
    num_heads: int = 8
    num_points: int = 4
    use_cross_scale: bool = True

    # ==========================
    # Data
    # ==========================
    data_root: str = "datasets/vimeo_triplet"
    batch_size: int = 4
    num_workers: int = 8
    crop_size: Optional[int] = None

    # ==========================
    # Training
    # ==========================
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Mixed precision
    use_amp: bool = False
    amp_dtype: str = "fp32"  # "fp16", "bf16", or "fp32"

    # Gradient checkpointing (trade compute for memory)
    use_checkpointing: bool = True  # Enables ~40% memory reduction during training

    # Scheduling
    lr_scheduler: str = "cosine"
    warmup_steps: int = 1000

    # ==========================
    # Logging
    # ==========================
    use_wandb: bool = False
    log_interval: int = 50
    val_interval: int = 1000
    save_interval: int = 5000
    n_val_samples: int = 8

    # Loss (override defaults)
    loss_config: Optional[Dict] = None

    # ==========================
    # Resume
    # ==========================
    resume: Optional[str] = None

    # ==========================
    # Hardware
    # ==========================
    device: str = "cuda"
    compile_model: bool = False

    # ==========================
    # Experiment
    # ==========================
    exp_name: Optional[str] = None
    project_name: str = "TEMPO"
    notes: str = ""

    # ==========================
    # Distributed
    # ==========================
    distributed: bool = False

    def to_dict(self):
        d = asdict(self)
        if d['loss_config'] is None:
            d['loss_config'] = {}
        return d
