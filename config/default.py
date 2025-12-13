# config/default.py
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List


@dataclass
class TrainingConfig:
    # Model Architecture
    base_channels: int = 64
    temporal_channels: int = 64
    encoder_depths: List[int] = field(default_factory=lambda: [3, 3, 18, 3])  # Deeper stage3
    decoder_depths: List[int] = field(default_factory=lambda: [3, 3, 9, 3])   # Deeper stage2
    num_heads: int = 4
    num_points: int = 4  # Deformable sampling points per head
    use_cross_scale: bool = True

    # Data
    data_root: str = "datasets/vimeo_triplet"
    batch_size: int = 4
    num_workers: int = 8
    crop_size: Optional[int] = None

    # Progressive Training (Phase 4)
    # Note: Vimeo90K is 448×256, so max crop is 256
    # Disabled by default - enable with --progressive flag
    progressive_training: bool = False
    progressive_crops: List[int] = field(default_factory=lambda: [128, 192, 256])
    progressive_epochs: List[int] = field(default_factory=lambda: [30, 60, 100])  # When to switch
    progressive_batch_sizes: List[int] = field(default_factory=lambda: [16, 8, 4])
    progressive_lrs: List[float] = field(default_factory=lambda: [2e-4, 1e-4, 5e-5])

    # Training
    epochs: int = 100
    learning_rate: float = 2e-4  # Higher initial for progressive
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    use_amp: bool = False
    amp_dtype: str = "fp32"
    lr_scheduler: str = "cosine"
    warmup_steps: int = 1000

    # Logging
    use_wandb: bool = False
    log_interval: int = 50
    val_interval: int = 1000
    save_interval: int = 5000
    n_val_samples: int = 8
    loss_config: Optional[Dict] = None

    # Uncertainty (Phase 4 - Learnable Loss Weighting)
    use_uncertainty: bool = True
    predict_uncertainty: bool = True  # Decoder predicts pixel uncertainty

    # Resume
    resume: Optional[str] = None

    # Hardware
    device: str = "cuda"
    compile_model: bool = False

    # Experiment
    exp_name: Optional[str] = None
    project_name: str = "TEMPO"
    notes: str = ""
    distributed: bool = False

    def to_dict(self):
        d = asdict(self)
        if d['loss_config'] is None:
            d['loss_config'] = {}
        return d
