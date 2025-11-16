# TEMPO: Temporal Multi-View Frame Synthesis

**TEMPO** is a deep learning model for video frame interpolation and extrapolation using time-aware pyramid attention. The model can generate intermediate frames (interpolation) or predict future/past frames (extrapolation) from a sequence of input frames.

## Features

- **Time-Aware Attention**: Explicit temporal modeling through learned position encodings
- **Multi-Scale Processing**: 4-level pyramid architecture for capturing features at different scales
- **Adaptive Temporal Weighting**: Bimodal weighting system that adapts to different motion patterns
- **Scene Cut Detection**: Built-in robustness for handling scene transitions
- **Deep Feature Extraction**: 28+ residual blocks with Squeeze-and-Excitation attention
- **Multi-Frame Support**: Works with 2+ anchor frames for flexible synthesis

## Architecture Overview

### Key Components

1. **Temporal Position Encoding**: Sinusoidal encoding with learnable projection for representing temporal relationships
2. **Frame Encoder**: Deep CNN with 4 scales (full, 1/2, 1/4, 1/8 resolution) and SE attention
3. **Time-Aware Pyramid Attention**: Sliding window deformable attention with temporal bias
4. **Temporal Weighter**: Adaptive softmax weighting over anchor frames
5. **Frame Decoder**: Progressive upsampling decoder with skip connections
6. **Scene Cut Detector**: Automatic detection and fallback for scene transitions

### Model Statistics

- **Parameters**: ~77M (base_channels=64)
- **Input**: N frames at arbitrary timestamps
- **Output**: Single interpolated/extrapolated frame
- **Scales**: 4 pyramid levels (1x, 1/2, 1/4, 1/8)
- **ResBlocks**: 28 total (4 + 6 + 6 + 12 across scales)

## Installation

### Requirements

```bash
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=9.5.0

# Training
tqdm>=4.65.0
tensorboard>=2.13.0

# Optional
wandb>=0.15.0  # For experiment tracking
lpips>=0.1.4   # For perceptual loss
```

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd tempo

# Install dependencies
pip install torch torchvision numpy pillow tqdm tensorboard

# Optional: Install WandB for logging
pip install wandb

# Optional: Install LPIPS for perceptual loss
pip install lpips
```

### Dataset Preparation

Download the Vimeo-90K triplet dataset:

```bash
# Create dataset directory
mkdir -p datasets/vimeo_triplet

# Download and extract (replace with actual download link)
wget http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip
unzip vimeo_triplet.zip -d datasets/vimeo_triplet/

# Expected structure:
# datasets/vimeo_triplet/
# ├── sequences/
# │   ├── 00001/
# │   │   ├── 0001/
# │   │   │   ├── im1.png
# │   │   │   ├── im2.png
# │   │   │   └── im3.png
# │   │   └── ...
# │   └── ...
# ├── tri_trainlist.txt
# └── tri_testlist.txt
```

## Quick Start

### Testing the Model

```bash
# Run smoke test to verify installation
python smoke_test.py

# Run comprehensive test
python test_v3_upgrade.py
```

Expected output:
```
✨ FrameEncoder initialized
✨ DecoderTimeFiLM initialized
✅ ALL TESTS PASSED
```

### Basic Training

```bash
# Train on Vimeo-90K with default settings
python train_tempo.py \
    --data_root datasets/vimeo_triplet \
    --batch_size 2 \
    --epochs 100 \
    --lr 5e-5
```

### Recommended Training (with Mixed Precision)

```bash
python train_tempo.py \
    --data_root datasets/vimeo_triplet \
    --batch_size 2 \
    --lr 5e-5 \
    --epochs 100 \
    --amp \
    --amp_dtype bf16 \
    --exp_name "tempo_baseline" \
    --use_wandb
```

### Distributed Training (Multi-GPU)

```bash
torchrun --nproc_per_node=2 train_tempo.py \
    --distributed \
    --batch_size 2 \
    --amp \
    --amp_dtype bf16 \
    --exp_name "tempo_ddp"
```

## Training Configuration

### Key Hyperparameters

```python
# Model architecture
base_channels: 64          # Base feature channels
temporal_channels: 64      # Temporal encoding dimension
attn_heads: 4             # Attention heads in pyramid
attn_points: 4            # Sampling points per head

# Training
batch_size: 2             # Batch size (reduce if OOM)
learning_rate: 5e-5       # Lower than typical for stability
epochs: 100               # Training epochs
warmup_steps: 2000        # LR warmup steps
grad_clip: 0.5            # Gradient clipping

# Mixed precision
use_amp: True             # Enable automatic mixed precision
amp_dtype: "bf16"         # bf16 or fp16

# Loss weights (auto-scheduled during training)
w_l1: 0.8                 # L1 reconstruction
w_ssim: 1.0               # Structural similarity
w_perceptual: 0.05        # LPIPS perceptual loss
w_freq_struct: 0.3        # Frequency domain structure
w_coherence: 0.2          # Temporal coherence
```

### Memory Requirements

| Resolution | Batch Size | GPU Memory | Recommendation |
|------------|------------|------------|----------------|
| 256x256    | 4          | ~12 GB     | RTX 3090, A5000 |
| 256x256    | 2          | ~8 GB      | RTX 3080 |
| 448x256    | 2          | ~10 GB     | RTX 3090 |
| 512x512    | 1          | ~10 GB     | RTX 3090 |

**Tips:**
- Enable mixed precision (`--amp --amp_dtype bf16`) for 30-40% memory reduction
- Use gradient accumulation if needed (modify `train_tempo.py`)
- Reduce `base_channels` to 48 for lighter model

## Project Structure

```
tempo/
├── model/
│   ├── tempo.py              # Main TEMPO model
│   ├── film.py               # Encoder/decoder with temporal fusion
│   ├── pyramid.py            # Time-aware pyramid attention
│   ├── temporal.py           # Temporal encoding and weighting
│   ├── utility.py            # Utility modules (cut detector, etc.)
│   └── loss/
│       └── tempo_loss.py     # Multi-component loss function
├── data/
│   └── data_vimeo_triplet.py # Vimeo-90K data loader
├── config/
│   ├── default.py            # Training configuration
│   ├── manager.py            # Run management and logging
│   └── dpp.py                # Distributed training setup
├── train_tempo.py            # Main training script
├── smoke_test.py             # Quick functionality test
└── test_v3_upgrade.py        # Comprehensive test suite
```

## Usage Examples

### 1. Frame Interpolation (2 frames → 1 middle frame)

```python
import torch
from model.tempo import build_tempo

# Build model
model = build_tempo(
    base_channels=64,
    temporal_channels=64,
    attn_heads=4,
    attn_points=4
)

# Prepare input
frames = torch.rand(1, 2, 3, 256, 256)  # [B, N, C, H, W]
anchor_times = torch.tensor([[0.0, 1.0]])  # Frame timestamps
target_time = torch.tensor([0.5])  # Target timestamp (middle)

# Forward pass
pred, aux = model(frames, anchor_times, target_time)
# pred: [1, 3, 256, 256] - interpolated frame
# aux: dict with weights, confidence, entropy, etc.
```

### 2. Forward Extrapolation (predict future)

```python
# Input: frames at t=0, t=1
# Output: frame at t=2 (one step ahead)
anchor_times = torch.tensor([[0.0, 1.0]])
target_time = torch.tensor([2.0])

pred, aux = model(frames, anchor_times, target_time)
```

### 3. Backward Extrapolation (predict past)

```python
# Input: frames at t=0, t=1
# Output: frame at t=-1 (one step back)
anchor_times = torch.tensor([[0.0, 1.0]])
target_time = torch.tensor([-1.0])

pred, aux = model(frames, anchor_times, target_time)
```

### 4. Multi-frame Input (3+ frames)

```python
# Use 3 frames for better temporal context
frames = torch.rand(1, 3, 3, 256, 256)  # 3 anchor frames
anchor_times = torch.tensor([[0.0, 0.4, 1.0]])
target_time = torch.tensor([0.5])

pred, aux = model(frames, anchor_times, target_time)
```

## Model Outputs

The model returns two values:

1. **pred**: `[B, 3, H, W]` - Synthesized RGB frame in [0, 1] range
2. **aux**: Dictionary containing:
   - `weights`: `[B, N]` - Temporal weighting over anchor frames
   - `conf_map`: `[B, 1, H, W]` - Per-pixel confidence scores
   - `attn_entropy`: `[B, 1, H, W]` - Attention uncertainty map
   - `cut_score`: `[B]` - Scene cut detection score
   - `fallback_mask`: `[B]` - Binary mask for scene cut fallback

## Loss Function

TEMPO uses a comprehensive multi-component loss:

### Core Reconstruction Losses

- **Charbonnier (L1)**: Robust L1 loss with learned epsilon per scale
- **MS-SSIM**: Multi-scale structural similarity
- **Frequency Loss**: Structure and phase preservation in frequency domain
- **Perceptual Loss**: LPIPS (optional, requires `lpips` package)

### Temporal Losses

- **Temporal Coherence**: Consistency with time-aware baseline
- **Occlusion Handling**: Uses attention entropy and confidence

### Regularization

- **Weight Regularization**: Entropy, smoothness, coverage, sparsity
- **Confidence Target**: Encourages confident predictions

### Loss Scheduling

Losses are automatically scheduled during training:
- Warmup for regularizers (first 5K steps)
- Delayed perceptual loss (after 2K steps)
- Ramped temporal coherence (5K-15K steps)

## Evaluation Metrics

The model tracks multiple metrics during training:

- **PSNR (dB)**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **MSE**: Mean Squared Error
- **Perceptual Distance**: LPIPS score (if enabled)

### Expected Performance (Vimeo-90K)

| Metric | Target |
|--------|--------|
| PSNR   | 34-35 dB |
| SSIM   | 0.95-0.96 |

Training typically requires 50-100 epochs to reach optimal performance.

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir runs/

# Navigate to http://localhost:6006
```

Logged metrics:
- Training/validation loss curves
- PSNR and SSIM over time
- Validation sample visualizations
- Learning rate schedule

### WandB (optional)

```bash
# Login to WandB
wandb login

# Enable in training
python train_tempo.py --use_wandb --exp_name "my_experiment"
```

## Checkpointing

Checkpoints are automatically saved during training:

- **Regular checkpoints**: Every 5,000 steps (configurable)
- **Best model**: Saved when validation PSNR improves
- **Resume training**: Use `--resume path/to/checkpoint.pth`

### Checkpoint Contents

- Model state dict
- Optimizer state
- Learning rate scheduler state
- Training step and epoch
- Best metric so far
- Full configuration

### Resuming Training

```bash
python train_tempo.py \
    --resume runs/2025_01_15_12_30_45_experiment/checkpoints/best_model.pth \
    --epochs 150
```

## Advanced Configuration

### Custom Loss Weights

Edit `config/default.py` or pass via command line:

```python
loss_config = {
    "w_l1": 0.8,
    "w_ssim": 1.0,
    "w_perceptual": 0.1,  # Increase for better perceptual quality
    "w_freq_struct": 0.3,
    "w_coherence": 0.15,
    "w_occlusion": 0.5,
}
```

### Model Variants

```python
# Lighter model (faster, less accurate)
model = build_tempo(
    base_channels=48,
    temporal_channels=48
)

# Heavier model (slower, more accurate)
model = build_tempo(
    base_channels=96,
    temporal_channels=96
)
```

### Data Augmentation

The Vimeo-90K loader supports:
- Random cropping: `crop_size=256`
- Horizontal flipping: `aug_flip=True`
- Mixed training modes: `mode="mix"` (interpolation + extrapolation)

## Troubleshooting

### CUDA Out of Memory

```bash
# Solution 1: Reduce batch size
--batch_size 1

# Solution 2: Enable mixed precision
--amp --amp_dtype bf16

# Solution 3: Reduce resolution (edit data loader)
crop_size=128

# Solution 4: Use gradient checkpointing (requires code modification)
```

### Training is Too Slow

```bash
# Enable mixed precision (30-40% speedup)
--amp --amp_dtype bf16

# Enable PyTorch 2.0 compile (10-20% speedup)
--compile

# Use multiple GPUs
torchrun --nproc_per_node=2 train_tempo.py --distributed
```

### NaN Loss

```bash
# Lower learning rate
--lr 2e-5

# Increase warmup
# Edit config/default.py: warmup_steps = 3000

# Tighter gradient clipping
# Edit config/default.py: grad_clip = 0.5

# Disable mixed precision temporarily
# Remove --amp flag
```

### Model Not Improving

**Check:**
1. Learning rate (try 2e-5 or 5e-5)
2. Loss weights (may need tuning)
3. Validation samples (are they getting sharper?)
4. Training curves (smooth or noisy?)

**Try:**
- Longer warmup: `warmup_steps: 3000`
- Different LR scheduler: `lr_scheduler: "step"`
- Adjust loss weights in `config/default.py`

## Citation

If you use TEMPO in your research, please cite:

```bibtex
@article{tempo2025,
  title={TEMPO: Temporal Multi-View Frame Synthesis with Time-Aware Pyramid Attention},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

## Acknowledgments

- Vimeo-90K dataset: Tianfan Xue et al.
- Inspired by techniques from RCAN, EDSR, IFRNet, and other SOTA methods
- Built with PyTorch

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: [your-email@example.com]

---

**Last Updated**: January 2025  
**PyTorch Version**: 2.0+  
**Python Version**: 3.8+
