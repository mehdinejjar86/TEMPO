# TEMPO: Temporal Multi-View Frame Synthesis

**Rethinking Video Frame Interpolation as Temporal View Synthesis**

*Author: Mehdi Nejjar*

---

## Overview

**TEMPO** introduces a paradigm shift in video frame interpolation by treating it as **temporal view synthesis** rather than optical flow estimation. Inspired by multi-view 3D reconstruction, TEMPO synthesizes novel temporal views by learning to fuse observations from arbitrary timestamps.

Unlike flow-based methods that warp pixels and require separate inpainting for disoccluded regions, TEMPO **generates** the target frame directly from temporally-fused features—naturally handling occlusions, large motions, and complex scene dynamics.

```
Traditional Flow-Based:
  Frame₀ → Estimate Flow → Warp Pixels → Blend → Inpaint Holes → Output
  Frame₁ → Estimate Flow → Warp Pixels ↗

TEMPO (Temporal View Synthesis):
  Frame₀ ─→ Encode ─→ Features ─┐
  Frame₁ ─→ Encode ─→ Features ─┼─→ Temporal Attention ─→ Decode ─→ Synthesize
  FrameN ─→ Encode ─→ Features ─┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Temporal View Synthesis** | Treats frames as temporal observations, not sources for pixel warping |
| **Arbitrary N-Frame Input** | Supports 2, 4, 8+ input frames—more observations = better synthesis |
| **Continuous Timestamps** | Interpolate or extrapolate to any point in time |
| **Deformable Temporal Attention** | Correlation-guided offsets for implicit motion handling |
| **Cross-Scale Guidance** | Coarse-to-fine refinement prevents hallucination artifacts |
| **Uncertainty-Aware Loss** | Learnable task weighting + pixel-level confidence |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         TEMPO Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: N frames [B, N, 3, H, W] + timestamps [B, N]            │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              ConvNeXt Encoder (per frame)                │    │
│  │         AdaLN-Zero temporal conditioning                 │    │
│  │         Outputs: 4 scales (C, 2C, 4C, 8C)               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           Deformable Temporal Attention                  │    │
│  │   • Correlation Pyramid (4-level feature matching)       │    │
│  │   • Learned offsets per head × points                   │    │
│  │   • Scaled by temporal distance |Δt|                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Cross-Scale Guidance                        │    │
│  │         8C → 4C → 2C → C (coarse-to-fine)               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               NAFNet Decoder                             │    │
│  │         Skip connections + uncertainty prediction        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│  Output: Synthesized frame [B, 3, H, W] + confidence map        │
│                                                                  │
│  Total Parameters: ~25M                                          │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone repository
git clone https://github.com/mehdinejjar/TEMPO.git
cd TEMPO

# Create environment
conda create -n tempo python=3.10
conda activate tempo

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio

# Install dependencies
pip install tqdm pillow numpy torchmetrics wandb tensorboard
```

## Quick Start

```python
import torch
from model.tempo import build_tempo

# Build model
model = build_tempo(
    base_channels=64,
    temporal_channels=64,
    encoder_depths=[3, 3, 18, 3],
    decoder_depths=[3, 3, 9, 3],
    num_heads=4,
    num_points=4,
    predict_uncertainty=True,
)

# Input: N frames at arbitrary timestamps
frames = torch.randn(1, 4, 3, 256, 256)  # [B, N, C, H, W]
timestamps = torch.tensor([[0.0, 0.33, 0.67, 1.0]])  # [B, N]
target_time = torch.tensor([0.5])  # [B]

# Synthesize target frame
output, aux = model(frames, timestamps, target_time)
# output: [1, 3, 256, 256]
# aux: {weights, confidence, entropy, log_var, ...}
```

## Training

**Standard training:**
```bash
python train_tempo.py \
    --data_root datasets/vimeo_triplet \
    --batch_size 4 \
    --epochs 100 \
    --amp --amp_dtype bf16
```

**With progressive training (coarse-to-fine):**
```bash
python train_tempo.py \
    --data_root datasets/vimeo_triplet \
    --progressive \
    --amp --amp_dtype bf16
```

**Without uncertainty loss:**
```bash
python train_tempo.py \
    --data_root datasets/vimeo_triplet \
    --no_uncertainty \
    --batch_size 4 \
    --amp --amp_dtype bf16
```

**Multi-GPU (DDP):**
```bash
torchrun --nproc_per_node=4 train_tempo.py \
    --data_root datasets/vimeo_triplet \
    --batch_size 4 \
    --distributed \
    --amp --amp_dtype bf16
```

### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | datasets/vimeo_triplet | Path to dataset |
| `--batch_size` | 2 | Batch size per GPU |
| `--lr` | 1e-4 | Learning rate |
| `--epochs` | 100 | Training epochs |
| `--amp` | off | Enable mixed precision |
| `--amp_dtype` | fp32 | AMP dtype (fp16, bf16) |
| `--progressive` | off | Enable progressive training |
| `--no_uncertainty` | off | Disable uncertainty loss |
| `--resume` | None | Checkpoint to resume from |
| `--use_wandb` | off | Enable W&B logging |

## Smoke Test

```bash
python smoke_test.py
```

## Dataset

### Vimeo-90K Triplet

```bash
# Download from: http://toflow.csail.mit.edu/
wget http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip
unzip vimeo_triplet.zip -d datasets/
```

Expected structure:
```
datasets/vimeo_triplet/
├── sequences/
│   └── XXXXX/XXXX/{im1.png, im2.png, im3.png}
├── tri_trainlist.txt
└── tri_testlist.txt
```

## Project Structure

```
TEMPO/
├── model/
│   ├── tempo.py              # Main model
│   ├── temporal_attention.py # Deformable attention + fusion
│   ├── convnext_nafnet.py    # Encoder + Decoder
│   ├── temporal.py           # Time encoding + weighting
│   └── loss/
│       └── tempo_loss.py     # Uncertainty-aware loss
├── data/
│   └── data_vimeo_triplet.py # Dataset loader
├── config/
│   ├── default.py            # Training config
│   ├── manager.py            # Run management
│   └── dpp.py                # Distributed utils
├── train_tempo.py            # Training script
├── smoke_test.py             # Quick verification
└── README.md
```

## License

MIT License

---

*TEMPO: Frames are observations. Synthesis is reconstruction. Time is just another dimension.*
