# TEMPO: Temporal Multi-View Frame Synthesis

<p align="center">
  <img src="assets/tempo_banner.png" alt="TEMPO Banner" width="800"/>
</p>

<p align="center">
  <b>Rethinking Video Frame Interpolation as Temporal View Synthesis</b>
</p>

<p align="center">
  <a href="#key-innovations">Innovations</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

**TEMPO** introduces a paradigm shift in video frame interpolation by treating it as **temporal view synthesis** rather than optical flow estimation. Inspired by multi-view 3D reconstruction (NeRF), TEMPO synthesizes novel temporal views by learning to fuse observations from arbitrary timestamps.

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

## Key Innovations

| Feature | Description |
|---------|-------------|
| **Temporal View Synthesis** | Treats frames as temporal observations of a scene, not sources for pixel warping |
| **Arbitrary N-Frame Input** | Supports 2, 4, 8+ input frames—more observations = better synthesis |
| **Continuous Timestamps** | Interpolate or extrapolate to any point in time |
| **Deformable Temporal Attention** | Correlation-guided offsets for implicit motion handling |
| **Cross-Scale Guidance** | Coarse-to-fine refinement prevents hallucination artifacts |
| **No Explicit Flow** | Avoids flow estimation failures on fast motion and occlusions |

### Why Synthesis > Flow?

| Scenario | Flow-Based | TEMPO |
|----------|-----------|-------|
| Small motion | ✅ Excellent | ✅ Good |
| Large/fast motion | ❌ Flow breaks | ✅ Attention reaches far |
| Occlusions | ❌ Requires inpainting | ✅ Naturally synthesized |
| Motion blur | ❌ No clear matches | ✅ Generates from features |
| N>2 frames | ❌ Limited to 2 | ✅ Fuses all observations |

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
│  │         Depths: [3, 3, 9, 3]  ~12M params               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           Deformable Temporal Attention                  │    │
│  │   • Correlation Pyramid (4-level feature matching)       │    │
│  │   • Learned offsets per head (4) × points (4)           │    │
│  │   • Scaled by temporal distance |Δt|                    │    │
│  │   • Confidence from attention entropy                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Cross-Scale Guidance                        │    │
│  │         8C → 4C → 2C → C (coarse-to-fine)               │    │
│  │         Prevents hallucination artifacts                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               NAFNet Decoder                             │    │
│  │         Skip connections from encoder                    │    │
│  │         Depths: [3, 3, 3, 3]  ~8M params                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│  Output: Synthesized frame [B, 3, H, W] + confidence map        │
│                                                                  │
│  Total Parameters: ~22-25M                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

**1. Sinusoidal Time Encoding**
```python
# Continuous timestamps → feature space
t = 0.5  # target time
enc = sin/cos positional encoding → MLP projection
# Enables arbitrary temporal queries
```

**2. Correlation Pyramid**
```python
# Multi-scale feature matching guides attention offsets
# Similar to RAFT correlation volume, but for temporal fusion
for level in [1x, 1/2x, 1/4x, 1/8x]:
    corr = dot_product(query_features, value_features)
# Provides motion guidance without explicit flow
```

**3. Deformable Temporal Attention**
```python
# Query: "What should exist at target time t?"
# Keys/Values: Features from all N observations
# Offsets: Learned per-head sampling locations (correlation-guided)
# Output: Temporally fused features for synthesis
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/TEMPO.git
cd TEMPO

# Create environment
conda create -n tempo python=3.10
conda activate tempo

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Requirements File

```txt
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pillow>=9.0.0
tqdm>=4.64.0
tensorboard>=2.11.0
wandb>=0.13.0  # optional
torchmetrics>=0.11.0
einops>=0.6.0
```

## Usage

### Quick Start

```python
import torch
from model.tempo import build_tempo

# Build model
model = build_tempo(
    base_channels=64,
    temporal_channels=64,
    encoder_depths=[3, 3, 9, 3],
    decoder_depths=[3, 3, 3, 3],
    attn_heads=4,
    attn_points=4,
)

# Input: N frames at arbitrary timestamps
frames = torch.randn(1, 4, 3, 256, 256)  # [B, N, C, H, W]
timestamps = torch.tensor([[0.0, 0.33, 0.67, 1.0]])  # [B, N]
target_time = torch.tensor([0.5])  # [B]

# Synthesize target frame
output, aux = model(frames, timestamps, target_time)
# output: [1, 3, 256, 256]
# aux: {weights, confidence, entropy, ...}
```

### Training

**Single GPU:**
```bash
python train_tempo.py \
    --data_root datasets/vimeo_triplet \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --amp \
    --amp_dtype bf16
```

**Multi-GPU (DDP):**
```bash
torchrun --nproc_per_node=4 train_tempo.py \
    --data_root datasets/vimeo_triplet \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --amp \
    --amp_dtype bf16 \
    --distributed
```

**Resume Training:**
```bash
python train_tempo.py \
    --resume runs/tempo_exp/checkpoints/latest.pt \
    --data_root datasets/vimeo_triplet
```

### Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--base_channels` | 64 | Base channel dimension |
| `--temporal_channels` | 64 | Temporal encoding dimension |
| `--encoder_depths` | [3,3,9,3] | ConvNeXt encoder depths |
| `--decoder_depths` | [3,3,3,3] | NAFNet decoder depths |
| `--attn_heads` | 4 | Attention heads |
| `--attn_points` | 4 | Deformable sampling points per head |
| `--batch_size` | 4 | Batch size per GPU |
| `--lr` | 1e-4 | Learning rate |
| `--epochs` | 100 | Training epochs |
| `--amp` | False | Enable automatic mixed precision |
| `--amp_dtype` | fp32 | AMP dtype (fp16, bf16, fp32) |

### Inference

```python
import torch
from model.tempo import build_tempo
from PIL import Image
import torchvision.transforms as T

# Load model
model = build_tempo()
model.load_state_dict(torch.load('checkpoints/tempo_best.pt')['model_state'])
model.eval().cuda()

# Load frames
transform = T.Compose([T.ToTensor()])
frame0 = transform(Image.open('frame0.png')).unsqueeze(0)
frame1 = transform(Image.open('frame1.png')).unsqueeze(0)

# Stack frames
frames = torch.stack([frame0, frame1], dim=1).cuda()  # [1, 2, 3, H, W]
timestamps = torch.tensor([[0.0, 1.0]]).cuda()

# Interpolate at t=0.5
with torch.no_grad():
    output, _ = model(frames, timestamps, torch.tensor([0.5]).cuda())

# Save result
T.ToPILImage()(output[0].cpu()).save('interpolated.png')
```

### Multi-Frame Interpolation (TEMPO's Advantage)

```python
# Use 4 frames instead of 2
frames = torch.stack([frame0, frame1, frame2, frame3], dim=1)
timestamps = torch.tensor([[0.0, 0.33, 0.67, 1.0]])

# More observations = better synthesis
# Flow-based methods can't leverage this!
output, aux = model(frames, timestamps, target_time=torch.tensor([0.5]))

# Check confidence (higher with more frames)
print(f"Confidence: {aux['confidence'].mean():.3f}")
```

### Smoke Test

```bash
# Verify installation and model
python smoke_test.py
```

## Dataset Preparation

### Vimeo-90K Triplet

```bash
# Download
wget http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip
unzip vimeo_triplet.zip -d datasets/

# Structure
datasets/vimeo_triplet/
├── sequences/
│   ├── 00001/
│   │   ├── 0001/
│   │   │   ├── im1.png
│   │   │   ├── im2.png
│   │   │   └── im3.png
│   │   └── ...
│   └── ...
├── tri_trainlist.txt
└── tri_testlist.txt
```

### Custom Dataset

Implement a dataset that returns:
```python
{
    'frames': torch.Tensor,      # [N, 3, H, W]
    'timestamps': torch.Tensor,  # [N]
    'target': torch.Tensor,      # [3, H, W]
    'target_time': torch.Tensor, # [1]
}
```

## Results

### Vimeo-90K Triplet (N=2)

| Method | PSNR ↑ | SSIM ↑ | Parameters |
|--------|--------|--------|------------|
| RIFE | 35.61 | 0.978 | 10.4M |
| AMT-S | 35.84 | 0.979 | 12.3M |
| EMA-VFI | 36.12 | 0.980 | 21.5M |
| **TEMPO** | **TBD** | **TBD** | 22.5M |

### Multi-Frame Advantage (N=4)

| Method | N=2 | N=4 | N=8 |
|--------|-----|-----|-----|
| Flow-based | ✅ | ❌ N/A | ❌ N/A |
| **TEMPO** | ✅ | ✅ +X dB | ✅ +Y dB |

*Results on N>2 coming soon—this is where TEMPO's architecture provides unique advantages.*

### Challenging Scenarios

| Scenario | Flow-Based | TEMPO |
|----------|-----------|-------|
| SNU-FILM Extreme | Degrades | Robust |
| Large displacement | Flow failure | Attention handles |
| Occlusion-heavy | Inpainting artifacts | Natural synthesis |

## Model Zoo

| Model | PSNR | Params | Download |
|-------|------|--------|----------|
| TEMPO-S | TBD | ~15M | Coming soon |
| TEMPO-B | TBD | ~22M | Coming soon |
| TEMPO-L | TBD | ~40M | Coming soon |

## Project Structure

```
TEMPO/
├── model/
│   ├── tempo.py                    # Main model
│   ├── temporal_view_synthesis.py  # Deformable attention + fusion
│   ├── convnext_nafnet.py          # Encoder + Decoder
│   ├── temporal.py                 # Temporal weighting
│   └── loss/
│       └── tempo_loss.py           # Multi-component loss
├── data/
│   └── data_vimeo_triplet.py       # Dataset loader
├── config/
│   ├── default.py                  # Training config
│   ├── manager.py                  # Run management
│   └── dpp.py                      # Distributed training utils
├── train_tempo.py                  # Training script
├── smoke_test.py                   # Quick verification
├── requirements.txt
└── README.md
```

## Technical Details

### Loss Function

```python
Total Loss = λ₁·Charbonnier + λ₂·Perceptual + λ₃·Frequency + λ₄·Census

# Charbonnier: Smooth L1 for reconstruction
# Perceptual: VGG features for visual quality
# Frequency: FFT domain for sharpness
# Census: Edge preservation
```

### Training Schedule

- **Epochs 1-5**: Reconstruction loss only (warm-up)
- **Epochs 5-20**: Add perceptual loss
- **Epochs 20+**: Full loss with frequency and census
- **LR Schedule**: Cosine annealing to 1e-6

### Augmentation

- Random horizontal flip
- Random crop (256×256 default)
- Temporal reversal (bidirectional training)

## FAQ

**Q: Why not use optical flow at all?**

A: Flow estimation fundamentally assumes pixel correspondence exists. For occluded regions, there is no correspondence—flow must guess, and inpainting must fill. TEMPO sidesteps this by synthesizing directly from features, where the decoder learns to generate plausible content.

**Q: How does TEMPO handle large motion?**

A: The deformable attention offsets are guided by a correlation pyramid (similar to RAFT) but scaled by temporal distance. Large |Δt| allows larger offsets. Additionally, cross-scale guidance ensures coarse-level features (which capture global motion) inform fine-level synthesis.

**Q: Can TEMPO extrapolate?**

A: Yes! Set `target_time` outside the range of `anchor_times`. The continuous time encoding and attention mechanism naturally support extrapolation, though quality degrades for far extrapolation.

**Q: Why is N>2 important?**

A: With N=2, TEMPO has the same information as flow-based methods. With N>2, TEMPO can:
- See occluded regions from other viewpoints
- Use sharper frames when some have motion blur  
- Build higher confidence through redundant observations

This is the key architectural advantage.

## Acknowledgements

- [NAFNet](https://github.com/megvii-research/NAFNet) for the decoder architecture
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) for the encoder design
- [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) for inspiration and baselines
- [NeRF](https://www.matthewtancik.com/nerf) for the multi-view synthesis philosophy

## Citation

```bibtex
@article{tempo2024,
  title={TEMPO: Temporal Multi-View Frame Synthesis},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>TEMPO</b>: Frames are observations. Synthesis is reconstruction. Time is just another dimension.
</p>
