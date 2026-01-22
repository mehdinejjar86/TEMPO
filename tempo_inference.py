# confocal_tempo_eval.py
import cv2
import math
import sys
import torch
import numpy as np
import argparse
import os
import warnings
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

sys.path.append('.')
from model.tempo import build_tempo
VFIMAMBA_ROOT = "/home/groups/ChangLab/govindsa/confocal_project/datasets/benchmarking/VFIMamba"
sys.path.insert(0, VFIMAMBA_ROOT)
from benchmark.utils.pytorch_msssim import ssim_matlab


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True,
                   help='Path to TEMPO checkpoint file')
parser.add_argument('--data_path', type=str, required=True,
                   help='Path to confocal dataset (should contain tri_testlist.txt and sequences/)')
parser.add_argument('--device', type=str, default='cuda',
                   help='Device to run on (cuda/cpu)')

# Model architecture arguments (should match your training config)
parser.add_argument('--base_channels', type=int, default=64)
parser.add_argument('--temporal_channels', type=int, default=64)
parser.add_argument('--encoder_depths', type=int, nargs='+', default=[3, 3, 12, 3])
parser.add_argument('--decoder_depths', type=int, nargs='+', default=[3, 3, 3, 3])
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--num_points', type=int, default=4)
parser.add_argument('--use_cross_scale', action='store_true', default=True)

args = parser.parse_args()

# Build model with same architecture as training
print("🏗️ Building TEMPO model...")
model = build_tempo(
    base_channels=args.base_channels,
    temporal_channels=args.temporal_channels,
    encoder_depths=args.encoder_depths,
    decoder_depths=args.decoder_depths,
    num_heads=args.num_heads,
    num_points=args.num_points,
    use_cross_scale=args.use_cross_scale,
    use_checkpointing=False,  # Disable for inference
).to(args.device)

# Load checkpoint
print(f"📂 Loading checkpoint: {args.model_path}")
checkpoint = torch.load(args.model_path, map_location=args.device)

# Handle different checkpoint formats and DDP/compile prefixes
state_dict = checkpoint.get("model_state", checkpoint)

# Remove DDP wrapper prefixes if present
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# Remove torch.compile prefixes if present  
if any("_orig_mod" in k for k in state_dict.keys()):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "")
        new_state_dict[new_k] = v
    state_dict = new_state_dict

missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print(f"⚠️  Missing keys: {missing}")
if unexpected:
    print(f"⚠️  Unexpected keys: {unexpected}")

model.eval()
print(f"✅ Model loaded successfully")

# Print model info
total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"📊 Model: {total_params:.2f}M parameters")

print(f'=========================Starting testing=========================')
print(f'Dataset: Confocal TEMPO   Device: {args.device}')

# Load dataset
path = args.data_path
tri_testlist = os.path.join(path, 'tri_testlist.txt')

if not os.path.exists(tri_testlist):
    raise FileNotFoundError(f"tri_testlist.txt not found at {tri_testlist}")

with open(tri_testlist, 'r') as f:
    test_sequences = [line.strip() for line in f if line.strip()]

psnr_list, ssim_list = [], []

print(f"Found {len(test_sequences)} test sequences")

for seq_name in test_sequences:
    if len(seq_name) <= 1:
        continue
        
    # Load images (same structure as Vimeo90K)
    im1_path = os.path.join(path, 'sequences', seq_name, 'im1.png')
    im2_path = os.path.join(path, 'sequences', seq_name, 'im2.png') 
    im3_path = os.path.join(path, 'sequences', seq_name, 'im3.png')
    
    # Check if files exist
    if not all(os.path.exists(p) for p in [im1_path, im2_path, im3_path]):
        print(f"Skipping {seq_name} - missing files")
        continue
    
    # Load images
    I0 = cv2.imread(im1_path)  # First frame
    I1 = cv2.imread(im2_path)  # Ground truth (middle)
    I2 = cv2.imread(im3_path)  # Last frame
    
    if I0 is None or I1 is None or I2 is None:
        print(f"Skipping {seq_name} - failed to load images")
        continue
    
    # Convert to tensors [1, 2, 3, H, W] for TEMPO (N=2 frames)
    frames = torch.stack([
        torch.tensor(I0.transpose(2, 0, 1)).float() / 255.0,  # im1
        torch.tensor(I2.transpose(2, 0, 1)).float() / 255.0,  # im3
    ], dim=0).unsqueeze(0).to(args.device)  # [1, 2, 3, H, W]
    
    # Anchor times for interpolation: [0.0, 1.0] (start and end)
    anchor_times = torch.tensor([[0.0, 1.0]]).to(args.device)  # [1, 2]
    
    # Target time: 0.5 (middle frame)
    target_time = torch.tensor([0.5]).to(args.device)  # [1]
    
    # Run TEMPO inference
    with torch.no_grad():
        pred, aux = model(frames, anchor_times, target_time)
    
    # ⚠️ REMOVED CLAMPING - Match VFIMamba evaluation
    # pred = pred.clamp(0, 1)  # REMOVED THIS LINE
    
    # Convert ground truth to tensor (match VFIMamba format exactly)
    gt_tensor = torch.tensor(I1.transpose(2, 0, 1)).to(args.device).unsqueeze(0) / 255.0
    
    # Calculate SSIM on unclamped predictions (like VFIMamba)
    ssim_score = ssim_matlab(gt_tensor, pred.unsqueeze(0) if pred.dim() == 3 else pred).detach().cpu().numpy()
    
    # Convert to numpy for PSNR calculation (no clamping)
    pred_np = pred[0].detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    gt_np = I1.astype(np.float32) / 255.0  # [H, W, 3]
    
    # Calculate PSNR (match VFIMamba - no epsilon added)
    mse = ((pred_np - gt_np) ** 2).mean()
    psnr = -10 * math.log10(mse)
    
    psnr_list.append(psnr)
    ssim_list.append(ssim_score)
    
    # Print running average (match VFIMamba format)
    print("Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))

print("="*50)
print("FINAL RESULTS:")
print(f"Total sequences processed: {len(psnr_list)}")
print(f"Final Average PSNR: {float(np.mean(psnr_list)):.4f} dB")
print(f"Final Average SSIM: {float(np.mean(ssim_list)):.4f}")
print("="*50)

# Additional statistics
if len(psnr_list) > 0:
    print(f"\nDetailed Statistics:")
    print(f"PSNR - Min: {np.min(psnr_list):.3f}, Max: {np.max(psnr_list):.3f}, Std: {np.std(psnr_list):.3f}")
    print(f"SSIM - Min: {np.min(ssim_list):.4f}, Max: {np.max(ssim_list):.4f}, Std: {np.std(ssim_list):.4f}")