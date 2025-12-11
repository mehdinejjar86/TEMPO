import warnings
warnings.filterwarnings("ignore", message="Arguments other than a weight enum")

import torch
from torch.optim import AdamW
from torch.amp import autocast

from model.tempo import build_tempo
from model.loss.tempo_loss import tempo_loss


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.pow(2).sum().item()
    return total ** 0.5


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )

    # Check for bfloat16 support
    use_bf16 = torch.amp.autocast_mode.is_autocast_available(device_type=device.type)
    print(f"Device: {device}")
    print(f"Using bfloat16: {use_bf16}")

    print("\n" + "="*60)
    print("TEMPO: Temporal Multi-View Frame Synthesis")
    print("="*60)
    
    # Build model with new simplified API
    model = build_tempo(
        base_channels=64,
        temporal_channels=64,
        encoder_depths=[3, 3, 9, 3],
        decoder_depths=[3, 3, 3, 3],
        attn_heads=4,
    ).to(device)
    model.train()

    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # --------------------------
    # 1) Interpolation (N=4): predict in the middle
    # --------------------------
    print("\n[Test 1] Interpolation with N=4 frames")
    B, N, H, W = 1, 4, 768, 768
    frames = torch.rand(B, N, 3, H, W, device=device)
    anchor_times = torch.tensor([[0.0, 0.3, 0.7, 1.0]] * B, device=device)
    target_time = torch.tensor([0.5] * B, device=device)
    target_rgb = torch.rand(B, 3, H, W, device=device)

    opt.zero_grad(set_to_none=True)

    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        out, aux = model(frames, anchor_times, target_time)
        print(f"  Output: {out.shape}, dtype: {out.dtype}")
        print(f"  Weights: {aux['weights'][0].cpu().numpy().round(3)}")
        print(f"  Confidence: mean={aux['confidence'].mean():.3f}")
        print(f"  Entropy: mean={aux['entropy'].mean():.3f}")
        loss, logs = tempo_loss(out, target_rgb, aux, anchor_times, target_time, frames=frames)

    loss.backward()
    gnorm = grad_norm(model)
    opt.step()

    print(f"  Loss: {loss.item():.4f}, Grad norm: {gnorm:.3f}")
    print(f"  L1: {logs.get('l1', 0):.4f}, SSIM: {logs.get('ssim', 0):.4f}, PSNR: {logs.get('psnr', 0):.2f}")

    # --------------------------
    # 2) Forward extrapolation (N=2): predict one step ahead
    # --------------------------
    print("\n[Test 2] Extrapolation with N=2 frames")
    B, N, H, W = 2, 2, 256, 256
    frames = torch.rand(B, N, 3, H, W, device=device)
    anchor_times = torch.tensor([[0.0, 1.0]] * B, device=device)
    target_time = torch.tensor([1.5] * B, device=device)  # Extrapolate
    target_rgb = torch.rand(B, 3, H, W, device=device)

    opt.zero_grad(set_to_none=True)

    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        out, aux = model(frames, anchor_times, target_time)
        print(f"  Output: {out.shape}, dtype: {out.dtype}")
        print(f"  Weights: {aux['weights'][0].cpu().numpy().round(3)}")
        loss, logs = tempo_loss(out, target_rgb, aux, anchor_times, target_time, frames=frames)

    loss.backward()
    gnorm = grad_norm(model)
    opt.step()

    print(f"  Loss: {loss.item():.4f}, Grad norm: {gnorm:.3f}")
    print(f"  L1: {logs.get('l1', 0):.4f}, SSIM: {logs.get('ssim', 0):.4f}, PSNR: {logs.get('psnr', 0):.2f}")

    # --------------------------
    # 3) Many frames (N=8): test scalability
    # --------------------------
    print("\n[Test 3] Many observations N=8")
    B, N, H, W = 1, 8, 128, 128
    frames = torch.rand(B, N, 3, H, W, device=device)
    anchor_times = torch.linspace(0, 1, N, device=device).unsqueeze(0).expand(B, -1)
    target_time = torch.tensor([0.45] * B, device=device)
    target_rgb = torch.rand(B, 3, H, W, device=device)

    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        out, aux = model(frames, anchor_times, target_time)
        print(f"  Output: {out.shape}")
        print(f"  Weights: {aux['weights'][0].cpu().numpy().round(3)}")

    # --------------------------
    # 4) Model Summary
    # --------------------------
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    enc_params = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    fus_params = sum(p.numel() for p in model.fusion.parameters()) / 1e6
    dec_params = sum(p.numel() for p in model.decoder.parameters()) / 1e6

    print(f"  Encoder (ConvNeXt):      {enc_params:.2f}M")
    print(f"  Fusion (CrossAttention): {fus_params:.2f}M")
    print(f"  Decoder (NAFNet):        {dec_params:.2f}M")
    print(f"  Total:                   {total_params:.2f}M")

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
