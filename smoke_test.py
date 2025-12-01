import torch
from torch.optim import AdamW
from torch.amp import autocast # Import autocast

from model.tempo import build_tempo
from model.loss.tempo_loss import tempo_loss # wrapper we added

def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.pow(2).sum().item()
    return (total ** 0.5)

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # --- [NEW] Check for bfloat16 support ---
    use_bf16 = torch.amp.autocast_mode.is_autocast_available(device_type=device.type)
    print(f"Device: {device}")
    print(f"Using bfloat16: {use_bf16}")

    print("\nTesting TEMPO – forward & backward …")
    model = build_tempo(base_channels=64, temporal_channels=64,
                        attn_heads=4, attn_points=4, attn_levels_max=4,
                        window_size=8, shift_size=4, dt_bias_gain=1.25,
                        max_offset_scale=1.5, cut_thresh=0.4).to(device)
    model.train()

    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # --------------------------
    # 1) Interpolation (N=2): predict the middle
    # anchors: t0=0, t1=1 → target at 0.5
    # --------------------------
    B, N, H, W = 1, 4, 768, 768 
    frames = torch.rand(B, N, 3, H, W, device=device)
    anchor_times = torch.tensor([[0.0, 0.4, 0.8, 1.0]] * B, device=device)
    target_time = torch.tensor([0.5] * B, device=device)
    target_rgb = torch.rand(B, 3, H, W, device=device)

    opt.zero_grad(set_to_none=True)

    # --- [NEW] Use autocast for mixed precision ---
    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        out, aux = model(frames, anchor_times, target_time)
        print(f"[interp] ✓ Output: {out.shape}, dtype: {out.dtype}")
        print(f"[interp] ✓ Weights[0]: {aux['weights'][0].detach().cpu().numpy()}")
        loss, logs = tempo_loss(out, target_rgb, aux, anchor_times, target_time, frames=frames)

    loss.backward()
    gnorm = grad_norm(model)
    opt.step()

    print(f"[interp] loss={loss.item():.4f} grad_norm={gnorm:.3f}  "
      f"l1={logs.get('l1', 0):.4f} ssim={logs.get('ssim', 0):.4f} perc={logs.get('perceptual', 0):.4f}")

    # --------------------------
    # 2) Forward extrapolation (N=2): predict one step ahead
    # anchors: t0=0, t1=1 → target at 2.0
    # --------------------------
    print("-" * 20)
    B, N, H, W = 2, 2, 256, 256 # Using a smaller size to avoid OOM
    frames = torch.rand(B, N, 3, H, W, device=device)
    anchor_times = torch.tensor([[0.0, 1.0]] * B, device=device)
    target_time = torch.tensor([2.0] * B, device=device)
    target_rgb = torch.rand(B, 3, H, W, device=device)

    opt.zero_grad(set_to_none=True)

    # --- [NEW] Use autocast for mixed precision ---
    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        out, aux = model(frames, anchor_times, target_time)
        print(f"[extra-fwd] ✓ Output: {out.shape}, dtype: {out.dtype}")
        print(f"[extra-fwd] ✓ Weights[0]: {aux['weights'][0].detach().cpu().numpy()}")
        loss, logs = tempo_loss(out, target_rgb, aux, anchor_times, target_time, frames=frames)
    
    loss.backward()
    gnorm = grad_norm(model)
    opt.step()

    print(f"[extra-fwd] loss={loss.item():.4f} grad_norm={gnorm:.3f}  "
      f"l1={logs.get('l1', 0):.4f} ssim={logs.get('ssim', 0):.4f} perc={logs.get('perceptual', 0):.4f}")

    print("\nAll good ✅")