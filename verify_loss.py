import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model.loss.tempo_loss import build_tempo_loss

def test_uncertainty_loss():
    print("🧪 Testing TEMPO Uncertainty Loss...\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Test with uncertainty (default)
    print("\n[1] Testing with uncertainty=True")
    loss_fn = build_tempo_loss(use_uncertainty=True).to(device)
    
    # Check if parameters are learnable
    print("  Checking learnable parameters...")
    has_params = any(p.requires_grad for p in loss_fn.parameters())
    if has_params:
        print("  ✓ Loss has learnable parameters")
    else:
        print("  ❌ Loss has NO learnable parameters!")
        
    # Create dummy inputs
    B, N, C, H, W = 2, 3, 3, 64, 64
    pred = torch.randn(B, C, H, W, device=device, requires_grad=True)
    target = torch.randn(B, C, H, W, device=device)
    frames = torch.randn(B, N, C, H, W, device=device)
    anchor_times = torch.linspace(0, 1, N).unsqueeze(0).expand(B, N).to(device)
    target_time = torch.full((B,), 0.5, device=device)
    
    aux = {
        'weights': torch.softmax(torch.randn(B, N, device=device), dim=1),
        'conf_map': torch.sigmoid(torch.randn(B, 1, H, W, device=device)),
        'attn_entropy': torch.rand(B, 1, H, W, device=device),
        'fallback_mask': torch.zeros(B, device=device)
    }
    
    # Forward pass
    print("  Running forward pass...")
    loss, metrics = loss_fn(pred, target, frames, anchor_times, target_time, aux)
    
    print(f"  ✓ Loss value: {loss.item():.4f}")
    print("  ✓ Metrics returned:", list(metrics.keys()))
    
    # Check for uncertainty metrics
    uncertainty_keys = [k for k in metrics.keys() if 'uncertainty/' in k]
    if uncertainty_keys:
        print(f"  ✓ Found uncertainty metrics: {uncertainty_keys}")
    else:
        print("  ❌ No uncertainty metrics found!")
        
    # Backward pass
    print("  Running backward pass...")
    loss.backward()
    if pred.grad is not None:
        print("  ✓ Gradients computed for prediction")
    else:
        print("  ❌ No gradients for prediction!")
        
    # Check gradients for loss parameters
    loss_grads = [p.grad for p in loss_fn.parameters() if p.requires_grad]
    if any(g is not None for g in loss_grads):
         print("  ✓ Gradients computed for loss parameters")
    else:
         print("  ❌ No gradients for loss parameters!")

    # 2. Test without uncertainty (fixed weights)
    print("\n[2] Testing with uncertainty=False")
    loss_fn_fixed = build_tempo_loss(use_uncertainty=False).to(device)
    
    # Forward pass
    loss_fixed, metrics_fixed = loss_fn_fixed(pred, target, frames, anchor_times, target_time, aux)
    print(f"  ✓ Loss value: {loss_fixed.item():.4f}")
    
    uncertainty_keys_fixed = [k for k in metrics_fixed.keys() if 'uncertainty/' in k]
    if not uncertainty_keys_fixed:
        print("  ✓ No uncertainty metrics (expected)")
    else:
        print(f"  ❌ Found uncertainty metrics unexpectedly: {uncertainty_keys_fixed}")

    print("\n✅ Loss verification complete!")

if __name__ == "__main__":
    test_uncertainty_loss()
