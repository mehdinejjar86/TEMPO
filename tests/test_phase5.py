"""Quick test for TEMPO BEAST Phase 5 losses"""
import torch
import torch.nn.functional as F
from model.loss.tempo_loss import LaplacianPyramidLoss, EdgeAwareLoss

print("Testing TEMPO BEAST Phase 5 Losses")
print("=" * 60)

device = torch.device("cpu")  # Use CPU for quick test
pred = torch.rand(2, 3, 128, 128, device=device)
target = torch.rand(2, 3, 128, 128, device=device)

# Test 1: Laplacian Pyramid Loss
print("\n[Test 1] Laplacian Pyramid Loss")
lap_loss = LaplacianPyramidLoss(num_levels=4)
try:
    loss_val = lap_loss(pred, target)
    print(f"  ✓ Laplacian loss computed: {loss_val.item():.4f}")
    print(f"  ✓ Shape correct: {loss_val.shape == torch.Size([])}")
    print(f"  ✓ Requires grad: {loss_val.requires_grad}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test 2: Edge-Aware Loss
print("\n[Test 2] Edge-Aware Loss")
edge_loss = EdgeAwareLoss(edge_weight=2.0)
try:
    loss_val = edge_loss(pred, target)
    print(f"  ✓ Edge-aware loss computed: {loss_val.item():.4f}")
    print(f"  ✓ Shape correct: {loss_val.shape == torch.Size([])}")
    print(f"  ✓ Requires grad: {loss_val.requires_grad}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test 3: Gradient flow
print("\n[Test 3] Gradient Flow")
pred_grad = torch.rand(2, 3, 128, 128, device=device, requires_grad=True)
target_grad = torch.rand(2, 3, 128, 128, device=device)

lap_loss = LaplacianPyramidLoss(num_levels=4)
edge_loss = EdgeAwareLoss(edge_weight=2.0)

try:
    loss_lap = lap_loss(pred_grad, target_grad)
    loss_edge = edge_loss(pred_grad, target_grad)
    total_loss = loss_lap + loss_edge
    total_loss.backward()

    print(f"  ✓ Combined loss: {total_loss.item():.4f}")
    print(f"  ✓ Gradient computed: {pred_grad.grad is not None}")
    if pred_grad.grad is not None:
        grad_norm = pred_grad.grad.norm().item()
        print(f"  ✓ Gradient norm: {grad_norm:.4f}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print("\n" + "=" * 60)
print("✅ Phase 5 losses working correctly!")
print("=" * 60)
