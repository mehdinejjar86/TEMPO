#!/usr/bin/env python3
"""
Smoke test for TEMPO v2 with Hybrid Swin+ConvNeXt encoder/decoder
Tests basic forward pass, backward pass, and shape correctness
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model.tempo import build_tempo

def test_forward_pass():
    """Test basic forward pass with different input configurations"""
    print("🧪 Testing TEMPO v2 with Hybrid Swin+ConvNeXt...\n")
    
    # Build model
    print("Building model...")
    model = build_tempo(
        base_channels=64,  # Will be ignored, uses fixed dims [96,192,384,768]
        temporal_channels=64,
        attn_heads=4,
        attn_points=4
    )
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Test cases
    test_cases = [
        {
            "name": "Standard triplet (256x256)",
            "B": 2, "N": 3, "H": 256, "W": 256
        },
        {
            "name": "High-res (512x512)",
            "B": 1, "N": 3, "H": 512, "W": 512
        },
        {
            "name": "Multiple views (N=5)",
            "B": 1, "N": 5, "H": 256, "W": 256
        },
    ]
    
    print("\nRunning forward pass tests:\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        B, N, H, W = test['B'], test['N'], test['H'], test['W']
        
        # Create dummy inputs
        frames = torch.randn(B, N, 3, H, W)
        anchor_times = torch.linspace(0, 1, N).unsqueeze(0).expand(B, N)
        target_time = torch.full((B,), 0.5)
        
        # Forward pass
        try:
            with torch.no_grad():
                output, aux = model(frames, anchor_times, target_time)
            
            # Verify shapes
            assert output.shape == (B, 3, H, W), f"Output shape mismatch: {output.shape}"
            assert 'weights' in aux, "Missing 'weights' in aux"
            assert 'conf_map' in aux, "Missing 'conf_map' in aux"
            assert aux['weights'].shape == (B, N), f"Weights shape mismatch: {aux['weights'].shape}"
            assert aux['conf_map'].shape == (B, 1, H, W), f"Conf map shape mismatch: {aux['conf_map'].shape}"
            
            # Check value ranges
            assert output.min() >= 0 and output.max() <= 1, "Output not in [0,1]"
            assert torch.isclose(aux['weights'].sum(dim=1), torch.ones(B), atol=1e-5).all(), "Weights don't sum to 1"
            
            print(f"  ✓ Shape: {tuple(output.shape)}")
            print(f"  ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"  ✓ Weights sum: {aux['weights'].sum(dim=1).mean():.6f}")
            print(f"  ✓ Confidence mean: {aux['conf_map'].mean():.3f}")
            print()
            
        except Exception as e:
            print(f"  ❌ FAILED: {str(e)}")
            raise
    
    # Test backward pass (backpropagation)
    print("=" * 60)
    print("Testing Backward Pass (Backpropagation):\n")
    
    model.train()  # Set to training mode
    
    # Use smaller test for backward (saves memory)
    print("Backward pass test (768x768)")
    B, N, H, W = 1, 4, 768, 768
    
    try:
        # Create inputs with gradient tracking
        frames = torch.randn(B, N, 3, H, W, requires_grad=True)
        anchor_times = torch.linspace(0, 1, N).unsqueeze(0).expand(B, N)
        target_time = torch.full((B,), 0.5)
        target_gt = torch.randn(B, 3, H, W)  # Fake ground truth
        
        # Forward pass
        output, aux = model(frames, anchor_times, target_time)
        
        # Compute simple loss
        loss = ((output - target_gt) ** 2).mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert frames.grad is not None, "No gradient for input frames"
        
        # Check gradients are not all zero
        grad_norm = frames.grad.abs().sum().item()
        assert grad_norm > 0, "Gradients are all zero"
        
        # Count parameters with gradients
        params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params_count = sum(1 for _ in model.parameters())
        
        print(f"  ✓ Loss computed: {loss.item():.6f}")
        print(f"  ✓ Backward pass successful")
        print(f"  ✓ Input gradients computed (norm: {grad_norm:.6f})")
        print(f"  ✓ Parameters with gradients: {params_with_grad}/{total_params_count}")
        
        # Check for NaN or Inf in gradients
        has_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters())
        has_inf = any(p.grad is not None and torch.isinf(p.grad).any() for p in model.parameters())
        
        if has_nan:
            print(f"  ⚠️  WARNING: NaN detected in gradients")
        else:
            print(f"  ✓ No NaN in gradients")
            
        if has_inf:
            print(f"  ⚠️  WARNING: Inf detected in gradients")
        else:
            print(f"  ✓ No Inf in gradients")
        
        # Gradient statistics
        grad_stats = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_stats.append({
                    'name': name,
                    'mean': param.grad.abs().mean().item(),
                    'max': param.grad.abs().max().item()
                })
        
        if grad_stats:
            avg_grad = sum(s['mean'] for s in grad_stats) / len(grad_stats)
            max_grad = max(s['max'] for s in grad_stats)
            print(f"  ✓ Average gradient magnitude: {avg_grad:.6e}")
            print(f"  ✓ Maximum gradient magnitude: {max_grad:.6e}")
        
        print()
        
    except Exception as e:
        print(f"  ❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_forward_pass()
