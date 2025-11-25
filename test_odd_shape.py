
import torch
import sys
from model.tempo import build_tempo

def test_odd_resolution():
    B, N, H, W = 1, 2, 255, 355
    print(f"Testing odd resolution ({H}x{W})...")
    model = build_tempo()
    model.eval()
    
    frames = torch.randn(B, N, 3, H, W)
    anchor_times = torch.linspace(0, 1, N).unsqueeze(0)
    target_time = torch.tensor([0.5])
    
    try:
        model(frames, anchor_times, target_time)
        print("Success!")
    except Exception as e:
        print(f"Failed as expected: {e}")

if __name__ == "__main__":
    test_odd_resolution()
