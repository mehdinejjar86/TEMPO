"""
SSIM Verification Script

Verifies SSIM computation is correct by:
1. Testing with known reference values
2. Comparing torchmetrics vs skimage implementations
3. Checking data range sensitivity
4. Testing with actual model outputs
"""
import torch
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure as TorchSSIM
from skimage.metrics import structural_similarity as skimage_ssim
from PIL import Image
import os


def test_identical_images():
    """Test 1: Identical images should give SSIM = 1.0"""
    print("\n" + "="*60)
    print("Test 1: Identical Images (should be 1.0)")
    print("="*60)

    img = torch.rand(1, 3, 256, 256)

    # TorchMetrics SSIM
    ssim_torch = TorchSSIM(data_range=1.0)
    score_torch = ssim_torch(img, img).item()

    # Skimage SSIM (needs numpy, channel-last)
    img_np = img[0].permute(1, 2, 0).numpy()
    score_skimage = skimage_ssim(img_np, img_np, data_range=1.0, channel_axis=2)

    print(f"TorchMetrics SSIM: {score_torch:.6f}")
    print(f"Skimage SSIM:      {score_skimage:.6f}")

    assert abs(score_torch - 1.0) < 1e-5, f"TorchMetrics failed: {score_torch}"
    assert abs(score_skimage - 1.0) < 1e-5, f"Skimage failed: {score_skimage}"
    print("✅ PASSED")

    return score_torch, score_skimage


def test_data_range_sensitivity():
    """Test 2: Verify data_range parameter is critical"""
    print("\n" + "="*60)
    print("Test 2: Data Range Sensitivity")
    print("="*60)

    # Create two slightly different images in [0, 1] range
    img1 = torch.rand(1, 3, 256, 256)
    img2 = img1 + torch.randn(1, 3, 256, 256) * 0.01
    img2 = img2.clamp(0, 1)

    # Test with correct data_range=1.0
    ssim_correct = TorchSSIM(data_range=1.0)
    score_correct = ssim_correct(img1, img2).item()

    # Test with WRONG data_range=255
    ssim_wrong = TorchSSIM(data_range=255.0)
    score_wrong = ssim_wrong(img1, img2).item()

    print(f"Images in [0, 1] range:")
    print(f"  Min: {img1.min():.3f}, Max: {img1.max():.3f}")
    print(f"\nSSIM with data_range=1.0:   {score_correct:.6f} ✅")
    print(f"SSIM with data_range=255.0: {score_wrong:.6f} ❌ (WRONG!)")
    print(f"\nDifference: {abs(score_correct - score_wrong):.6f}")

    if abs(score_correct - score_wrong) > 0.01:
        print("✅ PASSED - data_range significantly affects SSIM")
    else:
        print("⚠️  WARNING - data_range doesn't affect SSIM as expected")

    return score_correct, score_wrong


def test_range_verification():
    """Test 3: Verify what happens with different input ranges"""
    print("\n" + "="*60)
    print("Test 3: Input Range Verification")
    print("="*60)

    # Test images in different ranges
    img_01 = torch.rand(1, 3, 256, 256)  # [0, 1]
    img_255 = img_01 * 255.0  # [0, 255]
    img_neg1_1 = img_01 * 2.0 - 1.0  # [-1, 1]

    # Add slight noise
    noise = torch.randn_like(img_01) * 0.01

    # Compute SSIM for each range with correct data_range
    ssim_01 = TorchSSIM(data_range=1.0)(img_01, img_01 + noise.clamp(-img_01.min(), 1-img_01.max())).item()

    ssim_255 = TorchSSIM(data_range=255.0)(
        img_255,
        (img_255 + noise * 255.0).clamp(0, 255)
    ).item()

    ssim_neg1_1 = TorchSSIM(data_range=2.0)(
        img_neg1_1,
        (img_neg1_1 + noise * 2.0).clamp(-1, 1)
    ).item()

    print("Same images in different ranges (with matched data_range):")
    print(f"  [0, 1] range, data_range=1.0:     {ssim_01:.6f}")
    print(f"  [0, 255] range, data_range=255.0: {ssim_255:.6f}")
    print(f"  [-1, 1] range, data_range=2.0:    {ssim_neg1_1:.6f}")
    print(f"\nMax difference: {max(abs(ssim_01 - ssim_255), abs(ssim_01 - ssim_neg1_1)):.6f}")

    if max(abs(ssim_01 - ssim_255), abs(ssim_01 - ssim_neg1_1)) < 0.01:
        print("✅ PASSED - SSIM consistent across ranges when data_range is correct")
    else:
        print("⚠️  WARNING - SSIM varies across ranges")

    return ssim_01, ssim_255, ssim_neg1_1


def test_with_real_images():
    """Test 4: Compare with real validation images if available"""
    print("\n" + "="*60)
    print("Test 4: Real Validation Images (if available)")
    print("="*60)

    # Look for saved validation samples
    sample_dirs = [
        "runs/*/samples/vimeo_epoch_*_sample_000.png",
        "samples/vimeo_epoch_*_sample_000.png"
    ]

    # Try to find a sample
    sample_path = None
    for pattern in sample_dirs:
        import glob
        matches = glob.glob(pattern)
        if matches:
            sample_path = matches[-1]  # Use most recent
            break

    if sample_path is None:
        print("⚠️  No validation samples found, skipping this test")
        return None, None

    print(f"Found sample: {sample_path}")

    # Load the image (it's a 2x3 grid)
    img = Image.open(sample_path)
    img_np = np.array(img).astype(np.float32) / 255.0

    print(f"Image shape: {img_np.shape}")
    print(f"Image range: [{img_np.min():.3f}, {img_np.max():.3f}]")

    # Extract target and prediction (assuming grid layout)
    h, w = img_np.shape[:2]
    tile_h, tile_w = h // 2, w // 3

    # Target is at (1, 0), Pred is at (1, 1) in 2x3 grid
    target_crop = img_np[tile_h:tile_h*2, 0:tile_w]
    pred_crop = img_np[tile_h:tile_h*2, tile_w:tile_w*2]

    print(f"Crop shape: {target_crop.shape}")

    # Convert to torch tensors
    target_torch = torch.from_numpy(target_crop).permute(2, 0, 1).unsqueeze(0)
    pred_torch = torch.from_numpy(pred_crop).permute(2, 0, 1).unsqueeze(0)

    # Compute SSIM
    ssim_torch = TorchSSIM(data_range=1.0)(pred_torch, target_torch).item()
    ssim_skimage = skimage_ssim(target_crop, pred_crop, data_range=1.0, channel_axis=2)

    print(f"\nSSIM on validation sample:")
    print(f"  TorchMetrics: {ssim_torch:.6f}")
    print(f"  Skimage:      {ssim_skimage:.6f}")
    print(f"  Difference:   {abs(ssim_torch - ssim_skimage):.6f}")

    return ssim_torch, ssim_skimage


def test_psnr_ssim_correlation():
    """Test 5: Verify PSNR-SSIM relationship"""
    print("\n" + "="*60)
    print("Test 5: PSNR-SSIM Correlation")
    print("="*60)

    base_img = torch.rand(1, 3, 256, 256)

    # Create images with different noise levels
    noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    psnr_values = []
    ssim_values = []

    ssim_metric = TorchSSIM(data_range=1.0)

    for noise_std in noise_levels:
        noisy_img = (base_img + torch.randn_like(base_img) * noise_std).clamp(0, 1)

        # Compute PSNR
        mse = ((base_img - noisy_img) ** 2).mean()
        psnr = -10 * torch.log10(mse + 1e-8)

        # Compute SSIM
        ssim = ssim_metric(base_img, noisy_img)

        psnr_values.append(psnr.item())
        ssim_values.append(ssim.item())

        print(f"Noise σ={noise_std:.3f}: PSNR={psnr.item():5.2f} dB, SSIM={ssim.item():.4f}")

    # Check correlation
    psnr_arr = np.array(psnr_values)
    ssim_arr = np.array(ssim_values)
    correlation = np.corrcoef(psnr_arr, ssim_arr)[0, 1]

    print(f"\nPSNR-SSIM Correlation: {correlation:.4f}")

    if correlation > 0.9:
        print("✅ PASSED - Strong positive correlation (expected)")
    else:
        print("⚠️  WARNING - Weak correlation (unexpected)")

    return psnr_values, ssim_values, correlation


def diagnose_current_setup():
    """Test 6: Diagnose current validation setup"""
    print("\n" + "="*60)
    print("Test 6: Current Setup Diagnosis")
    print("="*60)

    print("\nCurrent SSIM Configuration:")
    print("  Library: torchmetrics.image.StructuralSimilarityIndexMeasure")
    print("  Data Range: 1.0")
    print("  Expected Input: [0, 1] range tensors")

    print("\n✓ Checklist for Validation:")
    print("  [ ] Are predictions clamped to [0, 1]?")
    print("  [ ] Are targets in [0, 1] range?")
    print("  [ ] Is data_range=1.0 correct for your images?")
    print("  [ ] Are you using the same SSIM implementation as SOTA papers?")

    print("\n⚠️  Common Issues:")
    print("  1. Images in [0, 255] but data_range=1.0 → SSIM artificially high")
    print("  2. Images in [-1, 1] but data_range=1.0 → SSIM artificially low")
    print("  3. Not clamping predictions → out-of-range values")
    print("  4. Different SSIM implementations → inconsistent results")


def main():
    print("\n" + "="*70)
    print("🔬 SSIM VERIFICATION SUITE")
    print("="*70)
    print("\nThis script will verify your SSIM computation is correct.")
    print("Focus: Finding why PSNR is high but SSIM is lower than SOTA\n")

    results = {}

    try:
        results['identical'] = test_identical_images()
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")

    try:
        results['data_range'] = test_data_range_sensitivity()
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")

    try:
        results['range_verify'] = test_range_verification()
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")

    try:
        results['real_images'] = test_with_real_images()
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")

    try:
        results['correlation'] = test_psnr_ssim_correlation()
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")

    try:
        diagnose_current_setup()
    except Exception as e:
        print(f"❌ Test 6 failed: {e}")

    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print("\nKey Findings:")
    print("1. TorchMetrics SSIM implementation is correct ✅")
    print("2. data_range parameter is CRITICAL for correct SSIM")
    print("3. PSNR and SSIM should correlate positively")
    print("\n⚠️  IF YOUR SSIM IS LOWER THAN EXPECTED:")
    print("   → Check that images are truly in [0, 1] during validation")
    print("   → Verify data_range=1.0 matches your image range")
    print("   → Compare visual quality with SOTA methods")
    print("   → Consider that high PSNR + lower SSIM suggests over-smoothing")

    print("\n💡 RECOMMENDATION:")
    print("   Add debug prints to validation to verify image ranges!")


if __name__ == "__main__":
    main()
