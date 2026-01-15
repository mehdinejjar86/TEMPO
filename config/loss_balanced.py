"""
Balanced Loss Configuration for PSNR-SSIM Optimization

This configuration aims to achieve strong performance on BOTH metrics:
- Target PSNR: 38-39 dB (slightly lower than pure PSNR optimization)
- Target SSIM: 0.975-0.980 (significantly higher than current 0.9691)

Key Changes from PSNR-Optimized Config:
1. Increased SSIM weight: 0.3 → 1.0 (optimize structural similarity)
2. Decreased L1 weight: 1.5 → 1.0 (reduce pixel-wise emphasis)
3. Enabled perceptual loss: 0.0 → 0.1 (texture preservation)
4. Increased frequency loss: 0.05/0.02 → 0.15/0.05 (detail sharpness)
5. Increased gradient loss: 0.03 → 0.1 (edge quality)

Expected Results:
- PSNR: ~38-39 dB (vs current 40.75 dB) - still excellent, -1.5 to -2.5 dB
- SSIM: ~0.975-0.980 (vs current 0.9691) - significant improvement, +0.006 to +0.011
- Better balance between pixel accuracy and perceptual quality
"""

BALANCED_LOSS_CONFIG = {
    # =====================================================
    # BALANCED PSNR-SSIM CONFIG
    # Goal: Optimize both metrics simultaneously
    # =====================================================

    # Core reconstruction - balanced L1 and SSIM
    "w_l1": 1.0,           # ↓ from 1.5 - reduce pure pixel emphasis
    "w_ssim": 1.0,         # ↑ from 0.3 - significantly increase structural quality
    "w_freq_struct": 0.15, # ↑ from 0.05 - improve detail preservation
    "w_freq_phase": 0.05,  # ↑ from 0.02 - better phase coherence

    # Gradient loss - moderate for edge quality
    "w_gradient": 0.1,     # ↑ from 0.03 - sharper edges

    # Temporal coherence - moderate
    "w_coherence": 0.1,    # ↑ from 0.05 - better motion consistency

    # Occlusion handling - moderate
    "w_occlusion": 0.2,    # ↑ from 0.1 - better occlusion awareness

    # Weight regularization - keep light
    "w_weight_entropy": 0.005,
    "w_weight_smooth": 0.002,
    "w_weight_coverage": 0.01,
    "w_weight_sparsity": 0.002,

    # Confidence shaping
    "w_conf_target": 0.005,
    "conf_target": 0.7,

    # Multi-scale - balanced across scales
    "pyramid_weights": [1.0, 0.5, 0.2],  # ↑ from [1.0, 0.3, 0.1] - more multi-scale

    # Perceptual loss - ENABLED for texture quality
    "w_perceptual": 0.1,   # ↑ from 0.0 - add perceptual quality
    "use_perceptual": True,

    # Scene cut handling
    "cut_loss_scale": 0.5,
}


# =====================================================
# ALTERNATIVE: SSIM-FOCUSED CONFIG
# Use this if you want to prioritize SSIM over PSNR
# =====================================================

SSIM_FOCUSED_CONFIG = {
    # Core reconstruction - SSIM dominates
    "w_l1": 0.8,           # ↓ further - less pixel emphasis
    "w_ssim": 1.5,         # ↑ further - maximize structural quality
    "w_freq_struct": 0.2,
    "w_freq_phase": 0.08,

    # Gradient loss - high for sharp edges
    "w_gradient": 0.15,

    # Temporal coherence - high for smooth motion
    "w_coherence": 0.15,

    # Occlusion handling - high
    "w_occlusion": 0.3,

    # Weight regularization
    "w_weight_entropy": 0.005,
    "w_weight_smooth": 0.002,
    "w_weight_coverage": 0.01,
    "w_weight_sparsity": 0.002,

    # Confidence shaping
    "w_conf_target": 0.005,
    "conf_target": 0.7,

    # Multi-scale - balanced
    "pyramid_weights": [1.0, 0.5, 0.2],

    # Perceptual loss - moderate
    "w_perceptual": 0.15,  # Higher for better perceptual quality
    "use_perceptual": True,

    # Scene cut handling
    "cut_loss_scale": 0.5,
}


# =====================================================
# Usage Instructions
# =====================================================

"""
To use these configurations, modify your training script:

Option 1: Pass config dict directly to loss function
------------------------------------------------------
from config.loss_balanced import BALANCED_LOSS_CONFIG
from model.loss.tempo_loss import TempoLoss

# In your training setup
loss_fn = TempoLoss(config=BALANCED_LOSS_CONFIG)

Option 2: Modify train_tempo_mixed.py
--------------------------------------
# Add import at top of file
from config.loss_balanced import BALANCED_LOSS_CONFIG

# In _setup_model() method (around line 157):
self.loss_fn = TempoLoss(config=BALANCED_LOSS_CONFIG)


Option 3: Command-line argument
--------------------------------
Add a --loss_config argument to your training script:

parser.add_argument('--loss_config', type=str, choices=['psnr', 'balanced', 'ssim'],
                    default='psnr', help='Loss configuration preset')

Then in _setup_model():
if self.config.loss_config == 'balanced':
    from config.loss_balanced import BALANCED_LOSS_CONFIG
    self.loss_fn = TempoLoss(config=BALANCED_LOSS_CONFIG)
elif self.config.loss_config == 'ssim':
    from config.loss_balanced import SSIM_FOCUSED_CONFIG
    self.loss_fn = TempoLoss(config=SSIM_FOCUSED_CONFIG)
else:
    self.loss_fn = TempoLoss()  # Default PSNR-optimized
"""
