# tempo_loss.py
# Full-fledged loss for TEMPO v2
# - Charbonnier + MS-SSIM + Perceptual (LPIPS or VGG fallback)
# - Multi-scale with confidence masking
# - Temporal RGB-TV and optional triplet consistency
# - Weight entropy + bracket coverage
# - Attention entropy + confidence shaping
# - TV + luma/chroma stabilization
# - Cut/fallback gating
import math, warnings
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _batch_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute PSNR for a batch in dB using [0,1]-clamped float32 tensors.
    Returns a Python float for easy logging/aggregation.
    """
    p = pred.float().clamp(0, 1)
    t = target.float().clamp(0, 1)
    mse = F.mse_loss(p, t, reduction="mean").item()
    # avoid log of zero in rare identical-sample cases
    return -10.0 * math.log10(max(mse, eps)), mse


# -------------------------
# Loss Components
# -------------------------

class AdaptiveCharbonnier(nn.Module):
    """Charbonnier with learned epsilon per scale"""
    def __init__(self, scales: int = 3, eps_scale: float = 1e-3):
        super().__init__()
        self.eps_raw = nn.Parameter(torch.ones(scales))  # softplus → (0, +inf)
        self.eps_scale = eps_scale

    def forward(self, x: torch.Tensor, scale_idx: int = 0) -> torch.Tensor:
        eps = F.softplus(self.eps_raw[scale_idx]) * self.eps_scale
        return torch.sqrt(x * x + eps * eps)


class ConfidenceAwareSSIM(nn.Module):
    """SSIM that adapts to confidence maps"""
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        g = torch.tensor([math.exp(-(i - window_size // 2) ** 2 / (2 * sigma ** 2))
                          for i in range(window_size)])
        g = (g / g.sum()).float()
        window = (g.view(-1, 1) @ g.view(1, -1)).unsqueeze(0).unsqueeze(0)  # [1,1,ws,ws]
        self.register_buffer("window", window)
        self.ws = window_size

    def forward(self, x, y, conf=None):
        C = x.shape[1]
        win = self.window.to(device=x.device, dtype=x.dtype).expand(C, 1, self.ws, self.ws)

        mu_x = F.conv2d(x, win, padding=self.ws // 2, groups=C)
        mu_y = F.conv2d(y, win, padding=self.ws // 2, groups=C)

        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sigma_x2 = F.conv2d(x * x, win, padding=self.ws // 2, groups=C) - mu_x2
        sigma_y2 = F.conv2d(y * y, win, padding=self.ws // 2, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, win, padding=self.ws // 2, groups=C) - mu_xy

        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
        loss_map = (1.0 - ssim).clamp(0, 1)
        if conf is not None:
            loss_map = loss_map * conf
        return loss_map.mean()


class TemporalCoherenceLoss(nn.Module):
    """
    Coherence w.r.t. a time-aware baseline:
      - If bracketed: linear interp between nearest left/right anchors
      - Else (extrap): nearest neighbor baseline
    Entropy-scaled: lower attention entropy → allow more deviation
    """
    def forward(self, frames: torch.Tensor, times: torch.Tensor,
            pred: torch.Tensor, target_time: torch.Tensor,
            weights: torch.Tensor) -> torch.Tensor:
        """
        frames: [B,N,3,H,W], times: [B,N], pred: [B,3,H,W],
        target_time: [B] or [B,1], weights: [B,N]
        """
        B, N = times.shape
        # match dtypes to pred (bf16/fp16 safe)
        frames = frames.to(dtype=pred.dtype)

        tt = target_time.view(B, 1)

        dt = (times - tt).abs()  # [B,N]
        left_mask  = (times <= tt)
        right_mask = (times >= tt)

        big = torch.full_like(dt, 1e9)
        dist_left  = torch.where(left_mask,  (tt - times).abs(), big)
        dist_right = torch.where(right_mask, (times - tt).abs(), big)

        iL = dist_left.argmin(dim=1)   # [B]
        iR = dist_right.argmin(dim=1)  # [B]
        has_left  = left_mask.any(dim=1)
        has_right = right_mask.any(dim=1)
        bracketed = has_left & has_right

        baseline = pred.new_zeros(pred.shape)

        if bracketed.any():
            bidx = torch.nonzero(bracketed, as_tuple=True)[0]
            l = iL[bidx]; r = iR[bidx]
            t1 = times[bidx, l]; t2 = times[bidx, r]
            denom = (t2 - t1).clamp_min(1e-6)
            w2 = ((tt[bidx, 0] - t1) / denom).to(dtype=pred.dtype)
            w1 = (1.0 - w2).to(dtype=pred.dtype)
            f1 = frames[bidx, l]  # [b,3,H,W]
            f2 = frames[bidx, r]
            baseline[bidx] = f1 * w1.view(-1, 1, 1, 1) + f2 * w2.view(-1, 1, 1, 1)

        if (~bracketed).any():
            bidx = torch.nonzero(~bracketed, as_tuple=True)[0]
            j = dt[bidx].argmin(dim=1)
            baseline[bidx] = frames[bidx, j]

        # entropy scaling (lower entropy → allow more deviation)
        p = weights.clamp_min(1e-8)
        ent = -(p * p.log()).sum(dim=1, keepdim=True)               # [B,1]
        ent_norm = (ent / math.log(max(N, 2))).clamp(0.1, 1.0).view(B, 1, 1, 1).to(dtype=pred.dtype)

        return (pred - baseline).abs().mul(ent_norm).mean()



class FrequencyLoss(nn.Module):
    """Magnitude/phase loss in frequency domain (phase weighted by magnitude; computed in fp32)."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = pred.to(torch.float32)
        y = target.to(torch.float32)
        X = torch.fft.rfft2(x, norm="ortho")
        Y = torch.fft.rfft2(y, norm="ortho")

        magX, magY = X.abs(), Y.abs()
        phX, phY   = X.angle(), Y.angle()

        H, Wc = magX.shape[-2:]  # Wc = W//2+1
        low = torch.zeros_like(magX)
        low[..., :H//4, :Wc//4] = 1.0
        high = 1.0 - low

        L_struct = (magX - magY).abs()
        Ls = (L_struct * low).mean()
        Lh = (L_struct * high).mean()

        # phase on low-band, magnitude-weighted
        w = (magX + magY).clamp_min(1e-3) * low
        dphi = torch.atan2(torch.sin(phX - phY), torch.cos(phX - phY)).abs()
        Lp = (dphi * w).sum() / w.sum().clamp_min(1.0)
        return (Ls + 0.1 * Lh), Lp


class OcclusionAwareLoss(nn.Module):
    """Uses attention entropy and confidence to handle occlusions"""
    def __init__(self, entropy_threshold: float = 2.0):
        super().__init__()
        self.entropy_threshold = entropy_threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                conf: torch.Tensor, entropy: torch.Tensor) -> torch.Tensor:
        # Normalize entropy to [0,1]
        ent_norm = (entropy / self.entropy_threshold).clamp(0.0, 1.0)
        occ_score = ent_norm * (1.0 - conf)  # high entropy + low conf → likely occlusion
        weight = 1.0 - 0.7 * occ_score
        return (pred - target).abs().mul(weight).mean()


class BidirectionalConsistencyLoss(nn.Module):
    """
    TEMPO BEAST Phase 3: Bidirectional Consistency Loss

    Enforces temporal consistency by checking forward-backward synthesis.

    Forward pass:  [F0, F1, ...] → Ft
    Backward pass: [Ft, F_nearest, ...] → F0' (or F1')
    Loss: |F_anchor - F_anchor'|

    This helps the model learn temporally consistent representations and
    better handle motion and occlusions.

    Note: The backward synthesis must be computed externally (by the model)
    and passed to this loss. This is because the loss module shouldn't
    have direct access to the model to avoid circular dependencies.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Weight for consistency loss (default: 1.0)
        """
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        anchor_frame: torch.Tensor,        # [B, 3, H, W] - original anchor frame
        reconstructed_anchor: torch.Tensor, # [B, 3, H, W] - backward-synthesized anchor
        confidence: Optional[torch.Tensor] = None,  # [B, 1, H, W] - optional confidence map
    ) -> torch.Tensor:
        """
        Compute bidirectional consistency loss.

        Args:
            anchor_frame: Original anchor frame (e.g., F0 or F1)
            reconstructed_anchor: Backward-synthesized anchor frame (e.g., F0' or F1')
            confidence: Optional confidence map to weight the loss

        Returns:
            consistency_loss: Scalar loss value
        """
        # L1 consistency loss
        diff = (anchor_frame - reconstructed_anchor).abs()

        # Apply confidence weighting if available
        if confidence is not None:
            diff = diff * confidence

        return self.alpha * diff.mean()


class HomoscedasticUncertainty(nn.Module):
    """
    TEMPO BEAST Phase 4: Homoscedastic Uncertainty for Automatic Loss Balancing

    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., 2018)

    Learns task-level uncertainty (one log-variance per task) to automatically
    balance multiple loss terms during training.

    Loss formula: L = (1 / 2σ²) * L_task + log(σ)
    - First term: Precision-weighted task loss (higher σ → lower weight)
    - Second term: Regularization to prevent σ → ∞

    This eliminates manual hyperparameter tuning for loss weights!
    """

    def __init__(self, num_tasks: int = 7):
        """
        Args:
            num_tasks: Number of loss terms to balance (default: 7)
                      Tasks: L1, SSIM, freq_struct, freq_phase, temporal, occlusion, perceptual
        """
        super().__init__()
        # Initialize log-variances to 0 (σ² = 1, equal weighting at start)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply uncertainty-based weighting to multiple losses.

        Args:
            losses: List of scalar loss tensors (one per task)

        Returns:
            total_loss: Uncertainty-weighted sum of losses
            weights_dict: Effective weights (1/σ²) for monitoring
        """
        assert len(losses) == len(self.log_vars), \
            f"Expected {len(self.log_vars)} losses, got {len(losses)}"

        weighted_losses = []
        effective_weights = {}

        for i, loss in enumerate(losses):
            # Precision = 1 / σ² = exp(-log_var)
            precision = torch.exp(-self.log_vars[i])

            # Weighted loss: (1 / 2σ²) * L + (1/2) * log(σ²)
            weighted = 0.5 * precision * loss + 0.5 * self.log_vars[i]
            weighted_losses.append(weighted)

            # Store effective weight for monitoring (1/σ²)
            effective_weights[f"weight_{i}"] = precision.detach().item()

        total = torch.stack(weighted_losses).sum()

        return total, effective_weights


class HeteroscedasticLoss(nn.Module):
    """
    TEMPO BEAST Phase 4: Heteroscedastic Uncertainty for Pixel-Level Weighting

    Uses per-pixel uncertainty predictions to weight reconstruction loss.
    Uncertain regions (motion blur, occlusion) get lower weight.

    Loss formula: L = (1 / 2σ²) * (pred - target)² + (1/2) * log(σ²)
    - Pixel-wise adaptive weighting based on predicted uncertainty
    - Regularization prevents σ → ∞
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: How to reduce spatial dimensions ('mean' or 'sum')
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,              # [B, 3, H, W] - predicted RGB
        target: torch.Tensor,            # [B, 3, H, W] - target RGB
        log_var: torch.Tensor,           # [B, 1, H, W] - predicted log(σ²)
    ) -> torch.Tensor:
        """
        Compute heteroscedastic uncertainty-weighted reconstruction loss.

        Args:
            pred: Predicted image
            target: Ground truth image
            log_var: Predicted log-variance (uncertainty) per pixel

        Returns:
            loss: Scalar loss value
        """
        # Precision = 1 / σ²
        precision = torch.exp(-log_var)  # [B, 1, H, W]

        # Squared error
        sq_error = (pred - target).pow(2).mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Weighted loss: (1 / 2σ²) * error² + (1/2) * log(σ²)
        weighted_error = 0.5 * precision * sq_error
        regularization = 0.5 * log_var

        loss = weighted_error + regularization

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LaplacianPyramidLoss(nn.Module):
    """
    TEMPO BEAST Phase 5: Laplacian Pyramid Loss for Multi-Scale Edge Preservation

    Uses multi-scale bandpass filtering to preserve edges and fine details.
    Builds Gaussian pyramid then computes Laplacian as differences between levels.

    Laplacian pyramid: L_i = G_i - upsample(G_{i+1})
    where G_i is the i-th level of the Gaussian pyramid.

    This provides better edge preservation than standard MSE/L1 by explicitly
    supervising at multiple frequency bands.
    """

    def __init__(self, num_levels: int = 4, level_weights: Optional[List[float]] = None):
        """
        Args:
            num_levels: Number of pyramid levels (default: 4)
            level_weights: Weights for each level (default: [1.0, 0.8, 0.6, 0.4])
                          Emphasizes finer scales for sharper details
        """
        super().__init__()
        self.num_levels = num_levels
        self.level_weights = level_weights or [1.0, 0.8, 0.6, 0.4][:num_levels]

        # Gaussian kernel for pyramid construction (5x5)
        # σ = 1.0 for smooth downsampling
        kernel_1d = torch.tensor([1., 4., 6., 4., 1.]) / 16.0
        kernel_2d = kernel_1d.view(-1, 1) @ kernel_1d.view(1, -1)  # [5, 5]
        self.register_buffer("gaussian_kernel", kernel_2d.unsqueeze(0).unsqueeze(0))  # [1, 1, 5, 5]

    def _build_gaussian_pyramid(self, img: torch.Tensor) -> List[torch.Tensor]:
        """
        Build Gaussian pyramid by iterative smoothing and downsampling.

        Args:
            img: [B, C, H, W] input image

        Returns:
            pyramid: List of [B, C, H_i, W_i] at each level
        """
        pyramid = [img]
        current = img

        for _ in range(self.num_levels - 1):
            # Smooth with Gaussian kernel
            B, C, H, W = current.shape
            kernel = self.gaussian_kernel.to(device=current.device, dtype=current.dtype).expand(C, 1, 5, 5)
            smoothed = F.conv2d(current, kernel, padding=2, groups=C)

            # Downsample by 2
            downsampled = F.avg_pool2d(smoothed, kernel_size=2, stride=2)

            pyramid.append(downsampled)
            current = downsampled

        return pyramid

    def _build_laplacian_pyramid(self, img: torch.Tensor) -> List[torch.Tensor]:
        """
        Build Laplacian pyramid from Gaussian pyramid.

        Laplacian[i] = Gaussian[i] - upsample(Gaussian[i+1])

        Args:
            img: [B, C, H, W] input image

        Returns:
            laplacian: List of [B, C, H_i, W_i] Laplacian levels
        """
        gaussian = self._build_gaussian_pyramid(img)
        laplacian = []

        for i in range(len(gaussian) - 1):
            current_level = gaussian[i]
            next_level = gaussian[i + 1]

            # Upsample next level to match current size
            H, W = current_level.shape[-2:]
            upsampled = F.interpolate(next_level, size=(H, W), mode='bilinear', align_corners=False)

            # Laplacian = current - upsampled_next
            laplacian.append(current_level - upsampled)

        # Last level is just the coarsest Gaussian (residual)
        laplacian.append(gaussian[-1])

        return laplacian

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian pyramid loss between prediction and target.

        Args:
            pred: [B, C, H, W] predicted image
            target: [B, C, H, W] target image

        Returns:
            loss: Weighted sum of L1 losses at each pyramid level
        """
        # Build Laplacian pyramids
        laplacian_pred = self._build_laplacian_pyramid(pred)
        laplacian_target = self._build_laplacian_pyramid(target)

        # Compute weighted loss at each level
        total_loss = pred.new_tensor(0.0)

        for i, weight in enumerate(self.level_weights):
            if i < len(laplacian_pred):
                # L1 loss at this level
                level_loss = (laplacian_pred[i] - laplacian_target[i]).abs().mean()
                total_loss = total_loss + weight * level_loss

        return total_loss


class EdgeAwareLoss(nn.Module):
    """
    TEMPO BEAST Phase 5: Edge-Aware Loss with Gradient-Based Weighting

    Weights reconstruction loss by edge strength using Sobel gradient detection.
    Regions with strong edges get higher weight (2× by default).

    This improves perceptual quality by focusing on visually important edges.
    """

    def __init__(self, edge_weight: float = 2.0):
        """
        Args:
            edge_weight: Weight multiplier for edge regions (default: 2.0)
                        1.0 = no weighting, 2.0 = 2× weight for edges
        """
        super().__init__()
        self.edge_weight = edge_weight

        # Sobel kernels for gradient detection
        sobel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]) / 8.0

        sobel_y = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ]) / 8.0

        # Register as buffers [1, 1, 3, 3]
        self.register_buffer("sobel_x", sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer("sobel_y", sobel_y.unsqueeze(0).unsqueeze(0))

    def _compute_edge_map(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compute edge strength map using Sobel gradients.

        Args:
            img: [B, C, H, W] input image

        Returns:
            edge_map: [B, 1, H, W] gradient magnitude in [0, 1]
        """
        # Convert to grayscale for edge detection
        # Standard RGB→Y conversion: 0.299*R + 0.587*G + 0.114*B
        if img.shape[1] == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img

        # Apply Sobel filters
        sobel_x = self.sobel_x.to(device=img.device, dtype=img.dtype)
        sobel_y = self.sobel_y.to(device=img.device, dtype=img.dtype)

        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)

        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        # Normalize to [0, 1]
        grad_mag = grad_mag / (grad_mag.max() + 1e-6)

        return grad_mag

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute edge-aware L1 loss.

        Args:
            pred: [B, C, H, W] predicted image
            target: [B, C, H, W] target image

        Returns:
            loss: Edge-weighted L1 reconstruction loss
        """
        # Compute edge maps for both pred and target
        edge_pred = self._compute_edge_map(pred)
        edge_target = self._compute_edge_map(target)

        # Combined edge strength (max of pred and target)
        edge_strength = torch.max(edge_pred, edge_target)  # [B, 1, H, W]

        # Create weight map: 1.0 + (edge_weight - 1.0) * edge_strength
        # Non-edges: weight ≈ 1.0, Strong edges: weight ≈ edge_weight
        weight_map = 1.0 + (self.edge_weight - 1.0) * edge_strength

        # L1 loss with edge weighting
        l1_loss = (pred - target).abs()  # [B, C, H, W]

        # Apply weights (broadcast across channels)
        weighted_loss = l1_loss * weight_map

        return weighted_loss.mean()


class WeightRegularization(nn.Module):
    """Regularization for attention weights"""
    def forward(self, weights: torch.Tensor, times: torch.Tensor,
                target_time: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N = weights.shape
        p = weights.clamp_min(1e-8)

        losses = {}
        # 1) entropy towards ~half of max
        H = -(p * p.log()).sum(dim=1)                     # [B]
        target_H = 0.5 * math.log(max(N, 2))
        losses["entropy"] = (H - target_H).abs().mean()

        # 2) temporal smoothness in time order
        tt = target_time.view(B, 1)
        dt = (times - tt).abs()
        _, idx = dt.sort(dim=1)
        pw = p.gather(1, idx)
        if N > 1:
            losses["temporal_smooth"] = (pw[:, 1:] - pw[:, :-1]).abs().mean()

        # 3) coverage: nearest anchor should have non-trivial mass
        nearest = dt.argmin(dim=1)
        w_near = p.gather(1, nearest.view(-1, 1))
        losses["coverage"] = F.relu(0.1 - w_near).mean()

        # 4) sparsity: keep mass on top-2 reasonably high
        k = min(2, N)
        top2 = p.topk(k, dim=1).values.sum(dim=1)
        losses["sparsity"] = F.relu(0.6 - top2).mean()
        return losses


# -------------------------
# Main Loss Computer
# -------------------------

class TEMPOLoss(nn.Module):
    """
    Improved loss that:
    1. Adapts to N frames (works with 2 but scales)
    2. Leverages confidence and attention entropy
    3. Handles occlusions intelligently
    4. Preserves both structure and details
    """
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.cfg = {
            # Core reconstruction
            "w_l1": 0.8,
            "w_ssim": 1.0,
            "w_freq_struct": 0.3,
            "w_freq_phase": 0.1,

            # Temporal
            "w_coherence": 0.2,
            "w_bidirectional": 0.1,  # TEMPO BEAST Phase 3: Bidirectional consistency

            # Occlusion handling
            "w_occlusion": 0.5,

            # Weight regularization
            "w_weight_entropy": 0.01,
            "w_weight_smooth": 0.005,
            "w_weight_coverage": 0.02,
            "w_weight_sparsity": 0.005,

            # Confidence shaping
            "w_conf_target": 0.01,
            "conf_target": 0.7,

            # Multi-scale
            "pyramid_weights": [1.0, 0.5, 0.25],

            # Perceptual (optional)
            "w_perceptual": 0.05,
            "use_perceptual": True,

            # TEMPO BEAST Phase 4: Uncertainty
            "use_homoscedastic": True,   # Automatic loss balancing
            "use_heteroscedastic": True,  # Pixel-level uncertainty weighting
            "w_heteroscedastic": 0.5,     # Weight for heteroscedastic loss

            # TEMPO BEAST Phase 5: Advanced Losses
            "w_laplacian": 0.5,           # Laplacian pyramid loss
            "w_edge_aware": 0.3,          # Edge-aware weighting
            "laplacian_levels": 4,        # Number of pyramid levels
            "edge_weight_multiplier": 2.0, # 2× weight for edges

            # Scene cut handling
            "cut_loss_scale": 0.3,  # Reduce loss when cut detected
        }
        if config:
            self.cfg.update(config)

        self.charb = AdaptiveCharbonnier(scales=3, eps_scale=1e-3)
        self.ssim = ConfidenceAwareSSIM()
        self.freq_loss = FrequencyLoss()
        self.temporal = TemporalCoherenceLoss()
        self.bidirectional = BidirectionalConsistencyLoss()  # TEMPO BEAST Phase 3
        self.occlusion = OcclusionAwareLoss()
        self.weight_reg = WeightRegularization()

        # TEMPO BEAST Phase 4: Uncertainty modules
        self.homoscedastic = None
        if self.cfg["use_homoscedastic"]:
            # 7 tasks: L1, SSIM, freq_struct, freq_phase, temporal, occlusion, perceptual
            self.homoscedastic = HomoscedasticUncertainty(num_tasks=7)

        self.heteroscedastic_loss = None
        if self.cfg["use_heteroscedastic"]:
            self.heteroscedastic_loss = HeteroscedasticLoss(reduction='mean')

        # TEMPO BEAST Phase 5: Advanced losses
        self.laplacian_loss = LaplacianPyramidLoss(
            num_levels=self.cfg["laplacian_levels"],
            level_weights=[1.0, 0.8, 0.6, 0.4][:self.cfg["laplacian_levels"]]
        )
        self.edge_aware_loss = EdgeAwareLoss(
            edge_weight=self.cfg["edge_weight_multiplier"]
        )

        # Optional perceptual loss
        self.perceptual = None
        if self.cfg["use_perceptual"]:
            try:
                import lpips  # type: ignore
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*'pretrained' is deprecated.*")
                    self.perceptual = lpips.LPIPS(net="vgg")
                for p in self.perceptual.parameters():
                    p.requires_grad = False
            except Exception:
                self.perceptual = None
                self.cfg["w_perceptual"] = 0.0

    def _ensure_perc_on(self, device: torch.device):
        if self.perceptual is not None:
            # Only move if needed (handles MPS/CUDA/CPU)
            cur_dev = next(self.perceptual.parameters(), None)
            if (cur_dev is None) or (cur_dev.device != device):
                self.perceptual = self.perceptual.to(device)
                
    def forward(
        self,
        pred: torch.Tensor,           # [B,3,H,W]
        target: torch.Tensor,         # [B,3,H,W]
        frames: torch.Tensor,         # [B,N,3,H,W]
        anchor_times: torch.Tensor,   # [B,N]
        target_time: torch.Tensor,    # [B] or [B,1]
        aux: Dict[str, torch.Tensor], # model aux
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        device = pred.device
        B, _, H, W = pred.shape
        N = frames.shape[1]

        # Aux - support both old (conf_map) and new (confidence) naming
        weights = aux.get("weights", torch.full((B, max(N, 1)), 1.0 / max(N, 1), device=device))
        conf_map = aux.get("confidence", aux.get("conf_map", torch.ones(B, 1, H, W, device=device))).to(dtype=pred.dtype)
        attn_entropy = aux.get("entropy", aux.get("attn_entropy", torch.zeros(B, 1, H, W, device=device))).to(dtype=pred.dtype)
        fallback_mask = aux.get("fallback_mask", torch.zeros(B, device=device))

        # Scene-cut gate scalar per-sample → broadcast later (match pred dtype)
        gate = torch.where(
            fallback_mask > 0.5,
            torch.tensor(self.cfg["cut_loss_scale"], device=device, dtype=pred.dtype),
            torch.tensor(1.0, device=device, dtype=pred.dtype),
        ).view(B, 1, 1, 1)

        losses: Dict[str, float] = {}

        # ===== Multi-scale reconstruction =====
        pw = self.cfg["pyramid_weights"]
        total_recon = pred.new_tensor(0.0)

        for si, w_si in enumerate(pw[:3]):
            s = 2 ** si
            if s == 1:
                p_s, t_s = pred, target
                c_s = conf_map
            else:
                p_s = F.avg_pool2d(pred, s, s)
                t_s = F.avg_pool2d(target, s, s)
                c_s = F.avg_pool2d(conf_map, s, s)

            # L1 (Charb) + SSIM; both gated consistently
            L1 = self.charb(p_s - t_s, si) * c_s * gate
            SS = self.ssim(p_s, t_s, c_s) * gate.mean()

            total_recon = total_recon + w_si * (self.cfg["w_l1"] * L1.mean() + self.cfg["w_ssim"] * SS)

            if s == 1:
                losses["l1"] = float(L1.mean().detach())
                losses["ssim"] = float(SS.detach())

        # ===== Frequency domain loss (detail preservation) =====
        freq_struct, freq_phase = self.freq_loss(pred * gate, target * gate)
        total_recon = total_recon + self.cfg["w_freq_struct"] * freq_struct + self.cfg["w_freq_phase"] * freq_phase
        losses["freq_struct"] = float(freq_struct.detach())
        losses["freq_phase"] = float(freq_phase.detach())

        # ===== Temporal coherence (only if N > 1) =====
        if N > 1:
            L_temp = self.temporal(frames, anchor_times, pred, target_time, weights)
            total_recon = total_recon + self.cfg["w_coherence"] * L_temp
            losses["temporal"] = float(L_temp.detach())

        # ===== Bidirectional consistency (TEMPO BEAST Phase 3) =====
        # Only computed if backward synthesis is provided in aux
        if self.cfg["w_bidirectional"] > 0 and "backward_anchor" in aux:
            backward_anchor = aux["backward_anchor"]  # [B, 3, H, W]
            anchor_idx = aux.get("backward_anchor_idx", 0)  # Which anchor was reconstructed

            # Get the original anchor frame
            original_anchor = frames[:, anchor_idx]  # [B, 3, H, W]

            L_bidir = self.bidirectional(
                original_anchor,
                backward_anchor,
                confidence=conf_map
            )
            total_recon = total_recon + self.cfg["w_bidirectional"] * L_bidir
            losses["bidirectional"] = float(L_bidir.detach())

        # ===== Occlusion-aware RGB term =====
        L_occ = self.occlusion(pred, target, conf_map, attn_entropy)
        total_recon = total_recon + self.cfg["w_occlusion"] * L_occ
        losses["occlusion"] = float(L_occ.detach())

        # ===== Weight regularization =====
        reg = pred.new_tensor(0.0)
        wreg = self.weight_reg(weights, anchor_times, target_time)
        for k, v in wreg.items():  # entropy, temporal_smooth, coverage, sparsity
            coef = self.cfg.get(f"w_weight_{k}", 0.0)
            if coef > 0:
                reg = reg + coef * v
                losses[f"wreg_{k}"] = float(v.detach())

        # ===== Confidence target =====
        L_conf_tgt = ((conf_map - self.cfg["conf_target"]) ** 2).mean()
        reg = reg + self.cfg["w_conf_target"] * L_conf_tgt
        losses["conf_target"] = float(L_conf_tgt.detach())

        # ===== Perceptual loss (fp32) =====
        if self.cfg["w_perceptual"] > 0 and (self.perceptual is not None):
            self._ensure_perc_on(pred.device)
            
            p_perc, t_perc = pred, target
            
            # Dynamically downsample for LPIPS while preserving aspect ratio
            # Use bilinear interpolation (area mode requires divisible sizes)
            h_orig, w_orig = p_perc.shape[-2:]
            target_longest_edge = 256
            if max(h_orig, w_orig) > target_longest_edge:
                ratio = target_longest_edge / max(h_orig, w_orig)
                new_h, new_w = int(h_orig * ratio), int(w_orig * ratio)
                # Ensure minimum size of 32 for LPIPS
                new_h, new_w = max(32, new_h), max(32, new_w)
                p_perc = F.interpolate(p_perc, size=(new_h, new_w), mode='bilinear', align_corners=False)
                t_perc = F.interpolate(t_perc, size=(new_h, new_w), mode='bilinear', align_corners=False)

            # LPIPS expects [-1,1] and float32
            perc = self.perceptual((p_perc * 2 - 1).float(), (t_perc * 2 - 1).float()).mean()
            total_recon = total_recon + self.cfg["w_perceptual"] * perc
            losses["perceptual"] = float(perc.detach())

        # ===== TEMPO BEAST Phase 4: Apply Uncertainty-Based Weighting =====
        # This replaces manual loss weighting with learned uncertainty
        if self.homoscedastic is not None:
            # Collect task losses (without manual weights)
            # Tasks: [L1, SSIM, freq_struct, freq_phase, temporal, occlusion, perceptual]
            task_losses = []

            # 1. L1 (from finest scale)
            p_s, t_s, c_s = pred, target, conf_map
            L1_raw = self.charb(p_s - t_s, 0) * c_s * gate
            task_losses.append(L1_raw.mean())

            # 2. SSIM
            SS_raw = self.ssim(p_s, t_s, c_s) * gate.mean()
            task_losses.append(SS_raw)

            # 3. Frequency structure
            task_losses.append(freq_struct)

            # 4. Frequency phase
            task_losses.append(freq_phase)

            # 5. Temporal coherence (or zero if N=1)
            if N > 1 and "temporal" in locals():
                task_losses.append(L_temp)
            else:
                task_losses.append(pred.new_tensor(0.0))

            # 6. Occlusion-aware
            task_losses.append(L_occ)

            # 7. Perceptual (or zero if disabled)
            if self.cfg["w_perceptual"] > 0 and "perc" in locals():
                task_losses.append(perc)
            else:
                task_losses.append(pred.new_tensor(0.0))

            # Apply homoscedastic uncertainty weighting
            total_recon, eff_weights = self.homoscedastic(task_losses)

            # Log effective weights for monitoring
            losses.update({f"homo_{k}": v for k, v in eff_weights.items()})
            losses["total_homoscedastic"] = float(total_recon.detach())

            # Log learned log-variances (σ²)
            for i, log_var in enumerate(self.homoscedastic.log_vars):
                losses[f"homo_logvar_{i}"] = float(log_var.detach())

        # ===== Heteroscedastic Loss (Pixel-Level Uncertainty) =====
        if self.heteroscedastic_loss is not None and "uncertainty_log_var" in aux:
            log_var = aux["uncertainty_log_var"]  # [B, 1, H, W]

            L_hetero = self.heteroscedastic_loss(pred, target, log_var)
            total_recon = total_recon + self.cfg["w_heteroscedastic"] * L_hetero
            losses["heteroscedastic"] = float(L_hetero.detach())

        # ===== TEMPO BEAST Phase 5: Laplacian Pyramid Loss =====
        if self.cfg["w_laplacian"] > 0:
            L_laplacian = self.laplacian_loss(pred * gate, target * gate)
            total_recon = total_recon + self.cfg["w_laplacian"] * L_laplacian
            losses["laplacian"] = float(L_laplacian.detach())

        # ===== TEMPO BEAST Phase 5: Edge-Aware Loss =====
        if self.cfg["w_edge_aware"] > 0:
            L_edge = self.edge_aware_loss(pred * gate, target * gate)
            total_recon = total_recon + self.cfg["w_edge_aware"] * L_edge
            losses["edge_aware"] = float(L_edge.detach())

        total = total_recon + reg
        losses["total"] = float(total.detach())
        losses["cut_rate"] = float(fallback_mask.float().mean().detach())
        losses["conf_mean"] = float(conf_map.mean().detach())
        losses["entropy_mean"] = float(attn_entropy.mean().detach())
        
        # ===== Metrics (not used for gradients) =====
        psnr_db, mse_val = _batch_psnr(pred, target)
        losses["mse"] = float(mse_val)
        losses["psnr"] = float(psnr_db)

        return total, losses



def build_tempo_loss(config: Optional[Dict] = None) -> TEMPOLoss:
    return TEMPOLoss(config)


# ===== Training utilities =====

class LossScheduler:
    """Dynamically adjust loss weights during training (non-compounding)."""
    def __init__(self, loss_module: TEMPOLoss, schedule: Optional[Dict] = None):
        self.loss = loss_module
        self.base_cfg = dict(loss_module.cfg)  # frozen reference
        self.schedule = schedule or {
            "warmup_steps": 5_000,
            "perceptual_start": 2_000,
            "coherence_ramp": (5_000, 15_000),
            "bidirectional_ramp": (5_000, 15_000),  # TEMPO BEAST Phase 3
            "laplacian_ramp": (5_000, 15_000),      # TEMPO BEAST Phase 5
            "edge_aware_ramp": (5_000, 15_000),     # TEMPO BEAST Phase 5
        }
        self.step = 0

    def update(self, step: int):
        self.step = step
        cfg = self.loss.cfg
        base = self.base_cfg

        # start from base every time
        for k, v in base.items():
            cfg[k] = v

        # warmup for regularizers
        warm = min(1.0, step / max(1, self.schedule["warmup_steps"]))
        for k in ("w_weight_entropy", "w_weight_smooth", "w_weight_coverage",
                  "w_weight_sparsity", "w_conf_target"):
            cfg[k] = base[k] * warm

        # delay perceptual
        cfg["w_perceptual"] = 0.0 if step < self.schedule["perceptual_start"] else base["w_perceptual"]

        # ramp temporal coherence
        s, e = self.schedule["coherence_ramp"]
        if step <= s:
            cfg["w_coherence"] = 0.0
        elif step >= e:
            cfg["w_coherence"] = base["w_coherence"]
        else:
            cfg["w_coherence"] = base["w_coherence"] * ((step - s) / (e - s))

        # ramp bidirectional consistency (TEMPO BEAST Phase 3)
        s, e = self.schedule["bidirectional_ramp"]
        if step <= s:
            cfg["w_bidirectional"] = 0.0
        elif step >= e:
            cfg["w_bidirectional"] = base["w_bidirectional"]
        else:
            cfg["w_bidirectional"] = base["w_bidirectional"] * ((step - s) / (e - s))

        # ramp Laplacian pyramid loss (TEMPO BEAST Phase 5)
        s, e = self.schedule["laplacian_ramp"]
        if step <= s:
            cfg["w_laplacian"] = 0.0
        elif step >= e:
            cfg["w_laplacian"] = base["w_laplacian"]
        else:
            cfg["w_laplacian"] = base["w_laplacian"] * ((step - s) / (e - s))

        # ramp edge-aware loss (TEMPO BEAST Phase 5)
        s, e = self.schedule["edge_aware_ramp"]
        if step <= s:
            cfg["w_edge_aware"] = 0.0
        elif step >= e:
            cfg["w_edge_aware"] = base["w_edge_aware"]
        else:
            cfg["w_edge_aware"] = base["w_edge_aware"] * ((step - s) / (e - s))


class MetricTracker:
    """Track and aggregate metrics efficiently"""
    def __init__(self):
        self.metrics: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self.metrics[k] = self.metrics.get(k, 0.0) + float(v)
            self.counts[k] = self.counts.get(k, 0) + 1

    def get_averages(self) -> Dict[str, float]:
        return {k: self.metrics[k] / max(1, self.counts[k]) for k in self.metrics}

    def reset(self):
        self.metrics.clear()
        self.counts.clear()

_loss_singleton = None

def tempo_loss(pred, target, aux, anchor_times, target_time, frames=None, config=None):
    """
    Compat wrapper around TEMPOLoss:
      - keeps a singleton instance
      - enforces the v2 signature (needs 'frames')
    """
    global _loss_singleton
    if _loss_singleton is None:
        _loss_singleton = TEMPOLoss(config)

    if frames is None:
        raise ValueError("tempo_loss wrapper requires 'frames' (shape [B,N,3,H,W]) for temporal & weight terms.")

    return _loss_singleton(
        pred=pred,
        target=target,
        frames=frames,
        anchor_times=anchor_times,
        target_time=target_time,
        aux=aux,
    )