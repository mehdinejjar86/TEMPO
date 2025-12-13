# tempo_loss.py - TEMPO BEAST Uncertainty-Aware Loss System
"""
Full loss system with learnable uncertainty weighting:

1. Homoscedastic (Task-Level) Uncertainty:
   - Learnable log-variance per loss component
   - Automatic loss balancing during training
   - Based on Kendall et al. "Multi-Task Learning Using Uncertainty"

2. Heteroscedastic (Pixel-Level) Uncertainty:
   - Decoder predicts per-pixel variance
   - Down-weights uncertain regions (occlusions, motion blur)
   - Produces calibrated confidence maps

3. Loss Components:
   - Charbonnier (smooth L1)
   - SSIM (structural)
   - Perceptual (VGG features)
   - Gradient (edge sharpness)
   - Laplacian Pyramid (multi-scale edges)
   - Frequency (FFT domain)
   - Census (local structure)

4. Regularizers:
   - Bidirectional consistency
   - Offset smoothness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
import math

from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn


# =============================================================================
# Basic Loss Components
# =============================================================================

class CharbonnierLoss(nn.Module):
    """Charbonnier loss - smooth L1 variant, better for image reconstruction."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.eps ** 2)
        
        if weight is not None:
            loss = loss * weight
        
        return loss.mean()


class SSIMLoss(nn.Module):
    """SSIM as a loss (1 - SSIM) for direct optimization."""
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        
        gaussian = self._create_gaussian_kernel(window_size, sigma)
        self.register_buffer('window', gaussian)
    
    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) @ g.unsqueeze(0)
        return window.unsqueeze(0).unsqueeze(0)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = pred.shape
        
        window = self.window.expand(C, 1, -1, -1).to(pred.device)
        pad = self.window_size // 2
        
        mu1 = F.conv2d(pred, window, padding=pad, groups=C)
        mu2 = F.conv2d(target, window, padding=pad, groups=C)
        
        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=C) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        
        if weight is not None:
            ssim_map = ssim_map * weight
        
        return 1.0 - ssim_map.mean()


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss."""
    def __init__(self, layers: list = None, weights: list = None):
        super().__init__()
        
        if layers is None:
            layers = ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4']
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0]
        
        self.layers = layers
        self.weights = weights
        self.vgg = None
        self.layer_indices = {
            'relu1_2': 4, 'relu2_2': 9, 'relu3_4': 16, 'relu4_4': 23
        }
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _load_vgg(self, device):
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:24].to(device)
            vgg.eval()
            for p in vgg.parameters():
                p.requires_grad = False
            return vgg
        except Exception as e:
            print(f"⚠️ Could not load VGG: {e}")
            return None
    
    def _extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.vgg is None:
            self.vgg = self._load_vgg(x.device)
        if self.vgg is None:
            return {}
        
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        features = {}
        
        for name, idx in self.layer_indices.items():
            if name in self.layers:
                x = self.vgg[:idx+1](x) if not features else self.vgg[list(self.layer_indices.values())[list(self.layer_indices.keys()).index(name)-1]+1:idx+1](x)
                features[name] = x
        
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.vgg is None:
            self.vgg = self._load_vgg(pred.device)
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)
        
        pred_norm = (pred - self.mean.to(pred.device)) / self.std.to(pred.device)
        target_norm = (target - self.mean.to(target.device)) / self.std.to(target.device)
        
        loss = 0.0
        pred_feat, target_feat = pred_norm, target_norm
        prev_idx = 0
        
        for i, (name, idx) in enumerate(self.layer_indices.items()):
            if name in self.layers:
                pred_feat = self.vgg[prev_idx:idx+1](pred_feat)
                target_feat = self.vgg[prev_idx:idx+1](target_feat)
                loss += self.weights[self.layers.index(name)] * F.l1_loss(pred_feat, target_feat)
                prev_idx = idx + 1
        
        return loss / len(self.layers)


class GradientLoss(nn.Module):
    """Gradient loss using Sobel filters for edge sharpness."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def _gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] == 3:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        grad_x = F.conv2d(x, self.sobel_x.to(x.device), padding=1)
        grad_y = F.conv2d(x, self.sobel_y.to(x.device), padding=1)
        return grad_x, grad_y
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred_gx, pred_gy = self._gradient(pred)
        target_gx, target_gy = self._gradient(target)
        
        loss_x = (pred_gx - target_gx).abs()
        loss_y = (pred_gy - target_gy).abs()
        
        if weight is not None:
            weight_gray = weight.mean(dim=1, keepdim=True) if weight.shape[1] > 1 else weight
            loss_x = loss_x * weight_gray
            loss_y = loss_y * weight_gray
        
        return loss_x.mean() + loss_y.mean()
    
    def get_edge_weight(self, target: torch.Tensor) -> torch.Tensor:
        """Compute edge-aware weight map from target."""
        gx, gy = self._gradient(target)
        edge_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        edge_weight = 1.0 + edge_mag / (edge_mag.max() + 1e-6)
        return edge_weight


class FrequencyLoss(nn.Module):
    """FFT-based frequency domain loss."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # FFT doesn't support bfloat16 on MPS, cast to float32
        orig_dtype = pred.dtype
        pred_f32 = pred.float()
        target_f32 = target.float()
        
        pred_fft = torch.fft.fft2(pred_f32, norm='ortho')
        target_fft = torch.fft.fft2(target_f32, norm='ortho')
        
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        loss = F.l1_loss(pred_mag, target_mag)
        return loss.to(orig_dtype)


class CensusLoss(nn.Module):
    """Census transform loss for local structure preservation."""
    def __init__(self, patch_size: int = 7):
        super().__init__()
        self.patch_size = patch_size
        self.pad = patch_size // 2
    
    def _census_transform(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        B, C, H, W = x.shape
        x_pad = F.pad(x, [self.pad] * 4, mode='reflect')
        
        census = []
        center = x
        for i in range(self.patch_size):
            for j in range(self.patch_size):
                if i == self.pad and j == self.pad:
                    continue
                neighbor = x_pad[:, :, i:i+H, j:j+W]
                census.append((center > neighbor).float())
        
        return torch.cat(census, dim=1)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_census = self._census_transform(pred)
        target_census = self._census_transform(target)
        
        diff = (pred_census - target_census).abs()
        return diff.mean()


# =============================================================================
# New Loss Components (Phase 4)
# =============================================================================

class LaplacianPyramidLoss(nn.Module):
    """Multi-scale Laplacian pyramid loss for sharp edges at all scales."""
    def __init__(self, num_levels: int = 4, sigma: float = 1.0):
        super().__init__()
        self.num_levels = num_levels
        
        # Create Gaussian kernel for downsampling
        size = 5
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.unsqueeze(1) @ g.unsqueeze(0)
        self.register_buffer('gaussian', kernel.view(1, 1, size, size))
    
    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        kernel = self.gaussian.expand(C, 1, -1, -1).to(x.device)
        x_blur = F.conv2d(x, kernel, padding=2, groups=C)
        return F.interpolate(x_blur, scale_factor=0.5, mode='bilinear', align_corners=False)
    
    def _upsample(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    
    def _build_pyramid(self, x: torch.Tensor) -> List[torch.Tensor]:
        pyramid = []
        current = x
        
        for _ in range(self.num_levels):
            down = self._downsample(current)
            up = self._upsample(down, current.shape[-2:])
            laplacian = current - up  # High frequency detail
            pyramid.append(laplacian)
            current = down
        
        pyramid.append(current)  # Residual (lowest frequency)
        return pyramid
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred_pyr = self._build_pyramid(pred)
        target_pyr = self._build_pyramid(target)
        
        loss = 0.0
        # Increasing weight at finer scales
        level_weights = [2 ** i for i in range(self.num_levels + 1)]
        total_weight = sum(level_weights)
        
        for i, (p, t) in enumerate(zip(pred_pyr, target_pyr)):
            level_loss = F.l1_loss(p, t)
            loss += level_weights[i] * level_loss / total_weight
        
        return loss


class EdgeAwareWeighting(nn.Module):
    """Compute edge-aware weights to focus on important regions."""
    def __init__(self):
        super().__init__()
        self.gradient = GradientLoss()
    
    def forward(self, target: torch.Tensor, base_weight: float = 1.0, 
                edge_boost: float = 2.0) -> torch.Tensor:
        """
        Returns weight map: higher at edges.
        Args:
            target: [B, C, H, W] target image
            base_weight: weight for flat regions
            edge_boost: additional weight at edges
        """
        edge_weight = self.gradient.get_edge_weight(target)
        return base_weight + edge_boost * (edge_weight - 1.0)


# =============================================================================
# Uncertainty Modules
# =============================================================================

class HomoscedasticUncertainty(nn.Module):
    """
    Learnable task-level uncertainty for automatic loss weighting.
    
    Based on: "Multi-Task Learning Using Uncertainty to Weigh Losses"
    by Kendall, Gal, Cipolla (CVPR 2018)
    
    Each task has learnable log-variance σ²:
    weighted_loss = loss / (2 * σ²) + log(σ)
    
    If σ is large → loss weight is small (network is uncertain about this task)
    If σ is small → loss weight is large (network is confident)
    The log(σ) term prevents σ from going to infinity.
    """
    def __init__(self, num_tasks: int, init_log_var: float = 0.0):
        super().__init__()
        # Learnable log-variance for each task
        # Initialized to 0 → σ² = 1 → weight = 0.5
        self.log_vars = nn.Parameter(torch.full((num_tasks,), init_log_var))
    
    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            losses: List of scalar loss tensors, one per task
        Returns:
            total_loss: Weighted sum with uncertainty
            weights: Dict of learned weights for logging
        """
        assert len(losses) == len(self.log_vars)
        
        total = 0.0
        weights = {}
        
        for i, loss in enumerate(losses):
            # Precision (inverse variance) = exp(-log_var) = 1/σ²
            precision = torch.exp(-self.log_vars[i])
            
            # Weighted loss: loss/(2σ²) + log(σ) = loss * precision/2 + log_var/2
            weighted = precision * loss + self.log_vars[i]
            total += weighted
            
            weights[f'task_{i}_weight'] = precision.item()
            weights[f'task_{i}_log_var'] = self.log_vars[i].item()
        
        return total, weights
    
    def get_weights(self) -> torch.Tensor:
        """Get current learned weights (precisions)."""
        return torch.exp(-self.log_vars)


class HeteroscedasticLoss(nn.Module):
    """
    Pixel-level uncertainty weighting.
    
    Network predicts RGB + log_variance per pixel.
    Uncertain regions (occlusions, blur) get down-weighted.
    
    Loss = |pred - target|² / (2 * σ²) + log(σ²) / 2
    """
    def __init__(self, min_log_var: float = -10.0, max_log_var: float = 10.0):
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pred: [B, 3, H, W] predicted RGB
            target: [B, 3, H, W] ground truth
            log_var: [B, 1, H, W] predicted log-variance
        Returns:
            loss: Weighted reconstruction loss
            uncertainty_map: [B, 1, H, W] normalized uncertainty
        """
        # Clamp log_var for stability
        log_var = log_var.clamp(self.min_log_var, self.max_log_var)
        
        # Precision = 1/σ²
        precision = torch.exp(-log_var)
        
        # Per-pixel squared error
        diff_sq = (pred - target).pow(2).mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Gaussian NLL: |pred-target|²/(2σ²) + log(σ²)/2
        loss = precision * diff_sq + log_var
        
        # Uncertainty map (normalized for visualization)
        uncertainty_map = torch.sigmoid(log_var)  # 0 = confident, 1 = uncertain
        
        return loss.mean(), uncertainty_map


# =============================================================================
# Motion Regularizers
# =============================================================================

class BidirectionalConsistencyLoss(nn.Module):
    """
    Enforce that offsets from different frames are consistent.
    velocity = offset / temporal_distance should be similar across frames.
    """
    def forward(self, offsets: torch.Tensor, rel_times: torch.Tensor) -> torch.Tensor:
        """
        Args:
            offsets: [B, N, 2, H, W] per-frame offsets
            rel_times: [B, N] relative temporal distances
        """
        B, N, _, H, W = offsets.shape
        
        if N < 2:
            return torch.tensor(0.0, device=offsets.device)
        
        # Normalize offsets by time to get velocity
        rel_times_safe = rel_times.abs().clamp(min=0.1).view(B, N, 1, 1, 1)
        velocities = offsets / rel_times_safe
        
        # Consistency: all velocities should be similar
        mean_velocity = velocities.mean(dim=1, keepdim=True)
        consistency_loss = ((velocities - mean_velocity) ** 2).mean()
        
        return consistency_loss


class OffsetSmoothnessLoss(nn.Module):
    """
    Encourage spatially smooth offsets (motion should be locally coherent).
    """
    def forward(self, offsets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            offsets: [B, N, 2, H, W] or [B, 2, H, W]
        """
        if offsets.dim() == 5:
            # Average over frames
            offsets = offsets.mean(dim=1)
        
        # Compute spatial gradients
        grad_x = offsets[:, :, :, 1:] - offsets[:, :, :, :-1]
        grad_y = offsets[:, :, 1:, :] - offsets[:, :, :-1, :]
        
        smoothness = grad_x.pow(2).mean() + grad_y.pow(2).mean()
        return smoothness


# =============================================================================
# Main TEMPO Uncertainty Loss
# =============================================================================

class TEMPOUncertaintyLoss(nn.Module):
    """
    Full uncertainty-aware loss for TEMPO BEAST.
    
    Combines:
    - Homoscedastic uncertainty (learnable task weights)
    - Heteroscedastic uncertainty (pixel-level confidence)
    - All loss components
    - Motion regularizers
    """
    
    TASK_NAMES = ['recon', 'ssim', 'perceptual', 'gradient', 'laplacian', 'frequency', 'census']
    
    def __init__(
        self,
        use_homoscedastic: bool = True,
        use_heteroscedastic: bool = True,
        use_edge_aware: bool = True,
        # Initial weights (used if homoscedastic is disabled)
        recon_weight: float = 1.0,
        ssim_weight: float = 0.2,
        perceptual_weight: float = 0.1,
        gradient_weight: float = 0.1,
        laplacian_weight: float = 0.1,
        frequency_weight: float = 0.05,
        census_weight: float = 0.05,
        # Regularizers (not learned, fixed weights)
        bidirectional_weight: float = 0.1,
        offset_smooth_weight: float = 0.05,
        # Uncertainty settings
        init_log_var: float = 0.0,
    ):
        super().__init__()
        
        self.use_homoscedastic = use_homoscedastic
        self.use_heteroscedastic = use_heteroscedastic
        self.use_edge_aware = use_edge_aware
        
        # Fixed weights (used when homoscedastic is disabled)
        self.fixed_weights = {
            'recon': recon_weight,
            'ssim': ssim_weight,
            'perceptual': perceptual_weight,
            'gradient': gradient_weight,
            'laplacian': laplacian_weight,
            'frequency': frequency_weight,
            'census': census_weight,
        }
        self.bidirectional_weight = bidirectional_weight
        self.offset_smooth_weight = offset_smooth_weight
        
        # Loss components
        self.charbonnier = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.perceptual = PerceptualLoss()
        self.gradient = GradientLoss()
        self.laplacian = LaplacianPyramidLoss()
        self.frequency = FrequencyLoss()
        self.census = CensusLoss()
        
        # Edge-aware weighting
        if use_edge_aware:
            self.edge_weighting = EdgeAwareWeighting()
        
        # Homoscedastic uncertainty (learnable task weights)
        if use_homoscedastic:
            self.task_uncertainty = HomoscedasticUncertainty(
                num_tasks=len(self.TASK_NAMES),
                init_log_var=init_log_var
            )
        
        # Heteroscedastic (pixel-level) uncertainty
        if use_heteroscedastic:
            self.pixel_uncertainty = HeteroscedasticLoss()
        
        # Motion regularizers
        self.bidirectional = BidirectionalConsistencyLoss()
        self.offset_smooth = OffsetSmoothnessLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        log_var: Optional[torch.Tensor] = None,
        aux: Optional[Dict] = None,
        anchor_times: Optional[torch.Tensor] = None,
        target_time: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            pred: [B, 3, H, W] predicted frame
            target: [B, 3, H, W] ground truth
            log_var: [B, 1, H, W] predicted log-variance (optional)
            aux: dict with offsets, confidences, etc.
            anchor_times: [B, N] timestamps
            target_time: [B] or [B, 1] target timestamp
            step: current training step (for scheduling)
        """
        loss_dict = {}
        
        # =====================================================================
        # 1. Edge-aware weighting (optional)
        # =====================================================================
        edge_weight = None
        if self.use_edge_aware:
            edge_weight = self.edge_weighting(target)
            loss_dict['edge_weight_mean'] = edge_weight.mean()
        
        # =====================================================================
        # 2. Heteroscedastic (pixel-level) uncertainty
        # =====================================================================
        pixel_weight = None
        if self.use_heteroscedastic and log_var is not None:
            hetero_loss, uncertainty_map = self.pixel_uncertainty(pred, target, log_var)
            loss_dict['heteroscedastic'] = hetero_loss
            loss_dict['uncertainty_mean'] = uncertainty_map.mean()
            loss_dict['uncertainty_std'] = uncertainty_map.std()
            
            # Combine with edge weighting: confident at edges should be more important
            pixel_weight = torch.exp(-log_var)  # High precision where confident
            if edge_weight is not None:
                pixel_weight = pixel_weight * edge_weight
        elif edge_weight is not None:
            pixel_weight = edge_weight
        
        # =====================================================================
        # 3. Compute individual losses
        # =====================================================================
        task_losses = []
        
        # Reconstruction (Charbonnier)
        recon_loss = self.charbonnier(pred, target, pixel_weight)
        task_losses.append(recon_loss)
        loss_dict['recon'] = recon_loss
        
        # SSIM
        ssim_loss = self.ssim_loss(pred, target, pixel_weight)
        task_losses.append(ssim_loss)
        loss_dict['ssim_loss'] = ssim_loss
        
        # Perceptual
        perc_loss = self.perceptual(pred, target)
        task_losses.append(perc_loss)
        loss_dict['perceptual'] = perc_loss
        
        # Gradient
        grad_loss = self.gradient(pred, target, pixel_weight)
        task_losses.append(grad_loss)
        loss_dict['gradient'] = grad_loss
        
        # Laplacian pyramid
        lap_loss = self.laplacian(pred, target, pixel_weight)
        task_losses.append(lap_loss)
        loss_dict['laplacian'] = lap_loss
        
        # Frequency
        freq_loss = self.frequency(pred, target)
        task_losses.append(freq_loss)
        loss_dict['frequency'] = freq_loss
        
        # Census
        census_loss = self.census(pred, target)
        task_losses.append(census_loss)
        loss_dict['census'] = census_loss
        
        # =====================================================================
        # 4. Combine with uncertainty weighting
        # =====================================================================
        if self.use_homoscedastic:
            main_loss, learned_weights = self.task_uncertainty(task_losses)
            
            # Log learned weights
            for i, name in enumerate(self.TASK_NAMES):
                loss_dict[f'weight_{name}'] = learned_weights[f'task_{i}_weight']
                loss_dict[f'log_var_{name}'] = learned_weights[f'task_{i}_log_var']
        else:
            # Use fixed weights
            main_loss = sum(
                self.fixed_weights[name] * task_losses[i]
                for i, name in enumerate(self.TASK_NAMES)
            )
        
        # =====================================================================
        # 5. Add heteroscedastic loss (if using)
        # =====================================================================
        if self.use_heteroscedastic and log_var is not None:
            main_loss = main_loss + 0.1 * hetero_loss
        
        # =====================================================================
        # 6. Motion regularizers
        # =====================================================================
        if aux is not None and anchor_times is not None:
            offsets = aux.get('offsets', None)
            
            # Handle nested dict from multi-scale fusion
            if isinstance(offsets, dict):
                offsets = offsets.get('scale2', offsets.get('scale1', None))
            
            if offsets is not None and target_time is not None:
                # Bidirectional consistency
                if target_time.dim() == 1:
                    target_time = target_time.unsqueeze(1)
                rel_times = target_time - anchor_times
                
                bidir_loss = self.bidirectional(offsets, rel_times)
                loss_dict['bidirectional'] = bidir_loss
                main_loss = main_loss + self.bidirectional_weight * bidir_loss
                
                # Offset smoothness
                smooth_loss = self.offset_smooth(offsets)
                loss_dict['offset_smooth'] = smooth_loss
                main_loss = main_loss + self.offset_smooth_weight * smooth_loss
        
        # =====================================================================
        # 7. Compute metrics
        # =====================================================================
        with torch.no_grad():
            mse = F.mse_loss(pred, target)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            loss_dict['psnr'] = psnr
            loss_dict['ssim'] = ssim_fn(pred, target, data_range=1.0)
            loss_dict['l1'] = F.l1_loss(pred, target)
        
        loss_dict['total'] = main_loss
        
        return main_loss, loss_dict
    
    def get_learned_weights(self) -> Dict[str, float]:
        """Get current learned task weights."""
        if self.use_homoscedastic:
            weights = self.task_uncertainty.get_weights()
            return {name: weights[i].item() for i, name in enumerate(self.TASK_NAMES)}
        return self.fixed_weights.copy()


# =============================================================================
# Loss Scheduler (for gradual enabling)
# =============================================================================

class UncertaintyLossScheduler:
    """
    Schedule when to enable different loss features.
    
    The uncertainty system will learn automatically, but we can still
    control when certain components become active.
    """
    def __init__(
        self,
        loss_fn: TEMPOUncertaintyLoss,
        warmup_steps: int = 1000,
        heteroscedastic_start: int = 5000,
        perceptual_start: int = 2000,
    ):
        self.loss_fn = loss_fn
        self.warmup_steps = warmup_steps
        self.heteroscedastic_start = heteroscedastic_start
        self.perceptual_start = perceptual_start
        
        self._original_use_hetero = loss_fn.use_heteroscedastic
    
    def update(self, step: int):
        """Update loss settings based on training step."""
        # Enable heteroscedastic after warmup
        if step < self.heteroscedastic_start:
            self.loss_fn.use_heteroscedastic = False
        else:
            self.loss_fn.use_heteroscedastic = self._original_use_hetero


# =============================================================================
# Metric Tracker
# =============================================================================

class MetricTracker:
    """Track and smooth metrics for logging."""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history = defaultdict(list)
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
    
    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            if torch.is_tensor(v):
                v = v.item()
            self.history[k].append(v)
            if len(self.history[k]) > self.window_size:
                self.history[k].pop(0)
            self.totals[k] += v
            self.counts[k] += 1
    
    def get_smoothed(self) -> Dict[str, float]:
        return {k: sum(v) / len(v) for k, v in self.history.items() if v}
    
    def get_average(self) -> Dict[str, float]:
        return {k: self.totals[k] / self.counts[k] for k in self.totals if self.counts[k] > 0}
    
    def get_averages(self) -> Dict[str, float]:
        """Alias for get_average for backward compatibility."""
        return self.get_average()
    
    def reset(self):
        self.history.clear()
        self.totals.clear()
        self.counts.clear()


# =============================================================================
# Builder Functions
# =============================================================================

def build_tempo_loss(
    config: Optional[Dict] = None,
    use_uncertainty: bool = True,
    **kwargs
) -> TEMPOUncertaintyLoss:
    """Build TEMPO loss with uncertainty."""
    
    defaults = {
        'use_homoscedastic': True,
        'use_heteroscedastic': True,
        'use_edge_aware': True,
        'recon_weight': 1.0,
        'ssim_weight': 0.2,
        'perceptual_weight': 0.1,
        'gradient_weight': 0.1,
        'laplacian_weight': 0.1,
        'frequency_weight': 0.05,
        'census_weight': 0.05,
        'bidirectional_weight': 0.1,
        'offset_smooth_weight': 0.05,
    }
    
    if config:
        defaults.update(config)
    defaults.update(kwargs)
    
    if not use_uncertainty:
        defaults['use_homoscedastic'] = False
        defaults['use_heteroscedastic'] = False
    
    return TEMPOUncertaintyLoss(**defaults)


# Keep old name for compatibility
def LossScheduler(*args, **kwargs):
    """Alias for backward compatibility."""
    return UncertaintyLossScheduler(*args, **kwargs)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 2, 3, 64, 64
    
    pred = torch.randn(B, C, H, W, device=device)
    target = torch.randn(B, C, H, W, device=device)
    log_var = torch.randn(B, 1, H, W, device=device)
    
    # Test with full uncertainty
    print("Testing TEMPOUncertaintyLoss...")
    loss_fn = build_tempo_loss(use_uncertainty=True).to(device)
    
    total_loss, loss_dict = loss_fn(pred, target, log_var)
    
    print(f"✓ Total loss: {total_loss.item():.4f}")
    print(f"✓ PSNR: {loss_dict['psnr'].item():.2f} dB")
    print(f"✓ SSIM: {loss_dict['ssim'].item():.4f}")
    print(f"✓ Uncertainty mean: {loss_dict.get('uncertainty_mean', 0):.4f}")
    
    print("\nLearned weights:")
    for name in TEMPOUncertaintyLoss.TASK_NAMES:
        w = loss_dict.get(f'weight_{name}', 0)
        print(f"  {name}: {w:.4f}")
    
    print(f"\n✓ Loss params: {sum(p.numel() for p in loss_fn.parameters())} learnable")
    print("🔥 Uncertainty loss ready!")
