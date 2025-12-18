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
from typing import Dict, Optional, Tuple

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