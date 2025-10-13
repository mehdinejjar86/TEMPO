# tempo_loss.py
# Full-fledged loss for TEMPO v2
# - Charbonnier + MS-SSIM + Perceptual (LPIPS or VGG fallback)
# - Multi-scale with confidence masking
# - Temporal RGB-TV and optional triplet consistency
# - Weight entropy + bracket coverage
# - Attention entropy + confidence shaping
# - TV + luma/chroma stabilization
# - Cut/fallback gating

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utils
# -------------------------

def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)

def rgb_to_yuv(x: torch.Tensor) -> torch.Tensor:
    # x: [B,3,H,W] in [0,1]
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return torch.cat([y, u, v], dim=1)

def tv_loss(img: torch.Tensor) -> torch.Tensor:
    dh = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean()
    dw = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
    return dh + dw

def build_gaussian_window(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    g = torch.tensor(
        [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
         for x in range(window_size)], device=device
    )
    g = (g / g.sum()).unsqueeze(0)
    window = g.t() @ g  # [ws,ws]
    return window

def _ssim_map(x, y, window, C1=0.01**2, C2=0.03**2):
    # x,y: [B,3,H,W], window: [ws,ws]
    B, C, H, W = x.shape
    ws = window.shape[0]
    window = window.view(1, 1, ws, ws).to(x.dtype).to(x.device)
    window = window.expand(C, 1, ws, ws)

    mu_x = F.conv2d(x, window, padding=ws//2, groups=C)
    mu_y = F.conv2d(y, window, padding=ws//2, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=ws//2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=ws//2, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=ws//2, groups=C) - mu_xy

    ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim

def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    win = build_gaussian_window(window_size, sigma, x.device)
    ssim_map = _ssim_map(x, y, win)
    return (1.0 - ssim_map.clamp(0, 1)).mean()

def msssim_loss(x: torch.Tensor, y: torch.Tensor, levels: int = 3) -> torch.Tensor:
    # simple MS-SSIM with 3 levels; weights from original paper
    weights = [0.0448, 0.2856, 0.3001]  # truncated for 3 levels then renormalized
    weights = torch.tensor(weights[:levels], device=x.device, dtype=x.dtype)
    weights = weights / weights.sum()

    losses = []
    x_i, y_i = x, y
    for _ in range(levels):
        losses.append(ssim_loss(x_i, y_i))
        x_i = F.avg_pool2d(x_i, kernel_size=2, stride=2, count_include_pad=False)
        y_i = F.avg_pool2d(y_i, kernel_size=2, stride=2, count_include_pad=False)
    losses = torch.stack(losses)
    return (losses * weights).sum()

class PerceptualLoss(nn.Module):
    """
    Wrapper that tries LPIPS first; if not available, falls back to VGG19 features.
    If neither is available, returns zero.
    """
    def __init__(self, net: str = "vgg"):
        super().__init__()
        self.mode = "none"
        self.lpips = None
        self.vgg = None
        try:
            import lpips  # type: ignore
            self.lpips = lpips.LPIPS(net='vgg')
            self.mode = "lpips"
        except Exception:
            try:
                from torchvision.models import vgg19, VGG19_Weights
                vgg = vgg19(weights=VGG19_Weights.DEFAULT)
                for p in vgg.parameters(): p.requires_grad = False
                self.vgg = vgg[:16]  # up to relu3_3
                self.mode = "vgg"
            except Exception:
                self.mode = "none"

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.mode == "lpips":
            # LPIPS expects [-1,1]
            x_ = x * 2 - 1
            y_ = y * 2 - 1
            return self.lpips(x_, y_).mean()
        elif self.mode == "vgg":
            # VGG expects [0,1], but no strict normalization for a lightweight proxy
            fx = self.vgg(x)
            fy = self.vgg(y)
            return (fx - fy).abs().mean()
        else:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)


# -------------------------
# Main Loss
# -------------------------

def default_config() -> Dict:
    return dict(
        # reconstruction
        w_charb=1.0, w_msssim=0.2, w_perc=0.05,
        # temporal (off unless triplets provided)
        w_temporal_tv=0.02, w_triplet_latent=0.1,  # triplet latent requires you to pass neighbours
        # weights & attention hygiene
        w_weight_entropy=0.002, w_coverage=0.01,
        w_attn_entropy=0.01, w_conf_reg=0.005,
        # priors
        w_tv=0.001, w_luma=0.1, w_chroma=0.02,
        # reconstruction details
        charb_eps=1e-3, msssim_levels=3,
        # confidence target
        conf_target=0.6,
        # cut gating scale (0..1): multiply recon terms by (1 - fallback_mask*gate)
        cut_gate=0.75,
        # multi-scale reconstruction weights (×1, ×1/2, ×1/4)
        pyr_weights=(1.0, 0.5, 0.25),
    )


class LossComputer(nn.Module):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.cfg = default_config()
        if config:
            self.cfg.update(config)
        self.perc = PerceptualLoss()

    @staticmethod
    def _down(x: torch.Tensor, s: int) -> torch.Tensor:
        if s == 1: return x
        # area is great for downsampling supervision
        return F.interpolate(x, scale_factor=1.0/s, mode='area')

    @staticmethod
    def _mask_like(x: torch.Tensor, to_size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=to_size, mode='bilinear', align_corners=False)

    @staticmethod
    def _bracket_coverage(weights: torch.Tensor, anchor_times: torch.Tensor, target_time: torch.Tensor) -> torch.Tensor:
        """
        Encourage some mass on both nearest left/right anchors when the target is bracketed.
        """
        B, N = weights.shape
        device = weights.device
        left_mask  = (anchor_times <= target_time)   # [B,N]
        right_mask = (anchor_times >= target_time)   # [B,N]
        big = torch.tensor(1e6, device=device)

        dist_left  = torch.where(left_mask,  (target_time - anchor_times).abs(), big)
        dist_right = torch.where(right_mask, (anchor_times - target_time).abs(), big)

        iL = dist_left.argmin(dim=1)   # [B]
        iR = dist_right.argmin(dim=1)  # [B]

        wL = weights[torch.arange(B, device=device), iL]
        wR = weights[torch.arange(B, device=device), iR]

        # If not bracketed (all left or all right), penalize nothing.
        has_left  = left_mask.any(dim=1).float()
        has_right = right_mask.any(dim=1).float()
        bracketed = (has_left * has_right)  # [B]

        tau = 0.2
        cover = F.relu(tau - (wL + wR)) * bracketed
        return cover.mean()

    def forward(
        self,
        pred: torch.Tensor,               # [B,3,H,W]
        target: torch.Tensor,             # [B,3,H,W]
        aux: Dict[str, torch.Tensor],     # from model: conf_map [B,1,H,W], attn_entropy [B,1,H,W], weights [B,N], fallback_mask [B]
        anchor_times: torch.Tensor,       # [B,N]
        target_time: torch.Tensor,        # [B] or [B,1]
        # Optional temporal neighbors (for triplet consistency)
        pred_prev: Optional[torch.Tensor] = None,
        pred_next: Optional[torch.Tensor] = None,
        target_prev: Optional[torch.Tensor] = None,
        target_next: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns: total_loss, logs
        """
        cfg = self.cfg
        device = pred.device
        B, _, H, W = pred.shape

        # --- confidence & gating ---
        conf_full = aux.get("conf_map", torch.ones(B, 1, H, W, device=device))
        conf_full = torch.clamp(conf_full.detach(), 1e-4, 1.0)
        attn_ent  = aux.get("attn_entropy", torch.zeros(B, 1, H, W, device=device))
        weights   = aux.get("weights", None)
        fallback_mask = aux.get("fallback_mask", torch.zeros(B, device=device))  # [B]

        # Build cut gate scalar per-sample → per-pixel mask
        # scale recon terms by (1 - cut_gate * fallback)
        if fallback_mask.ndim == 1:
            fb = fallback_mask.view(B, 1, 1, 1)
        else:
            fb = fallback_mask
        gate = (1.0 - cfg["cut_gate"] * fb).clamp(0.0, 1.0)  # [B,1,1,1]
        gate = gate.expand(-1, 1, H, W)

        # --- multi-scale pyramids ---
        pw1, pw2, pw4 = cfg["pyr_weights"]
        pred_p2, tgt_p2 = self._down(pred, 2), self._down(target, 2)
        pred_p4, tgt_p4 = self._down(pred, 4), self._down(target, 4)

        conf_p2 = self._mask_like(conf_full, pred_p2.shape[-2:])
        conf_p4 = self._mask_like(conf_full, pred_p4.shape[-2:])
        gate_p2 = self._mask_like(gate, pred_p2.shape[-2:])
        gate_p4 = self._mask_like(gate, pred_p4.shape[-2:])

        # --- reconstruction: Charbonnier + MS-SSIM + Perceptual ---
        L_charb = (charbonnier(pred - target, cfg["charb_eps"]) * conf_full * gate).mean() * pw1 \
                + (charbonnier(pred_p2 - tgt_p2, cfg["charb_eps"]) * conf_p2 * gate_p2).mean() * pw2 \
                + (charbonnier(pred_p4 - tgt_p4, cfg["charb_eps"]) * conf_p4 * gate_p4).mean() * pw4

        L_msssim = (msssim_loss(pred, target, cfg["msssim_levels"]))  # [0,1]; we’ll weight & gate mildly
        # Light gating via mean gate (keeps structure term global but reduces cut impact)
        L_msssim = L_msssim * gate.mean()

        L_perc = self.perc(pred, target)

        # --- temporal smoothness (RGB TV) ---
        L_ttv = torch.tensor(0.0, device=device)
        if (pred_prev is not None) and (pred_next is not None):
            L_ttv = (pred - 0.5 * (pred_prev + pred_next)).abs().mean()

        # --- triplet latent consistency (RGB proxy if no latents) ---
        L_trip = torch.tensor(0.0, device=device)
        if (pred_prev is not None) and (pred_next is not None) and (target_prev is not None) and (target_next is not None):
            # consistency on reconstruction residuals as a proxy latent: (pred-gt) vs mid of neighbors
            res_mid = pred - target
            res_nb  = 0.5 * ((pred_prev - target_prev) + (pred_next - target_next))
            L_trip = (res_mid - res_nb).abs().mean()

        # --- weights hygiene ---
        L_went = torch.tensor(0.0, device=device)
        L_cover = torch.tensor(0.0, device=device)
        if weights is not None:
            p = torch.clamp(weights, 1e-8, 1.0)
            H_w = -(p * p.log()).sum(dim=1).mean()
            L_went = -H_w  # maximize entropy → add negative

            # coverage only when bracketed
            tgt = target_time if target_time.ndim == 2 else target_time.unsqueeze(1)
            L_cover = self._bracket_coverage(weights, anchor_times, tgt)

        # --- attention & confidence shaping ---
        L_attn = attn_ent.mean()
        c0 = cfg["conf_target"]
        L_conf = ((conf_full - c0) ** 2).mean()

        # --- priors ---
        L_tv   = tv_loss(pred)
        yuv_pred = rgb_to_yuv(pred)
        yuv_tgt  = rgb_to_yuv(target)
        L_luma   = (yuv_pred[:, :1] - yuv_tgt[:, :1]).abs().mean()
        L_chroma = (yuv_pred[:, 1:] - yuv_tgt[:, 1:]).abs().mean()

        # --- total ---
        loss = (
            cfg["w_charb"] * L_charb +
            cfg["w_msssim"] * L_msssim +
            cfg["w_perc"] * L_perc +
            cfg["w_temporal_tv"] * L_ttv +
            cfg["w_triplet_latent"] * L_trip +
            cfg["w_weight_entropy"] * L_went +
            cfg["w_coverage"] * L_cover +
            cfg["w_attn_entropy"] * L_attn +
            cfg["w_conf_reg"] * L_conf +
            cfg["w_tv"] * L_tv +
            cfg["w_luma"] * L_luma +
            cfg["w_chroma"] * L_chroma
        )

        logs = {
            "loss/total": float(loss.detach().cpu()),
            "rec/charb": float(L_charb.detach().cpu()),
            "rec/msssim": float(L_msssim.detach().cpu()),
            "rec/perc": float(L_perc.detach().cpu()),
            "temp/tv": float(L_ttv.detach().cpu()),
            "temp/triplet": float(L_trip.detach().cpu()),
            "weights/entropy": float(L_went.detach().cpu()),
            "weights/coverage": float(L_cover.detach().cpu()),
            "attn/entropy": float(L_attn.detach().cpu()),
            "conf/reg": float(L_conf.detach().cpu()),
            "prior/tv": float(L_tv.detach().cpu()),
            "prior/luma": float(L_luma.detach().cpu()),
            "prior/chroma": float(L_chroma.detach().cpu()),
            "conf/mean": float(conf_full.mean().detach().cpu()),
        }
        return loss, logs
    
_loss_singleton = None

def tempo_loss(pred, target, aux, anchor_times, target_time,
               pred_prev=None, pred_next=None, target_prev=None, target_next=None,
               config=None):
    """
    Thin wrapper around LossComputer.forward() that keeps a singleton instance.
    Returns (loss, logs)
    """
    global _loss_singleton
    if _loss_singleton is None:
        _loss_singleton = LossComputer(config=config)

    return _loss_singleton(
        pred=pred,
        target=target,
        aux=aux,
        anchor_times=anchor_times,
        target_time=target_time,
        pred_prev=pred_prev,
        pred_next=pred_next,
        target_prev=target_prev,
        target_next=target_next,
    )
