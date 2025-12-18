# convnext_nafnet.py
# ConvNeXt Encoder + NAFNet Decoder with AdaLN-Zero temporal conditioning
# Optimized for maximum PSNR/SSIM in temporal frame synthesis

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# ==============================================================================
# AdaLN-Zero: Adaptive Layer Normalization with Zero-Init Gate
# ==============================================================================

class AdaLNZero(nn.Module):
    """
    Adaptive LayerNorm with zero-initialized gate (DiT-style).
    Modulates normalized features with learned scale (γ), shift (β), and gate (α).
    
    output = x + α * block(γ * LayerNorm(x) + β)
    
    For conv features: operates on channel dimension
    """
    def __init__(self, feature_dim: int, temporal_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temporal_dim, feature_dim * 3)
        )
        # Zero-init the projection so blocks initially act as identity
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        Args:
            x: [B, C, H, W] feature maps
            t_emb: [B, Ct] temporal embedding
        Returns:
            (normed, γ, β, α) for use in blocks
        """
        B, C, H, W = x.shape
        
        # Project temporal embedding to modulation parameters
        params = self.proj(t_emb)  # [B, C*3]
        γ, β, α = params.chunk(3, dim=-1)  # each [B, C]
        
        # Reshape for broadcasting: [B, C, 1, 1]
        γ = γ.view(B, C, 1, 1)
        β = β.view(B, C, 1, 1)
        α = α.view(B, C, 1, 1)
        
        # Apply LayerNorm (need to permute for LayerNorm which expects [..., C])
        x_perm = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_norm = self.norm(x_perm).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Modulated features
        x_mod = γ * x_norm + β
        
        return x_mod, α


class AdaLNZeroSimple(nn.Module):
    """
    Simplified AdaLN for decoder where we just need scale+shift (no gate).
    """
    def __init__(self, feature_dim: int, temporal_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temporal_dim, feature_dim * 2)
        )
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        B, C, H, W = x.shape
        
        params = self.proj(t_emb)
        γ, β = params.chunk(2, dim=-1)
        γ = γ.view(B, C, 1, 1)
        β = β.view(B, C, 1, 1)
        
        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x_perm).permute(0, 3, 1, 2)
        
        return γ * x_norm + β


# ==============================================================================
# ConvNeXt Components
# ==============================================================================

class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first tensors (B, C, H, W)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


class DropPath(nn.Module):
    """Stochastic depth (drop path) for regularization."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block with AdaLN-Zero temporal conditioning.
    
    Structure:
        x → AdaLN(x, t) → 7×7 DWConv → LayerNorm → 1×1 Conv → GELU → 1×1 Conv → × α → + x
    """
    def __init__(
        self,
        dim: int,
        temporal_dim: int,
        expansion_ratio: int = 4,
        kernel_size: int = 7,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6
    ):
        super().__init__()
        
        hidden_dim = dim * expansion_ratio
        padding = kernel_size // 2
        
        # AdaLN-Zero for temporal conditioning
        self.adaln = AdaLNZero(dim, temporal_dim)
        
        # Depthwise conv (large kernel for receptive field)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim)
        
        # Normalization after dwconv
        self.norm = LayerNorm2d(dim)
        
        # Pointwise expansion and contraction (inverted bottleneck)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        
        # Layer scale (learnable per-channel scaling)
        self.layer_scale = nn.Parameter(
            layer_scale_init * torch.ones(dim)
        ) if layer_scale_init > 0 else None
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        Args:
            x: [B, C, H, W]
            t_emb: [B, Ct]
        """
        shortcut = x
        
        # AdaLN modulation
        x, α = self.adaln(x, t_emb)
        
        # Depthwise conv
        x = self.dwconv(x)
        x = self.norm(x)
        
        # Pointwise MLPs (permute for linear layers)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Layer scale
        if self.layer_scale is not None:
            x = x * self.layer_scale
        
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Apply gate and residual
        x = shortcut + self.drop_path(α * x)
        
        return x


class ConvNeXtStage(nn.Module):
    """A stage of ConvNeXt blocks with optional downsampling."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        temporal_dim: int,
        num_blocks: int,
        downsample: bool = True,
        drop_path_rates: list = None
    ):
        super().__init__()
        
        # Downsampling layer (if needed)
        if downsample:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_dim),
                nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
            )
        else:
            if in_dim != out_dim:
                self.downsample = nn.Sequential(
                    LayerNorm2d(in_dim),
                    nn.Conv2d(in_dim, out_dim, kernel_size=1)
                )
            else:
                self.downsample = nn.Identity()
        
        # ConvNeXt blocks
        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks
        
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(out_dim, temporal_dim, drop_path=drop_path_rates[i])
            for i in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x, t_emb)
        return x


# ==============================================================================
# NAFNet Components
# ==============================================================================

class SimpleGate(nn.Module):
    """Split-channel gating: x, y = split(input); output = x * y"""
    def forward(self, x: torch.Tensor):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """
    Simplified Channel Attention (SCA) from NAFNet.
    Uses global average pooling + 1×1 conv for channel reweighting.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(dim, dim, 1)
    
    def forward(self, x: torch.Tensor):
        attn = self.pool(x)
        attn = self.conv(attn)
        return x * attn


class NAFBlock(nn.Module):
    """
    NAFNet Block with AdaLN-Zero temporal conditioning.
    
    Structure:
        x → AdaLN(x, t) → 1×1 Conv → 3×3 DWConv → SimpleGate → SCA → 1×1 Conv → × α → + x
    
    Minimal nonlinearities for maximum information preservation.
    """
    def __init__(
        self,
        dim: int,
        temporal_dim: int,
        expansion_ratio: int = 2,
        kernel_size: int = 3,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        hidden_dim = dim * expansion_ratio
        padding = kernel_size // 2
        
        # AdaLN-Zero for temporal conditioning
        self.adaln = AdaLNZero(dim, temporal_dim)
        
        # Main path
        self.conv1 = nn.Conv2d(dim, hidden_dim, 1)  # Expand
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, 
                                 padding=padding, groups=hidden_dim)
        self.gate = SimpleGate()  # hidden_dim -> hidden_dim // 2
        self.sca = SimplifiedChannelAttention(hidden_dim // 2)
        self.conv2 = nn.Conv2d(hidden_dim // 2, dim, 1)  # Contract
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        shortcut = x
        
        # AdaLN modulation
        x, α = self.adaln(x, t_emb)
        
        # NAFNet path
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.gate(x)
        x = self.sca(x)
        x = self.conv2(x)
        
        # Gated residual
        x = shortcut + self.drop_path(α * x)
        
        return x


class NAFBlockNoGate(nn.Module):
    """NAFNet Block without AdaLN gate (simpler, for lighter decoder stages)."""
    def __init__(
        self,
        dim: int,
        temporal_dim: int,
        expansion_ratio: int = 2,
        kernel_size: int = 3,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        hidden_dim = dim * expansion_ratio
        padding = kernel_size // 2
        
        self.adaln = AdaLNZeroSimple(dim, temporal_dim)
        
        self.conv1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size,
                                 padding=padding, groups=hidden_dim)
        self.gate = SimpleGate()
        self.sca = SimplifiedChannelAttention(hidden_dim // 2)
        self.conv2 = nn.Conv2d(hidden_dim // 2, dim, 1)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        shortcut = x
        x = self.adaln(x, t_emb)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.gate(x)
        x = self.sca(x)
        x = self.conv2(x)
        return shortcut + self.drop_path(x)


# ==============================================================================
# ConvNeXt Encoder (Drop-in replacement for FrameEncoder)
# ==============================================================================

class ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt-based encoder with AdaLN-Zero temporal conditioning.
    
    Outputs features at 4 scales: 1×, 1/2×, 1/4×, 1/8×
    Same interface as original FrameEncoder.
    
    Default configuration optimized for quality:
        - Stage depths: [3, 3, 9, 3] (ConvNeXt-T inspired)
        - 7×7 depthwise convolutions
        - Layer scale + stochastic depth
    """
    def __init__(
        self,
        base_channels: int = 64,
        temporal_channels: int = 64,
        depths: list = None,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        
        if depths is None:
            depths = [3, 3, 18, 3]  # TEMPO BEAST: Scaled up for ~18M encoder params
        
        C = base_channels
        Ct = temporal_channels
        
        # Channel progression: C → C*2 → C*4 → C*8
        dims = [C, C * 2, C * 4, C * 8]
        
        # Stochastic depth rates (linear increase)
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # Stem: 3 → C at full resolution (no downsampling)
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=7, padding=3),
            LayerNorm2d(C)
        )
        
        # Stage 1: Full resolution (1×)
        self.stage1 = ConvNeXtStage(
            in_dim=dims[0],
            out_dim=dims[0],
            temporal_dim=Ct,
            num_blocks=depths[0],
            downsample=False,
            drop_path_rates=dpr[:depths[0]]
        )
        
        # Stage 2: 1/2× resolution
        self.stage2 = ConvNeXtStage(
            in_dim=dims[0],
            out_dim=dims[1],
            temporal_dim=Ct,
            num_blocks=depths[1],
            downsample=True,
            drop_path_rates=dpr[depths[0]:depths[0]+depths[1]]
        )
        
        # Stage 3: 1/4× resolution
        self.stage3 = ConvNeXtStage(
            in_dim=dims[1],
            out_dim=dims[2],
            temporal_dim=Ct,
            num_blocks=depths[2],
            downsample=True,
            drop_path_rates=dpr[depths[0]+depths[1]:depths[0]+depths[1]+depths[2]]
        )
        
        # Stage 4: 1/8× resolution
        self.stage4 = ConvNeXtStage(
            in_dim=dims[2],
            out_dim=dims[3],
            temporal_dim=Ct,
            num_blocks=depths[3],
            downsample=True,
            drop_path_rates=dpr[depths[0]+depths[1]+depths[2]:]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, frame: torch.Tensor, temporal_encoding: torch.Tensor):
        """
        Args:
            frame: [B, 3, H, W] input frame
            temporal_encoding: [B, Ct] temporal embedding
        
        Returns:
            f1: [B, C, H, W] - full resolution features
            f2: [B, C*2, H/2, W/2] - 1/2× features
            f3: [B, C*4, H/4, W/4] - 1/4× features
            f4: [B, C*8, H/8, W/8] - 1/8× features
        """
        x = self.stem(frame)
        
        f1 = self.stage1(x, temporal_encoding)
        f2 = self.stage2(f1, temporal_encoding)
        f3 = self.stage3(f2, temporal_encoding)
        f4 = self.stage4(f3, temporal_encoding)
        
        return f1, f2, f3, f4


# ==============================================================================
# NAFNet Decoder (Drop-in replacement for DecoderTimeFiLM)
# ==============================================================================

class NAFNetDecoder(nn.Module):
    """
    NAFNet-based decoder with AdaLN-Zero temporal conditioning.
    
    Takes fused features at 4 scales and reconstructs the output frame.
    Same interface as original DecoderTimeFiLM.
    
    Uses NAFNet blocks for maximum reconstruction quality (PSNR/SSIM).
    """
    def __init__(
        self,
        base_channels: int = 64,
        temporal_channels: int = 64,
        depths: list = None,
        drop_path_rate: float = 0.05
    ):
        super().__init__()
        
        if depths is None:
            depths = [3, 3, 9, 3]  # TEMPO BEAST: Scaled up for ~12M decoder params
        
        C = base_channels
        Ct = temporal_channels  # +1 for speed token handled by caller
        
        # Decoder goes from deep (C*8) to shallow (C)
        
        # Stage 4 → 3: 1/8× → 1/4×
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C * 8, C * 4, kernel_size=1)
        )
        self.merge4 = nn.Conv2d(C * 4 + C * 4, C * 4, kernel_size=1)  # Skip connection
        self.blocks4 = nn.ModuleList([
            NAFBlock(C * 4, Ct, drop_path=drop_path_rate)
            for _ in range(depths[0])
        ])
        
        # Stage 3 → 2: 1/4× → 1/2×
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C * 4, C * 2, kernel_size=1)
        )
        self.merge3 = nn.Conv2d(C * 2 + C * 2, C * 2, kernel_size=1)
        self.blocks3 = nn.ModuleList([
            NAFBlock(C * 2, Ct, drop_path=drop_path_rate)
            for _ in range(depths[1])
        ])
        
        # Stage 2 → 1: 1/2× → 1×
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C * 2, C, kernel_size=1)
        )
        self.merge2 = nn.Conv2d(C + C, C, kernel_size=1)
        self.blocks2 = nn.ModuleList([
            NAFBlock(C, Ct, drop_path=drop_path_rate)
            for _ in range(depths[2])
        ])
        
        # Final refinement and output
        self.refine = nn.ModuleList([
            NAFBlockNoGate(C, Ct) for _ in range(depths[3])
        ])

        # TEMPO BEAST: Heteroscedastic uncertainty head
        # Predicts per-pixel log-variance for uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(C // 2, 1, 1),
            nn.Softplus()  # Ensures positive output: log(σ²)
        )

        self.to_rgb = nn.Sequential(
            LayerNorm2d(C),
            nn.Conv2d(C, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
        tenc_with_speed: torch.Tensor
    ):
        """
        Args:
            f1: [B, C, H, W] - full resolution fused features
            f2: [B, C*2, H/2, W/2] - 1/2× fused features
            f3: [B, C*4, H/4, W/4] - 1/4× fused features
            f4: [B, C*8, H/8, W/8] - 1/8× fused features
            tenc_with_speed: [B, Ct+1] temporal encoding with speed token

        Returns:
            out: [B, 3, H, W] - reconstructed frame
            uncertainty_log_var: [B, 1, H, W] - heteroscedastic uncertainty (log variance)
        """
        t_emb = tenc_with_speed
        
        # 1/8× → 1/4×
        x = self.up4(f4)
        x = self.merge4(torch.cat([x, f3], dim=1))
        for block in self.blocks4:
            x = block(x, t_emb)
        
        # 1/4× → 1/2×
        x = self.up3(x)
        x = self.merge3(torch.cat([x, f2], dim=1))
        for block in self.blocks3:
            x = block(x, t_emb)
        
        # 1/2× → 1×
        x = self.up2(x)
        x = self.merge2(torch.cat([x, f1], dim=1))
        for block in self.blocks2:
            x = block(x, t_emb)
        
        # Final refinement
        for block in self.refine:
            x = block(x, t_emb)

        # TEMPO BEAST: Predict heteroscedastic uncertainty (per-pixel log-variance)
        uncertainty_log_var = self.uncertainty_head(x)

        # To RGB
        out = self.to_rgb(x)

        return out, uncertainty_log_var


# ==============================================================================
# Factory functions (for easy drop-in replacement)
# ==============================================================================

def build_convnext_encoder(
    base_channels: int = 64,
    temporal_channels: int = 64,
    depths: list = None,
    drop_path_rate: float = 0.1
):
    """
    Build ConvNeXt encoder.
    
    Presets:
        - depths=[3, 3, 9, 3] : Quality-focused (default, ~18 blocks)
        - depths=[2, 2, 6, 2] : Balanced (~12 blocks)
        - depths=[2, 2, 2, 2] : Fast (~8 blocks)
    """
    return ConvNeXtEncoder(base_channels, temporal_channels, depths, drop_path_rate)


def build_nafnet_decoder(
    base_channels: int = 64,
    temporal_channels: int = 64,
    depths: list = None,
    drop_path_rate: float = 0.05
):
    """
    Build NAFNet decoder.
    
    Presets:
        - depths=[2, 2, 2, 2] : Standard (default)
        - depths=[3, 3, 3, 3] : Heavier refinement
        - depths=[1, 1, 2, 2] : Lightweight
    """
    return NAFNetDecoder(base_channels, temporal_channels, depths, drop_path_rate)


# ==============================================================================
# Quick test
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test encoder
    encoder = build_convnext_encoder(base_channels=64, temporal_channels=64).to(device)
    frame = torch.randn(2, 3, 256, 256).to(device)
    t_emb = torch.randn(2, 64).to(device)
    
    f1, f2, f3, f4 = encoder(frame, t_emb)
    print(f"Encoder outputs:")
    print(f"  f1: {f1.shape}")  # [2, 64, 256, 256]
    print(f"  f2: {f2.shape}")  # [2, 128, 128, 128]
    print(f"  f3: {f3.shape}")  # [2, 256, 64, 64]
    print(f"  f4: {f4.shape}")  # [2, 512, 32, 32]
    
    # Test decoder
    decoder = build_nafnet_decoder(base_channels=64, temporal_channels=65).to(device)
    t_emb_with_speed = torch.randn(2, 65).to(device)  # +1 for speed token
    
    out = decoder(f1, f2, f3, f4, t_emb_with_speed)
    print(f"\nDecoder output: {out.shape}")  # [2, 3, 256, 256]
    
    # Parameter counts
    enc_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    dec_params = sum(p.numel() for p in decoder.parameters()) / 1e6
    print(f"\nEncoder params: {enc_params:.2f}M")
    print(f"Decoder params: {dec_params:.2f}M")
    print(f"Total: {enc_params + dec_params:.2f}M")
