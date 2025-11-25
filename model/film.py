# film.py
# Pure ConvNeXt Encoder/Decoder with FiLM temporal conditioning
# Architecture:
#   Stage 1 (H/4):  ConvNeXt [96 ch,  3 blocks]
#   Stage 2 (H/8):  ConvNeXt [192 ch, 6 blocks]
#   Stage 3 (H/16): ConvNeXt [384 ch, 9 blocks]
#   Stage 4 (H/32): ConvNeXt [768 ch, 3 blocks]
# Total: 21 ConvNeXt blocks, ~49M params

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.layers import DropPath, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, trunc_normal_
import math

# ==========================
# Temporal FiLM conditioning
# ==========================

class TemporalFiLM(nn.Module):
    def __init__(self, feature_channels, temporal_channels):
        super().__init__()
        self.gamma_net = nn.Linear(temporal_channels, feature_channels)
        self.beta_net  = nn.Linear(temporal_channels, feature_channels)

    def forward(self, features, temporal_encoding):  # features: [B,C,H,W], enc: [B,Ct]
        gamma = self.gamma_net(temporal_encoding).unsqueeze(-1).unsqueeze(-1)
        beta  = self.beta_net(temporal_encoding).unsqueeze(-1).unsqueeze(-1)
        return gamma * features + beta


# ==========================
# LayerNorm (channel-first for CNNs)
# ==========================

class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# ==========================
# ConvNeXt Block
# ==========================

class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: DWConv 7x7 → LayerNorm → FFN (1x1 conv via Linear) → Layer Scale"""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # Depthwise
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Pointwise/1x1 conv via Linear
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] → [B, H, W, C]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W]
        x = input + self.drop_path(x)
        return x


# ==========================
# Window-based Multi-Head Self-Attention
# ==========================

class WindowAttention(nn.Module):
    """Window based multi-head self attention with relative position bias"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B*nW, Wh*Ww, C]
            mask: (0/-inf) mask with shape [nW, Wh*Ww, Wh*Ww] or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ==========================
# Swin Transformer Block
# ==========================

def window_partition(x, window_size):
    """Partition into non-overlapping windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with shifted windows"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, 
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        """x: [B, C, H, W]"""
        B, C, H, W = x.shape
        
        # Convert to [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        shortcut = x

        # Pad if needed
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._create_mask(Hp, Wp).to(x.device)
        else:
            shifted_x = x
            attn_mask = None

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, ws, ws, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(self.norm1(x_windows), mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Unpad if needed
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # Back to [B, C, H, W]
        return x.permute(0, 3, 1, 2).contiguous()

    def _create_mask(self, H, W):
        """Create attention mask for SW-MSA"""
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


# ==========================
# Cross-Attention Block (Decoder queries Encoder)
# ==========================




# ==========================
# Hybrid Swin+ConvNeXt Encoder
# ==========================

class FrameEncoder(nn.Module):
    """
    Pure ConvNeXt Encoder (NO Swin Transformer)
    
    Architecture:
    - Stage 1 (H/4):  ConvNeXt [96 ch,  3 blocks]
    - Stage 2 (H/8):  ConvNeXt [192 ch, 6 blocks]
    - Stage 3 (H/16): ConvNeXt [384 ch, 9 blocks]
    - Stage 4 (H/32): ConvNeXt [768 ch, 3 blocks]
    
    Total: 21 ConvNeXt blocks, ~49M params
    """
    def __init__(self, base_channels=64, temporal_channels=64, drop_path_rate=0.1):
        super().__init__()
        
        dims = [96, 192, 384, 768]
        depths = [3, 6, 9, 3]  # All ConvNeXt
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Stem: Patchify 4x4
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        # Stage 1: H/4
        cur = 0
        self.stage1 = nn.ModuleList([
            ConvNeXtBlock(dim=dims[0], drop_path=dpr[cur + i]) 
            for i in range(depths[0])
        ])
        cur += depths[0]
        self.film1 = TemporalFiLM(dims[0], temporal_channels)
        self.downsample1 = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)
        )
        
        # Stage 2: H/8 - Pure ConvNeXt
        self.stage2 = nn.ModuleList([
            ConvNeXtBlock(dim=dims[1], drop_path=dpr[cur + i]) 
            for i in range(depths[1])
        ])
        cur += depths[1]
        self.film2 = TemporalFiLM(dims[1], temporal_channels)
        self.downsample2 = nn.Sequential(
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)
        )
        
        # Stage 3: H/16 - Pure ConvNeXt
        self.stage3 = nn.ModuleList([
            ConvNeXtBlock(dim=dims[2], drop_path=dpr[cur + i]) 
            for i in range(depths[2])
        ])
        cur += depths[2]
        self.film3 = TemporalFiLM(dims[2], temporal_channels)
        self.downsample3 = nn.Sequential(
            LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2)
        )
        
        # Stage 4: H/32 - Pure ConvNeXt
        self.stage4 = nn.ModuleList([
            ConvNeXtBlock(dim=dims[3], drop_path=dpr[cur + i])
            for i in range(depths[3])
        ])
        self.film4 = TemporalFiLM(dims[3], temporal_channels)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, frame, temporal_encoding):
        """
        Args:
            frame: [B, 3, H, W]
            temporal_encoding: [B, Ct]
        Returns:
            f1, f2, f3, f4: Multi-scale features at [H/4, H/8, H/16, H/32]
        """
        # Stem
        x = self.stem(frame)
        
        # Stage 1
        for block in self.stage1:
            x = block(x)
        f1 = self.film1(x, temporal_encoding)
        x = self.downsample1(f1)
        
        # Stage 2 - Pure ConvNeXt
        for block in self.stage2:
            x = block(x)
        f2 = self.film2(x, temporal_encoding)
        x = self.downsample2(f2)
        
        # Stage 3 - Pure ConvNeXt
        for block in self.stage3:
            x = block(x)
        f3 = self.film3(x, temporal_encoding)
        x = self.downsample3(f3)
        
        # Stage 4 - Pure ConvNeXt
        for block in self.stage4:
            x = block(x)
        f4 = self.film4(x, temporal_encoding)
        
        return f1, f2, f3, f4


# ==========================
# Progressive Decoder with Cross-Attention
# ==========================

class UpsampleStage(nn.Module):
    """
    Single upsampling stage with:
    1. Upsample current features 2x
    2. Concatenate with encoder skip connection (U-Net style)
    3. Fusion + Refinement with ConvNeXt blocks
    4. Temporal FiLM conditioning
    """
    def __init__(self, in_dim, skip_dim, out_dim, 
                 num_refine_blocks=2, temporal_channels=65):
        super().__init__()
        
        # Upsample current features
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_dim, out_dim, 1),
            LayerNorm(out_dim, eps=1e-6, data_format="channels_first")
        )
        
        # Fusion of upsampled + skip features
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim + skip_dim, out_dim, 1),
            LayerNorm(out_dim, eps=1e-6, data_format="channels_first"),
            nn.GELU()
        )
        
        # Refinement blocks
        self.refine = nn.ModuleList([
            ConvNeXtBlock(out_dim) for _ in range(num_refine_blocks)
        ])
        
        # Temporal FiLM
        self.film = TemporalFiLM(out_dim, temporal_channels)
        
    def forward(self, x, skip, tenc):
        # Upsample decoder features
        x_up = self.upsample(x)  # [B, out_dim, H, W]
        
        # Robust skip connection: resize skip to match x_up if needed
        # This handles odd resolutions where H/2 * 2 != H
        if skip.shape[-2:] != x_up.shape[-2:]:
            skip = F.interpolate(skip, size=x_up.shape[-2:], mode='bilinear', align_corners=False)
        
        # Fuse (Concatenate)
        x_fused = self.fusion(torch.cat([x_up, skip], dim=1))
        
        # Refine with ConvNeXt blocks
        for block in self.refine:
            x_fused = block(x_fused)
            
        # Temporal conditioning
        x_out = self.film(x_fused, tenc)
        
        return x_out


class DecoderTimeFiLM(nn.Module):
    """
    Progressive decoder with cross-attention
    Upsamples from H/32 → H/4, then final 4x upsample to H
    """
    def __init__(self, base_channels=64, temporal_channels=64):
        super().__init__()
        
        # Note: base_channels ignored, using ConvNeXt-T dims
        # temporal_channels already includes +1 for speed token (passed as Ct+1 from TEMPO)
        temporal_channels_with_speed = temporal_channels
        
        # Upsampling stages (H/32 → H/16 → H/8 → H/4)
        # Upsampling stages (H/32 → H/16 → H/8 → H/4)
        self.up1 = UpsampleStage(
            in_dim=768, skip_dim=384, out_dim=384,
            num_refine_blocks=2,
            temporal_channels=temporal_channels_with_speed
        )
        
        self.up2 = UpsampleStage(
            in_dim=384, skip_dim=192, out_dim=192,
            num_refine_blocks=2,
            temporal_channels=temporal_channels_with_speed
        )
        
        self.up3 = UpsampleStage(
            in_dim=192, skip_dim=96, out_dim=96,
            num_refine_blocks=2,
            temporal_channels=temporal_channels_with_speed
        )
        
        # Final 4x upsample to full resolution
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(96, 96, 3, padding=1),
            LayerNorm(96, eps=1e-6, data_format="channels_first"),
            nn.GELU()
        )
        
        # Final refinement
        self.final_refine = nn.Sequential(
            ConvNeXtBlock(96),
            ConvNeXtBlock(96)
        )
        
        # To RGB
        self.to_rgb = nn.Sequential(
            nn.Conv2d(96, 3, 1),
            nn.Sigmoid()
        )
        
    def forward(self, f1, f2, f3, f4, tenc_with_speed):
        """
        Args:
            f1: [B, 96, H/4, W/4]
            f2: [B, 192, H/8, W/8]
            f3: [B, 384, H/16, W/16]
            f4: [B, 768, H/32, W/32]
            tenc_with_speed: [B, Ct+1]
        Returns:
            RGB output [B, 3, H, W]
        """
        # Progressive upsampling with cross-attention
        x = self.up1(f4, f3, tenc_with_speed)  # H/32 → H/16
        x = self.up2(x, f2, tenc_with_speed)   # H/16 → H/8
        x = self.up3(x, f1, tenc_with_speed)   # H/8 → H/4
        
        # Final upsample to full resolution
        x = self.final_upsample(x)             # H/4 → H
        
        # Final refinement
        x = self.final_refine(x)
        
        # To RGB
        out = self.to_rgb(x)
        
        return out
