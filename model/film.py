# film.py
# TEMPO V3 - Enhanced Feature Extraction
# Upgraded encoder/decoder with:
#   - 6+ ResBlocks per scale (vs 1-2 in V2)
#   - Squeeze-and-Excitation channel attention
#   - Multi-dilation receptive fields
#   - Enhanced temporal fusion with spatial attention
#   - Pre-activation design for better gradient flow

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# Core Building Blocks
# ==========================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.fc(x)


class EnhancedResBlock(nn.Module):
    """
    Enhanced residual block with:
    - Pre-activation (better gradient flow)
    - Squeeze-and-Excitation
    - Optional dilation for larger receptive field
    """
    def __init__(self, channels, dilation=1, use_se=True):
        super().__init__()
        padding = dilation
        self.norm1 = nn.GroupNorm(min(32, channels // 4), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(min(32, channels // 4), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, padding, dilation=dilation)
        self.act = nn.GELU()
        self.se = SEBlock(channels) if use_se else nn.Identity()
        
    def forward(self, x):
        identity = x
        
        out = self.act(self.norm1(x))
        out = self.conv1(out)
        out = self.act(self.norm2(out))
        out = self.conv2(out)
        out = self.se(out)
        
        return out + identity


class ResidualGroup(nn.Module):
    """
    Group of residual blocks with varying dilations
    for multi-scale receptive fields
    """
    def __init__(self, channels, num_blocks=6):
        super().__init__()
        # Vary dilation to capture different scales
        dilations = [1, 1, 2, 1, 2, 1][:num_blocks]
        self.blocks = nn.ModuleList([
            EnhancedResBlock(channels, dilation=d, use_se=True)
            for d in dilations
        ])
        
        # Residual scaling (from EDSR)
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        res = x
        for block in self.blocks:
            x = block(x)
        return res + x * self.res_scale


class EnhancedTemporalFusion(nn.Module):
    """
    Enhanced temporal fusion with spatial attention.
    More powerful than simple FiLM.
    """
    def __init__(self, feature_channels, temporal_channels):
        super().__init__()
        
        # Direct projection from temporal to feature space (no dimension change)
        self.gamma_net = nn.Linear(temporal_channels, feature_channels)
        self.beta_net = nn.Linear(temporal_channels, feature_channels)
        
        # Spatial attention to focus modulation
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(feature_channels, max(1, feature_channels // 8), 1),
            nn.GELU(),
            nn.Conv2d(max(1, feature_channels // 8), 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features, temporal_encoding):
        B, C, H, W = features.shape
        
        # FiLM parameters (direct projection)
        gamma = self.gamma_net(temporal_encoding).view(B, C, 1, 1)
        beta = self.beta_net(temporal_encoding).view(B, C, 1, 1)
        
        # Apply FiLM
        modulated = gamma * features + beta
        
        # Spatial attention weighting
        attn_map = self.spatial_attn(features)
        
        # Blend original and modulated features
        return features + attn_map * (modulated - features)


# ==========================================
# ENHANCED ENCODER (V3)
# ==========================================

class FrameEncoder(nn.Module):
    """
    Enhanced encoder V3 with:
    - 6 ResBlocks per scale (vs 1 in V2)
    - Squeeze-and-Excitation attention
    - Multi-dilation receptive fields
    - Enhanced temporal fusion
    - Residual scaling for better training
    
    Expected improvement: +2-3 PSNR, +0.02-0.03 SSIM
    """
    def __init__(self, base_channels=64, temporal_channels=64, num_blocks_per_scale=6):
        super().__init__()
        C = base_channels
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, C, 7, 1, 3),
            nn.GroupNorm(min(32, C // 4), C),
            nn.GELU()
        )
        
        # Scale 1: Full resolution - Light processing
        self.group1 = ResidualGroup(C, num_blocks=num_blocks_per_scale - 2)
        self.temporal_fusion1 = EnhancedTemporalFusion(C, temporal_channels)
        
        # Scale 2: 1/2 resolution
        self.down1 = nn.Sequential(
            nn.Conv2d(C, C*2, 3, 2, 1),
            nn.GroupNorm(min(32, C // 2), C*2),
            nn.GELU()
        )
        self.group2 = ResidualGroup(C*2, num_blocks=num_blocks_per_scale)
        self.temporal_fusion2 = EnhancedTemporalFusion(C*2, temporal_channels)
        
        # Scale 3: 1/4 resolution
        self.down2 = nn.Sequential(
            nn.Conv2d(C*2, C*4, 3, 2, 1),
            nn.GroupNorm(min(32, C), C*4),
            nn.GELU()
        )
        self.group3 = ResidualGroup(C*4, num_blocks=num_blocks_per_scale)
        self.temporal_fusion3 = EnhancedTemporalFusion(C*4, temporal_channels)
        
        # Scale 4: 1/8 resolution - Most processing
        self.down3 = nn.Sequential(
            nn.Conv2d(C*4, C*8, 3, 2, 1),
            nn.GroupNorm(min(32, C*2), C*8),
            nn.GELU()
        )
        self.group4_a = ResidualGroup(C*8, num_blocks=num_blocks_per_scale)
        self.group4_b = ResidualGroup(C*8, num_blocks=num_blocks_per_scale)
        self.temporal_fusion4 = EnhancedTemporalFusion(C*8, temporal_channels)
        
        print(f"✨ FrameEncoder V3 initialized:")
        print(f"   - {num_blocks_per_scale} blocks per scale")
        print(f"   - SE channel attention enabled")
        print(f"   - Multi-dilation receptive fields (1, 2)")
        print(f"   - Enhanced temporal fusion with spatial attention")
        print(f"   - Total conv layers: ~{(num_blocks_per_scale - 2) * 2 + num_blocks_per_scale * 2 * 4}")
    
    def forward(self, frame, temporal_encoding):
        """
        frame: [B, 3, H, W]
        temporal_encoding: [B, C_t]
        Returns: (f1, f2, f3, f4) - Multi-scale features
        """
        # Initial
        x = self.conv1(frame)
        
        # Scale 1: Full res
        f1 = self.group1(x)
        f1 = self.temporal_fusion1(f1, temporal_encoding)
        
        # Scale 2: 1/2
        x = self.down1(f1)
        f2 = self.group2(x)
        f2 = self.temporal_fusion2(f2, temporal_encoding)
        
        # Scale 3: 1/4
        x = self.down2(f2)
        f3 = self.group3(x)
        f3 = self.temporal_fusion3(f3, temporal_encoding)
        
        # Scale 4: 1/8 (most computation here)
        x = self.down3(f3)
        f4 = self.group4_a(x)
        f4 = self.group4_b(f4)
        f4 = self.temporal_fusion4(f4, temporal_encoding)
        
        return f1, f2, f3, f4


# ==========================================
# ENHANCED DECODER (V3)
# ==========================================

class DecoderTimeFiLM(nn.Module):
    """
    Enhanced decoder V3 with:
    - Multiple ResBlocks at each scale
    - Progressive upsampling with refinement
    - SE attention
    - Enhanced temporal fusion
    
    Matches the encoder capacity for better reconstruction.
    
    NOTE: temporal_channels parameter already includes the +1 for speed token
          (tempo.py passes Ct + 1), so we use it directly without adding +1 again.
    """
    def __init__(self, base_channels=64, temporal_channels=64):
        super().__init__()
        C = base_channels
        
        # temporal_channels already includes the speed token (+1)
        # as passed from tempo.py: DecoderTimeFiLM(C, Ct + 1)
        
        # Upsample 1/8 -> 1/4
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C*8, C*4, 3, 1, 1),
            nn.GroupNorm(min(32, C), C*4),
            nn.GELU()
        )
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(C*4 + C*4, C*4, 3, 1, 1),  # Concat with skip
            EnhancedResBlock(C*4, use_se=True),
            EnhancedResBlock(C*4, use_se=True)
        )
        self.temporal_fusion1 = EnhancedTemporalFusion(C*4, temporal_channels)
        
        # Upsample 1/4 -> 1/2
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C*4, C*2, 3, 1, 1),
            nn.GroupNorm(min(32, C // 2), C*2),
            nn.GELU()
        )
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(C*2 + C*2, C*2, 3, 1, 1),
            EnhancedResBlock(C*2, use_se=True),
            EnhancedResBlock(C*2, use_se=True)
        )
        self.temporal_fusion2 = EnhancedTemporalFusion(C*2, temporal_channels)
        
        # Upsample 1/2 -> 1/1
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C*2, C, 3, 1, 1),
            nn.GroupNorm(min(32, C // 4), C),
            nn.GELU()
        )
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(C + C, C, 3, 1, 1),
            EnhancedResBlock(C, use_se=True),
            EnhancedResBlock(C, use_se=True),
            EnhancedResBlock(C, use_se=True)
        )
        self.temporal_fusion3 = EnhancedTemporalFusion(C, temporal_channels)
        
        # Final output with residual
        self.final = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(C, C // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(C // 2, 3, 3, 1, 1)
        )
        
        print(f"✨ DecoderTimeFiLM V3 initialized:")
        print(f"   - Progressive upsampling with refinement")
        print(f"   - 2-3 EnhancedResBlocks per upsampling stage")
        print(f"   - Enhanced temporal fusion at each scale")
        print(f"   - Accepts temporal dim: {temporal_channels} (includes speed token)")
    
    def forward(self, f1, f2, f3, f4, tenc_with_speed):
        """
        f1-f4: Multi-scale features from encoder
        tenc_with_speed: [B, C_t + 1] (temporal encoding + speed token)
        """
        # Decode 1/8 -> 1/4
        x = self.up1(f4)
        x = torch.cat([x, f3], dim=1)
        x = self.conv_up1(x)
        x = self.temporal_fusion1(x, tenc_with_speed)
        
        # Decode 1/4 -> 1/2
        x = self.up2(x)
        x = torch.cat([x, f2], dim=1)
        x = self.conv_up2(x)
        x = self.temporal_fusion2(x, tenc_with_speed)
        
        # Decode 1/2 -> 1/1
        x = self.up3(x)
        x = torch.cat([x, f1], dim=1)
        x = self.conv_up3(x)
        x = self.temporal_fusion3(x, tenc_with_speed)
        
        # Final output
        out = self.final(x)
        return torch.sigmoid(out)
