# model/pyramid.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================
# Time-Aware Sliding Window Pyramid Attention (with fixes)
# =====================================

class TimeAwareSlidingWindowPyramidAttention(nn.Module):
    """
    Windowed deformable cross-attention with:
      - FiLM from target-time encoding on offsets/attn towers
      - Δt bias added to attention logits per-view
      - Head-wise offset scaling as a function of |Δt|
      - Outputs per-pixel confidence and attention entropy for occlusion handling
    """
    def __init__(self, channels, num_heads=4, num_points=4, num_levels=3,
                 window_size=8, shift_size=0, init_spatial_range=0.1,
                 temporal_channels=64, dt_bias_gain=1.0, max_offset_scale=1.5):
        super().__init__()
        assert channels % num_heads == 0
        assert 0 <= shift_size < window_size

        self.C = channels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_levels = num_levels
        self.head_dim = channels // num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.dt_bias_gain = dt_bias_gain
        self.max_offset_scale = max_offset_scale

        # towers
        self.sampling_offsets = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.GroupNorm(min(8, channels // 4), channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, num_heads * num_levels * num_points * 2, 1)
        )
        self.attention_weights = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.GroupNorm(min(8, channels // 4), channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, num_heads * num_levels * num_points, 1)
        )
        self.value_proj  = nn.Conv2d(channels, channels, 1)
        self.output_proj = nn.Conv2d(channels, channels, 1)

        # FiLM from target-time enc
        self.film_offsets = nn.Linear(temporal_channels, channels)
        self.film_attn    = nn.Linear(temporal_channels, channels)

        # Δt -> offset scale (per head)
        self.dt_scale_head = nn.Linear(1, num_heads)
        self.level_embed = nn.Parameter(torch.zeros(num_levels, channels))

        # Occlusion/confidence heads
        self.conf_head = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1)
        )

        self._reset_parameters(init_spatial_range)

    def _reset_parameters(self, init_range):
        nn.init.constant_(self.sampling_offsets[-1].weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]) * init_range
        grid = grid.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid[:, :, i, :] *= (i + 1) / self.num_points
        for lvl in range(self.num_levels):
            grid[:, lvl, :, :] *= (2 ** lvl) * 0.1
        with torch.no_grad():
            self.sampling_offsets[-1].bias.copy_(grid.view(-1))
        nn.init.constant_(self.attention_weights[-1].weight.data, 0.)
        nn.init.constant_(self.attention_weights[-1].bias.data, 0.)
        nn.init.normal_(self.level_embed, 0.0, 0.02)

    # --- Window helpers ---
    @staticmethod
    def window_partition(x, ws):
        B, C, H, W = x.shape
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H_pad, W_pad = H + pad_h, W + pad_w
        else:
            H_pad, W_pad = H, W
        H_w, W_w = H_pad // ws, W_pad // ws
        x = x.view(B, C, H_w, ws, W_w, ws).permute(0, 2, 4, 1, 3, 5).contiguous()
        windows = x.view(B * H_w * W_w, C, ws, ws)
        return windows, (H_w, W_w), (H_pad, W_pad), (H, W)

    @staticmethod
    def window_reverse(windows, ws, H_w, W_w, B, H_pad, W_pad, H, W):
        C = windows.shape[1]
        x = windows.view(B, H_w, W_w, C, ws, ws).permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, H_pad, W_pad)
        if H_pad > H or W_pad > W:
            x = x[:, :, :H, :W]
        return x

    def forward(self, query, values, rel_scalar, tgt_enc):
        """
        query  : [B,C,H,W] (time-weighted blend)
        values : [B,N,C,H,W] (per-view features)
        rel_scalar: [B,N] (normalized Δt per view)
        tgt_enc: [B,Ct] target-time encoding (FiLM)
        Returns: fused [B,C,H,W], conf_map [B,1,H,W], attn_entropy [B,1,H,W]
        """
        device = query.device
        B, C, H, W = query.shape
        N = min(values.shape[1], self.num_levels)
        ws = self.window_size

        # (Optional) Shift
        if self.shift_size > 0 and (H > 16 and W > 16):
            query = torch.roll(query, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            values = torch.stack([
                torch.roll(values[:, i], shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
                for i in range(values.shape[1])
            ], dim=1)

        # Windows
        q_win, (H_w, W_w), (H_pad, W_pad), (Ho, Wo) = self.window_partition(query, ws)
        num_win = H_w * W_w
        v_wins = []
        for i in range(N):
            v_win, *_ = self.window_partition(values[:, i], ws)
            v_wins.append(v_win)  # [B*nW,C,ws,ws]

        # FiLM from target time on towers
        film_o = self.film_offsets(tgt_enc).view(B, self.C, 1, 1)  # [B,C,1,1]
        film_a = self.film_attn(tgt_enc).view(B, self.C, 1, 1)
        film_o = film_o.repeat(num_win, 1, ws, ws)  # [B*nW,C,ws,ws]
        film_a = film_a.repeat(num_win, 1, ws, ws)

        # Towers
        off_in = q_win + film_o
        att_in = q_win + film_a
        offsets = self.sampling_offsets(off_in)  # [B*nW, heads*lvls*pts*2, ws, ws]
        attn    = self.attention_weights(att_in) # [B*nW, heads*lvls*pts, ws, ws]

        # Reshape attention for biasing
        attn = attn.view(B * num_win, self.num_heads, self.num_levels, self.num_points, ws, ws)
        attn = attn.permute(0, 4, 5, 1, 2, 3).contiguous()  # [B*nW, ws, ws, heads, lvls, pts]

        # --- Δt bias on logits (closer → higher) with proper tiling across windows ---
        dt_per_view = rel_scalar[:, :N]  # [B,N]
        dt = (
            dt_per_view
            .unsqueeze(1)  # [B,1,N]
            .unsqueeze(1)  # [B,1,1,N]
            .unsqueeze(1)  # [B,1,1,1,N]
            .unsqueeze(-1) # [B,1,1,1,N,1]
            .repeat(1, num_win, ws, ws, self.num_heads, 1, self.num_points)
            .view(B * num_win, ws, ws, self.num_heads, N, self.num_points)
        )

        attn_logits = attn.view(B * num_win, ws, ws, self.num_heads, self.num_levels * self.num_points)
        bias = - self.dt_bias_gain * dt.abs().view(B * num_win, ws, ws, self.num_heads, N * self.num_points)
        if self.num_levels > N:
            pad = torch.zeros(
                B * num_win, ws, ws, self.num_heads,
                (self.num_levels - N) * self.num_points, device=device
            )
            bias = torch.cat([bias, pad], dim=-1)

        attn = F.softmax(attn_logits + bias, dim=-1).view(
            B * num_win, ws, ws, self.num_heads, self.num_levels, self.num_points
        )

        # --- 4D reference grid (fixes grid_sample batch mismatch) ---
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, 1, ws, device=device),
            torch.linspace(0, 1, ws, device=device),
            indexing='ij'
        )
        ref_grid_simple = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0)  # [1, ws, ws, 2]

        # Reshape offsets to [B*nW, ws, ws, heads, lvls, pts, 2]
        offsets = offsets.view(B * num_win, self.num_heads, self.num_levels, self.num_points, 2, ws, ws)
        offsets = offsets.permute(0, 5, 6, 1, 2, 3, 4).contiguous()  # [B*nW, ws, ws, H, L, P, 2]

        out = torch.zeros(B * num_win, self.C, ws, ws, device=device)

        # per-view |Δt| for head-wise offset scaling
        dt_abs = rel_scalar[:, :N].abs().unsqueeze(-1)  # [B,N,1]

        for lvl in range(N):
            v_lvl = self.value_proj(v_wins[lvl] + self.level_embed[lvl].view(1, -1, 1, 1))
            for head in range(self.num_heads):
                hs, he = head * self.head_dim, (head + 1) * self.head_dim
                v_head = v_lvl[:, hs:he]  # [B*nW, head_dim, ws, ws]

                # per-head scale from |Δt| → [B,1] → [B*nW,1,1,1]
                hscale = 1.0 + (self.max_offset_scale - 1.0) * torch.sigmoid(self.dt_scale_head(dt_abs[:, lvl:lvl+1]))  # [B,H]
                hscale = hscale[..., head:head+1]  # [B,1]
                hscale = hscale.repeat_interleave(num_win, dim=0).view(B * num_win, 1, 1, 1)  # [B*nW,1,1,1]

                for pt in range(self.num_points):
                    # offsets for this (head, level, point): [B*nW, ws, ws, 2]
                    off = offsets[:, :, :, head, lvl, pt, :]  # [B*nW, ws, ws, 2]

                    # grid: ref + off, scaled, clamped to [-1,1]
                    xy = (ref_grid_simple + off).clamp(0, 1)   # [B*nW, ws, ws, 2]
                    xy = 2.0 * xy - 1.0
                    xy = torch.clamp(xy * hscale, -1.0, 1.0)
                    
                    padding_mode = 'zero' if torch.backends.mps.is_available() else 'border'
                    sampled = F.grid_sample(v_head, xy, mode='bilinear',
                                            align_corners=False, padding_mode=padding_mode)
                    w = attn[:, :, :, head, lvl, pt].unsqueeze(1)  # [B*nW,1,ws,ws]
                    out[:, hs:he] += sampled * w

        # attention entropy per pixel (over all L*P per head, averaged over heads)
        probs = (attn_logits + bias).softmax(dim=-1) + 1e-8
        ent = -(probs * probs.log()).sum(dim=-1)  # [B*nW, ws, ws, heads]
        attn_entropy = ent.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2).contiguous()  # [B*nW,1,ws,ws]

        fused = self.output_proj(out)
        fused = self.window_reverse(fused, ws, H_w, W_w, B, H_pad, W_pad, Ho, Wo)

        # reverse shift
        if self.shift_size > 0 and (H > 16 and W > 16):
            fused = torch.roll(fused, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        # confidence map (higher = more certain)
        conf_map = torch.sigmoid(self.conf_head(fused))  # [B,1,H,W]
        attn_entropy = self.window_reverse(attn_entropy, ws, H_w, W_w, B, H_pad, W_pad, Ho, Wo)

        return fused, conf_map, attn_entropy