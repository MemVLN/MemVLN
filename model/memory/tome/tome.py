import math
from typing import Any, Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn


# ---------------- ToMeAttention (trainable) ----------------
class ToMeAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, C)
        returns:
          - attn_out: (B, N, C)
          - k_mean: (B, N, head_dim)  (mean over heads of k)
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        k_mean = k.mean(1)  # (B, N, head_dim)
        return out, k_mean


# ---------------- ToMeBlock (attention -> merge -> mlp) ----------------
class ToMeBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0):
        """
        A single block that:
         - norm1 -> ToMeAttention -> residual add
         - (external) merge will be applied after this step using metric from attention
         - norm2 -> MLP -> residual add
        Note: merge is invoked externally so this block only returns (x_after_attn, metric).
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ToMeAttention(dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward_attn(self, x: torch.Tensor, size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply norm1 then ToMeAttention and residual add; return (x_after_attn, metric).
        x: (B, N, C)
        metric: (B, N, head_dim)
        """
        x_norm = self.norm1(x)
        x_attn, metric = self.attn(x_norm, size=size)
        x = x + x_attn
        return x, metric

    def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply norm2 -> mlp -> residual add
        """
        x_m = self.mlp(self.norm2(x))
        x = x + x_m
        return x