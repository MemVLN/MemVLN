import math
import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):
    def __init__(self, mode, dim, fourier_feats=32, learnable=False, mlp_hidden=128, mapping_type='nerf'):
        super().__init__()
        self.mode = mode
        self.dim = dim
        self.mapping_type = mapping_type # 'nerf' (axis-aligned) or 'rff' (random matrix)
        self.learnable = learnable
        
        # Input: dx, dy, sin(yaw), cos(yaw) (4 dims)
        self.in_dim = 4
        
        if mode == 'fourier':
            self.fourier_feats = fourier_feats
            
            if self.mapping_type == 'rff':
                # Gaussian Mapping: Mixes dimensions (B, 4) @ (4, F) -> (B, F)
                # Standard deviation controls the bandwidth
                if learnable:
                    self.rff_proj = nn.Linear(self.in_dim, fourier_feats, bias=False)
                else:
                    W = torch.randn(self.in_dim, fourier_feats) * 2.0 * torch.pi
                    self.register_buffer('W', W)
                ff_dim = fourier_feats * 2
            
            else: # 'nerf' style (Axis-aligned)
                # (B, 4, 1) * (1, 4, F) -> (B, 3, F)
                if learnable:
                    self.W = nn.Parameter(torch.randn(self.in_dim, fourier_feats)) # BUG
                else:
                    freqs = 2.0 ** torch.arange(fourier_feats).float()
                    W = freqs.unsqueeze(0).repeat(self.in_dim, 1)
                    self.register_buffer('W', W)
                ff_dim = self.in_dim * fourier_feats * 2
            
            mlp_in = ff_dim
        else:
            # MLP mode: just use dx, dy, sin(yaw), cos(yaw)
            mlp_in = 4 

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_hidden), 
            nn.GELU(),  # GELU usually works better than ReLU for transformers
            nn.Linear(mlp_hidden, dim),
            nn.LayerNorm(dim)
        )

        self._cached_param_dtype = None
        self._cached_param_device = None
    
    def _cache_param_info(self):
        if self._cached_param_dtype is None:
            p = next(self.parameters())
            self._cached_param_device = p.device
            self._cached_param_dtype = p.dtype

    def forward(self, pos):
        # pos: (..., 3) -> dx, dy, yaw_norm
        # Ensure yaw_norm is converted to radians for sin/cos logic if needed
        self._cache_param_info()
        assert pos.shape[-1] == 3

        x = pos.view(-1, 3)

        # Position normalization
        dx_dy = x[:, 0:2].clone()
        dx_dy /= 20.0

        # Yaw normalization
        yaw_rad = (torch.deg2rad(x[:, 2:3]) + torch.pi) % (2 * torch.pi) - torch.pi
        assert (yaw_rad <= torch.pi).all() and (yaw_rad >= -torch.pi).all()

        x_proc = torch.cat([dx_dy, torch.sin(yaw_rad), torch.cos(yaw_rad)], dim=-1).to(device=self._cached_param_device, dtype=self._cached_param_dtype)

        if self.mode == 'mlp':
            emb = self.mlp(x_proc)
        
        elif self.mode == 'fourier':
            if self.mapping_type == 'rff':
                # x: (N, 4), W: (4, F) -> (N, F)
                if self.learnable:
                    x_proj = self.rff_proj(x_proc)
                else:
                    x_proj = x_proc @ self.W
                # sin/cos mapping
                ff = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            
            else: # nerf
                # x_proc: (B, 4), W: (4, F)
                x_proj = x_proc.unsqueeze(-1) * self.W.unsqueeze(0) # (N, 3, F)
                x_proj = x_proj.flatten(1) # (N, 3*F)
                ff = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            
            emb = self.mlp(ff)
            
        return emb.view(*pos.shape[:-1], self.dim)
