import math
import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, mode, max_len, dim, mlp_hidden=64):
        super().__init__()
        self.mode = mode
        self.dim = dim
        self.max_len = max_len
        
        if mode == "learned":
            self.emb = nn.Embedding(max_len, dim)
            
        elif mode == "sinus":
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            # Fix parity issue: truncate div_term to match slice length
            div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            # Handle odd dim
            if dim % 2 == 1:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            else:
                pe[:, 1::2] = torch.cos(position * div_term)
                
            self.register_buffer('pe', pe)
            
        elif mode == "continuous":
            # Upgrade: Use Fourier MLP for continuous time too
            self.fourier_dim = 16
            freqs = 2.0 ** torch.arange(self.fourier_dim).float()
            self.register_buffer('freqs', freqs)
            
            mlp_in = self.fourier_dim * 2
            self.mlp = nn.Sequential(
                nn.Linear(mlp_in, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, dim)
            )

    def forward(self, idx_or_ts):
        if self.mode == "learned":
            # Safety Check
            idx = idx_or_ts.long().clamp(0, self.max_len - 1)
            return self.emb(idx)
            
        elif self.mode == "sinus":
            # Handle out-of-bound by extrapolating or clamping? 
            # Sinusoidal can technically compute on the fly, but looking up buffer is faster.
            # We assume clamping for safety here.
            idx = idx_or_ts.long().clamp(0, self.max_len - 1)
            return self.pe.to(device=idx.device)[idx]
            
        else: # continuous
            x = idx_or_ts.float().unsqueeze(-1) # (N, 1)
            # Fourier Encoding for scalar time
            freqs = self.freqs.to(device=x.device)
            x_proj = x * freqs.unsqueeze(0) # (N, F)
            x_emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            return self.mlp(x_emb)