import os
import numpy as np
from datasets import load_dataset
from video import compute_or_load_embeddings
from data_process import *
import torch
import torch.nn as nn

# -----------------------
# Load all datasets
# -----------------------
print("Loading datasets...")




import torch
import torch.nn as nn

# ----- Cross Network (Deep & Cross) -----
class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(dim))
    def forward(self, x0, x):
        # y = x0 * (x @ w) + b + x
        dot = (x * self.w).sum(dim=1, keepdim=True)  # (B,1)
        return x0 * dot + self.b + x

class CrossNet(nn.Module):
    def __init__(self, dim, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([CrossLayer(dim) for _ in range(n_layers)])
    def forward(self, x):
        x0, xl = x, x
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl

# ----- Head 1: Metadata encoder (simple MLP over cont + cat embeddings) -----
class MetaEncoder(nn.Module):
    def __init__(self, n_cont: int, hidden: int = 128):
        super().__init__()
        self.n_cont = n_cont
        self.bn = nn.BatchNorm1d(n_cont) if n_cont > 0 else None
        self.mlp = nn.Sequential(
            nn.Linear(n_cont, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
    def forward(self, x_cont=None):
        if self.n_cont > 0 and x_cont is not None:
            x = self.bn(x_cont) if self.bn is not None else x_cont
        else:
            raise ValueError("Expected continuous inputs")
        return self.mlp(x)

# ----- Head 2: Embedding encoder for concatenated video+text embedding -----
class EmbedEncoder(nn.Module):
    def __init__(self, in_dim=1024, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU()
        )
    def forward(self, x):
        return self.net(x)

# ----- Full two-head Deep & Cross regressor -----
class TwoHeadDCN(nn.Module):
    def __init__(self, meta_cont_dim,emb_in_dim=1024,
                 meta_hidden=128, emb_hidden=256, cross_layers=3):
        super().__init__()
        self.meta  = MetaEncoder(meta_cont_dim, hidden=meta_hidden)
        self.embs  = EmbedEncoder(emb_in_dim, hidden=emb_hidden)
        self.cross = CrossNet(meta_hidden + emb_hidden, n_layers=cross_layers)
        self.head  = nn.Sequential(
            nn.Linear(meta_hidden + emb_hidden, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x_cont,pair_emb):
        h_meta  = self.meta(x_cont)     # (B, Hm)
        h_embed = self.embs(pair_emb)           # (B, He)
        h = torch.cat([h_meta, h_embed], dim=1) # (B, Hm+He)
        h = self.cross(h)
        return self.head(h).squeeze(1)          # regression output


