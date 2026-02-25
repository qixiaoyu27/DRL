from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class WindAttentionExtractor(BaseFeaturesExtractor):
    """Self-state MLP + wind token attention."""

    def __init__(self, observation_space, features_dim: int = 256, use_attention: bool = True):
        super().__init__(observation_space, features_dim)
        self.use_attention = use_attention
        self.self_mlp = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())

        self.wind_proj = nn.Linear(2, 64)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.wind_out = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(64 + 128, features_dim), nn.ReLU())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        self_state = obs[:, :6]
        wind_raw = obs[:, 6:].reshape(obs.shape[0], -1, 2)

        s = self.self_mlp(self_state)
        w = self.wind_proj(wind_raw)

        if self.use_attention:
            w_att, _ = self.attn(w, w, w)
            w_feat = w_att.mean(dim=1)
        else:
            w_feat = w.mean(dim=1)

        w_feat = self.wind_out(w_feat)
        return self.fuse(torch.cat([s, w_feat], dim=1))
