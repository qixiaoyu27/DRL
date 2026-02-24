from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AttentionResidualExtractor(BaseFeaturesExtractor):
    """Tokenize scalar observation, then use attention + residual MLP blocks."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128, token_dim: int = 32, n_heads: int = 4):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        self.obs_dim = obs_dim
        self.token_dim = token_dim

        self.scalar_to_token = nn.Linear(1, token_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, obs_dim, token_dim))

        self.attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.norm2 = nn.LayerNorm(token_dim)
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim * token_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.unsqueeze(-1)
        tokens = self.scalar_to_token(x) + self.pos_embedding

        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        x = self.norm1(tokens + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return self.output(x)


class RecurrentAttentionResidualPolicy(RecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=AttentionResidualExtractor,
            features_extractor_kwargs={"features_dim": 128, "token_dim": 32, "n_heads": 4},
            lstm_hidden_size=128,
            n_lstm_layers=1,
            shared_lstm=True,
            enable_critic_lstm=False,
            net_arch=[128, 64],
        )
