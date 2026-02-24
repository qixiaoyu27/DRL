from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ResidualActorCritic(nn.Module):
    def __init__(self, obs_dim: int, grid_dim: int, action_dim: int, hidden_size: int = 128, attention_heads: int = 4):
        super().__init__()
        self.obs_embed = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.ReLU())
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.grid_proj = nn.Linear(1, hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, attention_heads, batch_first=True)
        self.grid_reduce = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())

        self.fuse = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.ReLU())
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_size, 1)

        self.grid_dim = grid_dim

    def forward(self, obs_seq: torch.Tensor, grid_flat: torch.Tensor):
        x = self.obs_embed(obs_seq)
        lstm_out, _ = self.lstm(x)
        temporal = lstm_out[:, -1, :]

        grid_tokens = grid_flat.unsqueeze(-1)
        tokens = self.grid_proj(grid_tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        spatial = self.grid_reduce(attn_out.mean(dim=1))

        fused = self.fuse(torch.cat([temporal, spatial], dim=-1))
        mean = torch.tanh(self.actor_mean(fused))
        std = torch.exp(self.actor_logstd).expand_as(mean)
        value = self.critic(fused).squeeze(-1)
        return mean, std, value


@dataclass
class Transition:
    obs_seq: np.ndarray
    grid: np.ndarray
    action: np.ndarray
    logprob: float
    reward: float
    done: float
    value: float


class PPOTrainer:
    def __init__(self, model: ResidualActorCritic, ppo_cfg: dict, device: str = "cpu"):
        self.model = model.to(device)
        self.cfg = ppo_cfg
        self.device = device
        self.optim = optim.Adam(self.model.parameters(), lr=float(ppo_cfg["lr"]))

    @torch.no_grad()
    def act(self, obs_seq: np.ndarray, grid: np.ndarray, baseline_action: np.ndarray):
        obs_t = torch.tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        grid_t = torch.tensor(grid, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, std, value = self.model(obs_t, grid_t)
        dist = Normal(mean, std)
        residual = dist.sample()
        action = torch.clamp(torch.tensor(baseline_action, device=self.device) + residual.squeeze(0), -1.0, 1.0)
        logprob = dist.log_prob(residual).sum(dim=-1).item()
        return action.cpu().numpy(), float(logprob), float(value.item())

    def compute_gae(self, transitions: list[Transition], last_value: float):
        gamma, lam = self.cfg["gamma"], self.cfg["gae_lambda"]
        adv = 0.0
        returns, advantages = [], []
        values = [t.value for t in transitions] + [last_value]
        for i in reversed(range(len(transitions))):
            delta = transitions[i].reward + gamma * values[i + 1] * (1 - transitions[i].done) - values[i]
            adv = delta + gamma * lam * (1 - transitions[i].done) * adv
            advantages.append(adv)
            returns.append(adv + values[i])
        advantages.reverse()
        returns.reverse()
        return np.array(returns, dtype=np.float32), np.array(advantages, dtype=np.float32)

    def update(self, transitions: list[Transition], last_value: float) -> dict:
        returns, adv = self.compute_gae(transitions, last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.tensor(np.stack([t.obs_seq for t in transitions]), dtype=torch.float32, device=self.device)
        grids = torch.tensor(np.stack([t.grid for t in transitions]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.stack([t.action for t in transitions]), dtype=torch.float32, device=self.device)
        old_logprobs = torch.tensor(np.array([t.logprob for t in transitions]), dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)

        n = obs.shape[0]
        batch_size = self.cfg["batch_size"]
        losses = {"policy": 0.0, "value": 0.0, "entropy": 0.0}

        for _ in range(self.cfg["epochs"]):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                b = idx[start : start + batch_size]
                mean, std, values = self.model(obs[b], grids[b])
                residual = actions[b]
                dist = Normal(mean, std)
                logprobs = dist.log_prob(residual).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(logprobs - old_logprobs[b])
                s1 = ratio * adv_t[b]
                s2 = torch.clamp(ratio, 1 - self.cfg["clip_eps"], 1 + self.cfg["clip_eps"]) * adv_t[b]
                policy_loss = -torch.min(s1, s2).mean()

                value_loss = ((returns_t[b] - values) ** 2).mean()
                loss = policy_loss + self.cfg["vf_coef"] * value_loss - self.cfg["ent_coef"] * entropy

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["max_grad_norm"])
                self.optim.step()

                losses["policy"] += float(policy_loss.item())
                losses["value"] += float(value_loss.item())
                losses["entropy"] += float(entropy.item())
        return losses
