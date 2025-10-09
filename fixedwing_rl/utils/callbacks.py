"""Custom callbacks used during RL training."""
from __future__ import annotations

from typing import List

from stable_baselines3.common.callbacks import BaseCallback


class MetricsRecorder(BaseCallback):
    """Collect episode rewards and losses for post-training plots."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.timesteps: List[int] = []
        self.rewards: List[float] = []
        self.losses: List[float] = []

    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            info = self.locals["infos"][0]
            if "episode" in info:
                self.timesteps.append(self.num_timesteps)
                self.rewards.append(info["episode"]["r"])
                if "loss" in info:
                    self.losses.append(info["loss"])
        if "loss" in self.locals:
            self.losses.append(float(self.locals["loss"]))
            self.timesteps.append(self.num_timesteps)
        return True
