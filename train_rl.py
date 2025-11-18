"""训练脚本：支持 PPO/SAC/TD3 并输出奖励-回合曲线."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs import PointToPointAvoidEnv


class EpisodeRewardCallback(BaseCallback):
    """收集每个 Episode 的奖励并绘图."""

    def __init__(self, save_path: Path):
        super().__init__()
        self.save_path = save_path
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def _on_step(self) -> bool:
        infos: Tuple[dict, ...] = self.locals.get("infos", tuple())
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True

    def _on_training_end(self) -> None:
        if not self.episode_rewards:
            return
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(self.episode_rewards, label="Episode Reward")
        window = max(1, len(self.episode_rewards) // 20)
        if window > 1:
            kernel = np.ones(window) / window
            smooth = np.convolve(self.episode_rewards, kernel, mode="valid")
            plt.plot(np.arange(window - 1, window - 1 + len(smooth)), smooth, label="Moving Avg")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("奖励-回合曲线")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="点对点飞行环境训练入口")
    parser.add_argument("--algo", choices=["ppo", "sac", "td3"], default="ppo", help="强化学习算法")
    parser.add_argument("--timesteps", type=int, default=200_000, help="训练总步数")
    parser.add_argument("--device", default="cuda", help="PyTorch 设备，如 cuda 或 cpu")
    parser.add_argument("--logdir", default="outputs", help="日志与图表的输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def make_env(seed: int):
    def _init():
        # Monitor 负责记录 episode 级别的统计指标
        env = PointToPointAvoidEnv(seed=seed)
        return Monitor(env)

    return _init


def build_agent(algo: str, env, device: str):
    # 使用较宽的 MLP，提高策略表达能力
    policy_kwargs = dict(net_arch=[256, 256])
    if algo == "ppo":
        return PPO("MlpPolicy", env, verbose=1, device=device, policy_kwargs=policy_kwargs)
    if algo == "sac":
        return SAC("MlpPolicy", env, verbose=1, device=device, policy_kwargs=policy_kwargs)
    if algo == "td3":
        return TD3("MlpPolicy", env, verbose=1, device=device, policy_kwargs=policy_kwargs)
    raise ValueError(f"未知算法: {algo}")


def main():
    args = parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # 构建 4 个并行环境，加速采样
    env = DummyVecEnv([make_env(args.seed + i) for i in range(4)])
    callback = EpisodeRewardCallback(Path(args.logdir) / "reward_curve.png")
    agent = build_agent(args.algo, env, args.device)

    agent.learn(total_timesteps=args.timesteps, callback=callback)
    agent.save(Path(args.logdir) / f"{args.algo}_policy")

    env.close()


if __name__ == "__main__":
    main()
