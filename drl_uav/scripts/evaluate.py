from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from drl_uav.algorithms.residual_ppo import ResidualActorCritic
from drl_uav.core.environment import FixedWingScanEnv
from drl_uav.core.utils import load_config, set_seed


def run_episode(env: FixedWingScanEnv, model: ResidualActorCritic, history_len: int = 8):
    obs, grid, state = env.reset()
    hist = deque([obs] * history_len, maxlen=history_len)
    traj = []
    total_reward = 0.0
    done = False

    while not done:
        obs_seq = torch.tensor(np.stack(hist), dtype=torch.float32).unsqueeze(0)
        grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, _, _ = model(obs_seq, grid_t)
        action = torch.clamp(torch.tensor(env.baseline.action(state)) + mean.squeeze(0), -1.0, 1.0).numpy()

        obs, grid, reward, done, info = env.step(action)
        state = env.state.copy()
        hist.append(obs)
        total_reward += reward
        traj.append((state["x"], state["y"], info["coverage"], reward))

    return total_reward, info["coverage"], traj, env.grid.copy()


def main(config_path: str, model_path: str, out_dir: str, episodes: int):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = FixedWingScanEnv(cfg["env"], seed=cfg["seed"])
    env.set_difficulty(3, 3)
    model = ResidualActorCritic(**cfg["model"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    rows = []
    for ep in range(episodes):
        reward, cov, traj, grid = run_episode(env, model)
        rows.append({"episode": ep, "reward": reward, "coverage": cov, "steps": len(traj)})

        xs, ys = [p[0] for p in traj], [p[1] for p in traj]
        plt.figure(figsize=(6, 6))
        plt.imshow(grid, origin="lower", cmap="Blues", alpha=0.5)
        plt.plot(xs, ys, color="crimson", linewidth=1.2)
        plt.title(f"Episode {ep} trajectory | coverage={cov:.3f}")
        plt.xlabel("X grid")
        plt.ylabel("Y grid")
        plt.tight_layout()
        plt.savefig(out / f"trajectory_ep{ep}.png", dpi=180)
        plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(out / "eval_metrics.csv", index=False)
    print(df.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="drl_uav/configs/default.yaml")
    parser.add_argument("--model", default="drl_uav/outputs/train/policy.pt")
    parser.add_argument("--out", default="drl_uav/outputs/eval")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    main(args.config, args.model, args.out, args.episodes)
