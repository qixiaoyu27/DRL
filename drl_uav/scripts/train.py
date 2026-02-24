from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import trange

from drl_uav.algorithms.residual_ppo import PPOTrainer, ResidualActorCritic, Transition
from drl_uav.core.environment import FixedWingScanEnv
from drl_uav.core.utils import load_config, set_seed


def main(config_path: str, out_dir: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = FixedWingScanEnv(cfg["env"], seed=cfg["seed"])
    model = ResidualActorCritic(**cfg["model"])
    trainer = PPOTrainer(model, cfg["ppo"], device=cfg["device"])

    history_len = 8
    logs = []
    update_idx = 0

    for stage in cfg["curriculum"]:
        env.set_difficulty(stage["wind_level"], stage["num_vortices"])
        for _ in trange(stage["updates"], desc=f"Stage wind={stage['wind_level']}"):
            update_idx += 1
            obs, grid, state = env.reset()
            hist = deque([obs] * history_len, maxlen=history_len)
            transitions: list[Transition] = []
            ep_reward = 0.0
            coverage = 0.0

            for _step in range(cfg["ppo"]["steps_per_update"]):
                obs_seq = np.stack(hist)
                base_action = env.baseline.action(state)
                action, logprob, value = trainer.act(obs_seq, grid, base_action)
                residual_action = action - base_action

                next_obs, next_grid, reward, done, info = env.step(action)
                transitions.append(
                    Transition(
                        obs_seq=obs_seq,
                        grid=grid,
                        action=residual_action,
                        logprob=logprob,
                        reward=reward,
                        done=float(done),
                        value=value,
                    )
                )
                ep_reward += reward
                coverage = info["coverage"]
                obs, grid, state = next_obs, next_grid, env.state.copy()
                hist.append(obs)
                if done:
                    obs, grid, state = env.reset()
                    hist = deque([obs] * history_len, maxlen=history_len)

            with torch.no_grad():
                last_obs_seq = np.stack(hist)
                last_base = env.baseline.action(state)
                _, _, last_value = trainer.act(last_obs_seq, grid, last_base)

            losses = trainer.update(transitions, last_value)
            logs.append(
                {
                    "update": update_idx,
                    "wind_level": stage["wind_level"],
                    "num_vortices": stage["num_vortices"],
                    "episode_reward": ep_reward,
                    "coverage": coverage,
                    **losses,
                }
            )

    torch.save(model.state_dict(), out / "policy.pt")
    pd.DataFrame(logs).to_csv(out / "train_log.csv", index=False)
    print(f"Training finished. Model/log saved to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="drl_uav/configs/default.yaml")
    parser.add_argument("--out", default="drl_uav/outputs/train")
    args = parser.parse_args()
    main(args.config, args.out)
