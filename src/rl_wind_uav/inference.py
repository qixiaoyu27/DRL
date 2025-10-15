"""Run inference with a trained policy and optionally stream to FlightGear."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .env.fixed_wing_route_env import FixedWingRouteEnv, RouteConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO route policy")
    parser.add_argument("--model", type=Path, required=True, help="Path to the trained model (zip)")
    parser.add_argument("--jsbsim-root", type=str, required=True, help="Path to JSBSim root")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--flightgear", action="store_true", help="Enable FlightGear visualization")
    parser.add_argument("--flightgear-path", type=str, default=None, help="Optional FlightGear executable path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RouteConfig(
        jsbsim_root=args.jsbsim_root,
        enable_flightgear=args.flightgear,
        flightgear_path=args.flightgear_path,
    )
    env = FixedWingRouteEnv(config)

    model = PPO.load(args.model, device="auto")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        truncated = False
        episode_reward = 0.0
        step = 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            wind = info.get("wind", np.zeros(2))
            print(
                f"Episode {ep+1} step {step:04d}: reward={reward:6.2f} distance_x={obs[-2]:8.1f}m distance_y={obs[-1]:8.1f}m wind=({wind[0]:5.2f},{wind[1]:5.2f})"
            )
            step += 1
        print(f"Episode {ep+1} finished. Return={episode_reward:.2f} done={done} truncated={truncated}")

    env.close()


if __name__ == "__main__":
    main()
