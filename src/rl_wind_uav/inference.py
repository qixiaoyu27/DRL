"""Run inference with a trained policy and optionally stream to FlightGear."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .env.fixed_wing_route_env import FixedWingRouteEnv, RouteConfig
from .configuration import (
    DEFAULT_INFERENCE_CONFIG,
    ConfigurationError,
    load_config,
    resolve_bool,
    resolve_int,
    resolve_jsbsim_root,
    resolve_model_path,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO route policy")
    parser.add_argument("--config", type=Path, default=None, help="Optional inference config JSON path")
    parser.add_argument("--model", type=Path, default=None, help="Path to the trained model (zip)")
    parser.add_argument("--jsbsim-root", type=str, default=None, help="Path to JSBSim root")
    parser.add_argument("--episodes", type=int, default=None, help="Number of evaluation episodes")
    parser.add_argument("--flightgear", action="store_true", help="Enable FlightGear visualization")
    parser.add_argument("--flightgear-path", type=str, default=None, help="Optional FlightGear executable path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        config_data = load_config(args.config, DEFAULT_INFERENCE_CONFIG)
        jsbsim_root = resolve_jsbsim_root(args.jsbsim_root, config_data)
        model_path = resolve_model_path(args.model, config_data)
        episodes = resolve_int(args.episodes, config_data, "episodes", 5)
        seed = resolve_int(args.seed, config_data, "seed", 0)
        flightgear_enabled = resolve_bool(args.flightgear, config_data, "flightgear")
        flightgear_path = resolve_path(
            Path(args.flightgear_path) if args.flightgear_path else None,
            config_data,
            "flightgear_path",
            None,
        )
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc

    config = RouteConfig(
        jsbsim_root=str(jsbsim_root),
        enable_flightgear=flightgear_enabled,
        flightgear_path=str(flightgear_path) if flightgear_path else None,
    )
    env = FixedWingRouteEnv(config)

    model = PPO.load(model_path, device="auto")

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
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
