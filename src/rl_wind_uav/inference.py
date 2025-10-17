"""Run inference with a trained policy and optionally stream to FlightGear."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":  # pragma: no cover - runtime safety for IDE execution
    package_root = Path(__file__).resolve().parents[1]
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)
    __package__ = "rl_wind_uav"

import numpy as np
from stable_baselines3 import PPO

from rl_wind_uav.env.fixed_wing_route_env import FixedWingRouteEnv
from rl_wind_uav.configuration import (
    DEFAULT_INFERENCE_CONFIG,
    ConfigurationError,
    build_route_config,
    gather_env_overrides,
    load_config,
    resolve_device,
    resolve_int,
    resolve_jsbsim_root,
    resolve_model_path,
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
    parser.add_argument("--device", type=str, default=None, help="Torch device string (e.g. cpu, cuda, cuda:0)")
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Use deterministic policy execution (default)",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Sample from the policy instead of using deterministic actions",
    )
    parser.set_defaults(deterministic=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        config_data = load_config(args.config, DEFAULT_INFERENCE_CONFIG)
        jsbsim_root = resolve_jsbsim_root(args.jsbsim_root, config_data)
        model_path = resolve_model_path(args.model, config_data)
        episodes = resolve_int(args.episodes, config_data, "episodes", 5)
        seed = resolve_int(args.seed, config_data, "seed", 0)
        env_overrides = gather_env_overrides(config_data)
        if args.flightgear:
            env_overrides["enable_flightgear"] = True
        if args.flightgear_path:
            env_overrides["flightgear_path"] = args.flightgear_path
        deterministic = bool(config_data.get("deterministic", True))
        if args.deterministic is not None:
            deterministic = args.deterministic
        route_config = build_route_config(jsbsim_root, env_overrides)
        device = resolve_device(args.device, config_data)
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc

    env = FixedWingRouteEnv(route_config)

    if route_config.enable_flightgear:
        env.render()

    model = PPO.load(model_path, device=device)

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        if route_config.enable_flightgear:
            env.render()
        done = False
        truncated = False
        episode_reward = 0.0
        step = 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=bool(deterministic))
            obs, reward, done, truncated, info = env.step(action)
            if route_config.enable_flightgear:
                env.render()
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
