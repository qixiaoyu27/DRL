"""Training entry-point for the fixed-wing route environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .env.fixed_wing_route_env import FixedWingRouteEnv, RouteConfig
from .configuration import (
    DEFAULT_TRAIN_CONFIG,
    ConfigurationError,
    load_config,
    resolve_int,
    resolve_jsbsim_root,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent for wind-aware routing")
    parser.add_argument("--config", type=Path, default=None, help="Optional path to a training config JSON file")
    parser.add_argument("--jsbsim-root", type=str, default=None, help="Path to JSBSim root (containing aircraft/)")
    parser.add_argument("--logdir", type=Path, default=None, help="Output directory for logs and checkpoints")
    parser.add_argument("--total-steps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--eval-freq", type=int, default=None, help="Evaluation frequency in steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--num-envs", type=int, default=None, help="Number of vectorized environments")
    return parser.parse_args()


def make_env(jsbsim_root: str, seed: int) -> gym.Env:
    def _factory() -> gym.Env:
        config = RouteConfig(jsbsim_root=jsbsim_root)
        env = FixedWingRouteEnv(config)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _factory


def save_config(logdir: Path, config: Dict[str, Any]) -> None:
    logdir.mkdir(parents=True, exist_ok=True)
    with open(logdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main() -> None:
    args = parse_args()

    try:
        config_data = load_config(args.config, DEFAULT_TRAIN_CONFIG)
        jsbsim_root = resolve_jsbsim_root(args.jsbsim_root, config_data)
        logdir = resolve_path(args.logdir, config_data, "logdir", Path("runs/ppo"))
        total_steps = resolve_int(args.total_steps, config_data, "total_steps", 2_000_000)
        eval_freq = resolve_int(args.eval_freq, config_data, "eval_freq", 50_000)
        seed = resolve_int(args.seed, config_data, "seed", 42)
        num_envs = resolve_int(args.num_envs, config_data, "num_envs", 1)
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc

    save_config(
        logdir,
        {
            "algo": "PPO",
            "total_steps": total_steps,
            "eval_freq": eval_freq,
            "seed": seed,
            "num_envs": num_envs,
        },
    )

    env_fns = [make_env(str(jsbsim_root), seed + i) for i in range(num_envs)]
    vec_env = DummyVecEnv(env_fns)

    eval_env = FixedWingRouteEnv(RouteConfig(jsbsim_root=str(jsbsim_root)))
    eval_env = Monitor(eval_env)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        seed=seed,
        tensorboard_log=str(logdir / "tb"),
        device="auto",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(logdir / "checkpoints"),
        log_path=str(logdir / "eval"),
        eval_freq=max(eval_freq // num_envs, 1),
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=total_steps, callback=eval_callback)
    model.save(logdir / "ppo_wind_route")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
