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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent for wind-aware routing")
    parser.add_argument("--jsbsim-root", type=str, required=True, help="Path to JSBSim root (containing aircraft/)")
    parser.add_argument("--logdir", type=Path, default=Path("runs/ppo"), help="Output directory for logs and checkpoints")
    parser.add_argument("--total-steps", type=int, default=2_000_000, help="Total training timesteps")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="Evaluation frequency in steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of vectorized environments")
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
    save_config(
        args.logdir,
        {
            "algo": "PPO",
            "total_steps": args.total_steps,
            "eval_freq": args.eval_freq,
            "seed": args.seed,
            "num_envs": args.num_envs,
        },
    )

    env_fns = [make_env(args.jsbsim_root, args.seed + i) for i in range(args.num_envs)]
    vec_env = DummyVecEnv(env_fns)

    eval_env = FixedWingRouteEnv(RouteConfig(jsbsim_root=args.jsbsim_root))
    eval_env = Monitor(eval_env)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(args.logdir / "tb"),
        device="auto",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.logdir / "checkpoints"),
        log_path=str(args.logdir / "eval"),
        eval_freq=max(args.eval_freq // args.num_envs, 1),
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.total_steps, callback=eval_callback)
    model.save(args.logdir / "ppo_wind_route")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
