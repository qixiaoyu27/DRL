"""Training entry-point for the fixed-wing route environment."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict

if __package__ is None or __package__ == "":  # pragma: no cover - runtime safety for IDE execution
    package_root = Path(__file__).resolve().parents[1]
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)
    __package__ = "rl_wind_uav"

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_wind_uav.callbacks import ConsoleLogCallback
from rl_wind_uav.env.fixed_wing_route_env import FixedWingRouteEnv, RouteConfig
from rl_wind_uav.configuration import (
    DEFAULT_TRAIN_CONFIG,
    ConfigurationError,
    build_route_config,
    extract_ppo_hyperparameters,
    gather_env_overrides,
    load_config,
    resolve_device,
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
    parser.add_argument("--device", type=str, default=None, help="Torch device string (e.g. cpu, cuda, cuda:0)")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help="Number of environment steps between checkpoint saves (0 disables)",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default=None,
        help="Filename prefix for periodic checkpoints and the final model",
    )
    parser.add_argument(
        "--console-log-interval",
        type=int,
        default=None,
        help="Number of environment steps between console summaries (0 disables)",
    )
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress-bar",
        dest="progress_bar",
        action="store_true",
        help="Force enable the tqdm training progress bar",
    )
    progress_group.add_argument(
        "--no-progress-bar",
        dest="progress_bar",
        action="store_false",
        help="Disable the tqdm training progress bar",
    )
    parser.set_defaults(progress_bar=None)
    return parser.parse_args()


def make_env(route_config: RouteConfig, seed: int) -> gym.Env:
    def _factory() -> gym.Env:
        config_copy = replace(route_config, wind_config=replace(route_config.wind_config))
        env = FixedWingRouteEnv(config_copy)
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
        device = resolve_device(args.device, config_data)
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc

    checkpoint_config = config_data.get("checkpoint", {}) if isinstance(config_data.get("checkpoint"), dict) else {}
    checkpoint_freq = (
        args.checkpoint_freq
        if args.checkpoint_freq is not None
        else checkpoint_config.get("freq", config_data.get("checkpoint_freq", 0))
    )
    checkpoint_prefix = (
        args.checkpoint_prefix
        if args.checkpoint_prefix
        else checkpoint_config.get("prefix", config_data.get("checkpoint_prefix", "ppo_wind_route"))
    )
    checkpoint_freq = int(checkpoint_freq or 0)

    progress_bar = config_data.get("progress_bar", True)
    if args.progress_bar is not None:
        progress_bar = args.progress_bar

    console_config = config_data.get("console_log", {}) if isinstance(config_data.get("console_log"), dict) else {}
    console_log_interval = (
        args.console_log_interval
        if args.console_log_interval is not None
        else console_config.get("interval_steps", console_config.get("log_interval", 0))
    )
    if "console_log_interval" in config_data and not console_log_interval:
        console_log_interval = config_data["console_log_interval"]
    try:
        console_log_interval = int(console_log_interval)
    except (TypeError, ValueError):
        console_log_interval = 0
    console_log_interval = max(console_log_interval, 0)

    env_overrides = gather_env_overrides(config_data)
    route_config = build_route_config(jsbsim_root, env_overrides)
    eval_route_config = replace(route_config, wind_config=replace(route_config.wind_config))

    env_fns = [make_env(route_config, seed + i) for i in range(num_envs)]
    vec_env = DummyVecEnv(env_fns)

    eval_env = FixedWingRouteEnv(eval_route_config)
    eval_env = Monitor(eval_env)

    policy, verbose, tensorboard_override, ppo_kwargs = extract_ppo_hyperparameters(config_data)
    tensorboard_log = (
        str(Path(str(tensorboard_override)).expanduser())
        if tensorboard_override not in (None, "")
        else str(logdir / "tb")
    )

    model = PPO(
        policy=policy,
        env=vec_env,
        verbose=verbose,
        seed=seed,
        tensorboard_log=tensorboard_log,
        device=device,
        **ppo_kwargs,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(logdir / "checkpoints"),
        log_path=str(logdir / "eval"),
        eval_freq=max(eval_freq // num_envs, 1),
        deterministic=True,
        render=False,
    )

    callbacks = [eval_callback]

    if checkpoint_freq > 0:
        checkpoint_dir = logdir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=max(checkpoint_freq // num_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix=checkpoint_prefix,
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callbacks.append(checkpoint_callback)

    if progress_bar:
        callbacks.append(ProgressBarCallback())

    if console_log_interval > 0:
        callbacks.append(ConsoleLogCallback(log_interval=max(console_log_interval // num_envs, 1)))

    callback = CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0]

    model.learn(total_timesteps=total_steps, callback=callback)
    final_model_path = logdir / f"{checkpoint_prefix}_final"
    model.save(final_model_path)
    print(f"Saved final policy to {final_model_path.with_suffix('.zip')}")

    vec_env.close()
    eval_env.close()

    config_snapshot: Dict[str, Any] = {
        "algo": "PPO",
        "total_steps": total_steps,
        "eval_freq": eval_freq,
        "seed": seed,
        "num_envs": num_envs,
        "checkpoint_freq": checkpoint_freq,
        "checkpoint_prefix": checkpoint_prefix,
        "progress_bar": progress_bar,
        "console_log_interval": console_log_interval,
        "device": device,
        "logdir": str(logdir),
        "env": asdict(route_config),
        "ppo": {
            "policy": policy,
            "verbose": verbose,
            "tensorboard_log": tensorboard_log,
            **ppo_kwargs,
        },
    }

    save_config(logdir, config_snapshot)


if __name__ == "__main__":
    main()
