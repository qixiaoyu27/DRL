"""Training script for fixed-wing UAV route planning with PPO/SAC."""
from __future__ import annotations

import os
import pathlib
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from fixedwing_rl.envs.fixedwing_jsbsim_env import FixedWingJSBSimEnv
from fixedwing_rl.utils import (
    MetricsRecorder,
    TrainingConfig,
    WindField3D,
    load_training_config,
)
from fixedwing_rl.utils.plotting import save_training_curves, save_wind_heatmap


CONFIG_ENV_VAR = "FIXEDWING_TRAIN_CONFIG"
DEFAULT_CONFIG_PATH = pathlib.Path(__file__).resolve().parent / "configs" / "default.yaml"


ALGOS = {
    "ppo": PPO,
    "sac": SAC,
}


def resolve_config_path(config_path: str | pathlib.Path | None = None) -> pathlib.Path:
    """Return the configuration path from an explicit value, env-var or default file."""

    if config_path is not None:
        path = pathlib.Path(config_path)
    else:
        env_override = os.getenv(CONFIG_ENV_VAR)
        path = pathlib.Path(env_override) if env_override else DEFAULT_CONFIG_PATH
    return path.expanduser().resolve()


def default_model_path(config: TrainingConfig) -> pathlib.Path:
    """Return the expected checkpoint path for the configured algorithm."""

    algo_key = config.algorithm.name.lower()
    filename = f"{algo_key}_model.zip"
    return pathlib.Path(config.output.model_dir).expanduser() / filename


def create_wind_field(config: TrainingConfig) -> Optional[WindField3D]:
    """Instantiate a :class:`WindField3D` if ERA5 data is configured."""

    wind_settings = config.wind_field
    if not wind_settings.era5_path:
        return None
    return WindField3D(
        wind_settings.era5_path,
        variable_u=wind_settings.u_component,
        variable_v=wind_settings.v_component,
        variable_w=wind_settings.w_component,
        altitude_variable=wind_settings.altitude,
    )


def make_env(config: TrainingConfig, wind_field: Optional[WindField3D], monitor: bool = True) -> gym.Env:
    env_config = config.environment.to_fixedwing_config(wind_field)
    env = FixedWingJSBSimEnv(env_config)
    if monitor:
        env = Monitor(env)
    return env


def evaluate(
    model,
    env: FixedWingJSBSimEnv,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs, _ = env.reset()
    lats, lons, winds = [], [], []
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        lat, lon, _ = info["position"]
        lats.append(lat)
        lons.append(lon)
        wind_vec = env._wind_vector()  # type: ignore[attr-defined]
        winds.append(float(np.linalg.norm(wind_vec)))
        if terminated or truncated:
            break
    trajectory = np.array(env.trajectory)
    return np.array(lats), np.array(lons), np.array(winds), trajectory


def main(config_path: str | pathlib.Path | None = None) -> None:
    config = load_training_config(resolve_config_path(config_path))

    algo_key = config.algorithm.name.lower()
    if algo_key not in ALGOS:
        raise ValueError(f"Unsupported algorithm '{config.algorithm.name}'. Expected one of {sorted(ALGOS)}.")
    algo_cls = ALGOS[algo_key]

    wind_field = create_wind_field(config)

    env = DummyVecEnv([lambda: make_env(config, wind_field)])
    callback = MetricsRecorder()

    model = algo_cls(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=config.output.tensorboard_log,
        device=config.algorithm.device,
        **config.algorithm.policy_kwargs,
    )

    model.learn(total_timesteps=config.algorithm.total_timesteps, callback=callback)

    model_dir = pathlib.Path(config.output.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = default_model_path(config)
    model.save(model_path)

    # Plot training curves
    plots_dir = pathlib.Path(config.output.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    timesteps = np.array(callback.timesteps, dtype=int)
    rewards = np.array(callback.rewards, dtype=float)
    losses = np.array(callback.losses, dtype=float) if callback.losses else np.zeros_like(timesteps, dtype=float)
    save_training_curves(timesteps, rewards, losses, plots_dir / "training_curves.png")

    # Evaluation rollout for visualization
    eval_env = make_env(config, wind_field, monitor=False)
    lats, lons, winds, trajectory = evaluate(model, eval_env, config.evaluation.steps)
    save_wind_heatmap(lats, lons, winds, trajectory, plots_dir / "wind_trajectory.png")

    eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
