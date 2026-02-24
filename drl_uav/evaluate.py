from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import RecurrentPPO

from drl_uav import config
from drl_uav.env import FixedWingCoverageEnv
from drl_uav.scenario_manager import generate_scenarios_json


DEFAULT_MODEL = config.LOG_DIR / "models/ppo_lstm_attn_res_final.zip"
DEFAULT_OUTPUT = config.LOG_DIR / "eval_trajectory.png"


def evaluate_once(model: RecurrentPPO, env: FixedWingCoverageEnv, deterministic: bool = True):
    obs, _ = env.reset()
    done = False
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)
    positions = [env.state[:2].copy()]
    rewards = []

    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_start = np.array([done], dtype=bool)
        positions.append(info["position"])
        rewards.append(reward)

    return {"positions": np.array(positions), "total_reward": float(np.sum(rewards)), "coverage_ratio": float(info["coverage_ratio"]) }


def save_trajectory_plot(env: FixedWingCoverageEnv, result: dict, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pos = result["positions"]

    plt.figure(figsize=(7, 7))
    plt.plot(pos[:, 0], pos[:, 1], "b-", lw=1.6, label="UAV Trajectory")
    plt.scatter(pos[0, 0], pos[0, 1], c="green", s=60, label="Start")
    plt.scatter(pos[-1, 0], pos[-1, 1], c="red", s=60, label="End")
    for i, vortex in enumerate(env.wind_field.vortices):
        circle = plt.Circle(vortex.center, vortex.core_radius, color="orange", fill=False, linestyle="--", alpha=0.7)
        plt.gca().add_patch(circle)
        plt.text(vortex.center[0], vortex.center[1], f"V{i+1}", color="orange")
    plt.xlim(0, env.map_size)
    plt.ylim(0, env.map_size)
    plt.title(f"Eval Trajectory | Coverage={result['coverage_ratio']:.3f} | Reward={result['total_reward']:.2f}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def main(model_path: Path = DEFAULT_MODEL, output_path: Path = DEFAULT_OUTPUT):
    generate_scenarios_json(config.SCENARIO_JSON, config.SCENARIO_COUNT, seed=config.TRAIN_SEED)
    env = FixedWingCoverageEnv(seed=config.TRAIN_SEED + 99)
    model = RecurrentPPO.load(str(model_path))
    result = evaluate_once(model, env)
    save_trajectory_plot(env, result, output_path)
    print(f"Eval done: reward={result['total_reward']:.2f}, coverage={result['coverage_ratio']:.3f}")
    print(f"Saved trajectory plot -> {output_path}")


if __name__ == "__main__":
    main()
