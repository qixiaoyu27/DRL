from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from drl_uav import config
from drl_uav.env import FixedWingCoverageEnv
from drl_uav.evaluate import evaluate_once, save_trajectory_plot
from drl_uav.models import RecurrentAttentionResidualPolicy
from drl_uav.scenario_manager import generate_scenarios_json


class LiveRewardCallback(BaseCallback):
    def __init__(self, save_dir: Path):
        super().__init__()
        self.save_dir = save_dir
        self.rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards.append(info["episode"]["r"])
                self._save_plot()
        return True

    def _save_plot(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(self.rewards, color="tab:blue")
        if len(self.rewards) >= 10:
            smooth = np.convolve(self.rewards, np.ones(10) / 10.0, mode="valid")
            plt.plot(np.arange(9, len(self.rewards)), smooth, color="tab:red", label="10-ep moving avg")
            plt.legend()
        plt.title("Training Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / "reward_curve.png", dpi=160)
        plt.close()


class CheckpointEvalCallback(BaseCallback):
    def __init__(self, save_dir: Path, eval_env: FixedWingCoverageEnv, every_steps: int):
        super().__init__()
        self.save_dir = save_dir
        self.eval_env = eval_env
        self.every_steps = every_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.every_steps != 0:
            return True

        self.save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.save_dir / f"ppo_lstm_attn_res_step_{self.num_timesteps}.zip"
        self.model.save(ckpt_path)

        result = evaluate_once(self.model, self.eval_env)
        fig_path = self.save_dir / f"eval_traj_step_{self.num_timesteps}.png"
        save_trajectory_plot(self.eval_env, result, fig_path)
        print(f"[Checkpoint] {ckpt_path.name} coverage={result['coverage_ratio']:.3f}")
        return True


def make_env(seed: int | None = None):
    def _build():
        return Monitor(FixedWingCoverageEnv(seed=seed))

    return _build


def main():
    generate_scenarios_json(config.SCENARIO_JSON, config.SCENARIO_COUNT, seed=config.TRAIN_SEED)

    log_dir = config.LOG_DIR
    model_dir = log_dir / "models"

    vec_env = DummyVecEnv([make_env(config.TRAIN_SEED)])
    eval_env = FixedWingCoverageEnv(seed=config.TRAIN_SEED + 7)

    model = RecurrentPPO(
        policy=RecurrentAttentionResidualPolicy,
        env=vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=128,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(log_dir / "tb"),
        verbose=1,
        seed=config.TRAIN_SEED,
    )

    callbacks = CallbackList(
        [
            LiveRewardCallback(save_dir=log_dir),
            CheckpointEvalCallback(save_dir=model_dir, eval_env=eval_env, every_steps=config.CHECKPOINT_STEPS),
        ]
    )

    model.learn(total_timesteps=config.TRAIN_TIMESTEPS, callback=callbacks, progress_bar=True)
    final_model = model_dir / "ppo_lstm_attn_res_final.zip"
    model.save(final_model)

    result = evaluate_once(model, eval_env)
    save_trajectory_plot(eval_env, result, log_dir / "final_eval_trajectory.png")
    print(f"Training complete. Final model: {final_model}")


if __name__ == "__main__":
    main()
