from __future__ import annotations

from pathlib import Path
import os
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from drl_uav.envs.coverage_env import FixedWingCoverageEnv, EnvConfig
from drl_uav.models.attention_extractor import WindAttentionExtractor
from drl_uav.training.callbacks import RewardPlotCallback


# ===== 写死参数，便于IDE直接运行 =====
RUN_NAME = "ppo_residual_lstm_attn"
ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "runs" / RUN_NAME
CKPT_DIR = LOG_DIR / "checkpoints"
MONITOR_DIR = LOG_DIR / "monitor"
TOTAL_STEPS = 2_000_000
N_ENVS = min(8, os.cpu_count() or 8)
SEED = 42
RESUME = True

# 消融开关
USE_LSTM = True
USE_ATTENTION = True
USE_RESIDUAL_POLICY = True


def make_env(rank: int):
    def _init():
        cfg = EnvConfig(use_residual_policy=USE_RESIDUAL_POLICY, use_attention_features=USE_ATTENTION)
        env = FixedWingCoverageEnv(cfg=cfg, seed=SEED + rank)
        mon_file = MONITOR_DIR / f"env_{rank}.monitor.csv"
        mon_file.parent.mkdir(parents=True, exist_ok=True)
        return Monitor(env, str(mon_file))

    return _init


def latest_checkpoint() -> Path | None:
    if not CKPT_DIR.exists():
        return None
    cks = sorted(CKPT_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    return cks[-1] if cks else None


def main():
    vec_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    policy_kwargs = dict(
        features_extractor_class=WindAttentionExtractor,
        features_extractor_kwargs=dict(features_dim=256, use_attention=USE_ATTENTION),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    ckpt = latest_checkpoint() if RESUME else None
    if USE_LSTM:
        if ckpt is not None:
            model = RecurrentPPO.load(str(ckpt), env=vec_env, device="cuda")
        else:
            model = RecurrentPPO(
                "MlpLstmPolicy",
                vec_env,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=512,
                gamma=0.995,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.005,
                vf_coef=0.5,
                tensorboard_log=str(LOG_DIR / "tb"),
                policy_kwargs=policy_kwargs,
                seed=SEED,
                device="cuda",
                verbose=1,
            )
    else:
        if ckpt is not None:
            model = PPO.load(str(ckpt), env=vec_env, device="cuda")
        else:
            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=512,
                gamma=0.995,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.005,
                vf_coef=0.5,
                tensorboard_log=str(LOG_DIR / "tb"),
                policy_kwargs=policy_kwargs,
                seed=SEED,
                device="cuda",
                verbose=1,
            )

    cbs = CallbackList(
        [
            CheckpointCallback(save_freq=25_000 // N_ENVS, save_path=str(CKPT_DIR), name_prefix="model"),
            RewardPlotCallback(monitor_csv=MONITOR_DIR / "env_0.monitor.csv", out_png=LOG_DIR / "reward_curve.png"),
        ]
    )

    model.learn(total_timesteps=TOTAL_STEPS, callback=cbs, progress_bar=True, reset_num_timesteps=not bool(ckpt))
    model.save(str(LOG_DIR / "final_model.zip"))
    vec_env.close()


if __name__ == "__main__":
    main()
