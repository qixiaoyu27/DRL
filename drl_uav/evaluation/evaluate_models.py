from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from drl_uav.envs.coverage_env import FixedWingCoverageEnv, EnvConfig
from drl_uav.envs.wind_field import WindField

ROOT = Path(__file__).resolve().parents[2]
MODEL_CANDIDATES = [
    ROOT / "runs" / "ppo_residual_lstm_attn" / "final_model.zip",
]
ENV_DIR = ROOT / "wind_cases"
OUT_DIR = ROOT / "paper_outputs"
EPISODES_PER_ENV = 1


def load_policy(model_path: Path):
    try:
        return RecurrentPPO.load(str(model_path), device="cuda"), "recurrent"
    except Exception:
        return PPO.load(str(model_path), device="cuda"), "feedforward"


def evaluate_one(model, model_type: str, wind_jsons: list[Path]) -> pd.DataFrame:
    records = []
    for wj in wind_jsons:
        env = FixedWingCoverageEnv(cfg=EnvConfig(), wind_json=str(wj))
        for _ in range(EPISODES_PER_ENV):
            obs, _ = env.reset(options={"wind_json": str(wj), "randomize_wind": False})
            done, trunc = False, False
            ep_reward = 0.0
            state, starts = None, np.ones((1,), dtype=bool)
            while not (done or trunc):
                if model_type == "recurrent":
                    action, state = model.predict(obs, state=state, episode_start=starts, deterministic=True)
                    starts[:] = False
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = env.step(action)
                ep_reward += reward
            records.append(
                {
                    "wind_case": wj.name,
                    "coverage": info["coverage"],
                    "episode_reward": ep_reward,
                    "steps": env.step_count,
                    "mean_ground_speed": info["ground_speed"],
                }
            )
    return pd.DataFrame(records)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wind_jsons = sorted(ENV_DIR.glob("*.json"))
    if not wind_jsons:
        raise FileNotFoundError("wind_cases目录为空，请先运行随机环境生成脚本")

    all_rows = []
    for model_path in MODEL_CANDIDATES:
        model, t = load_policy(model_path)
        df = evaluate_one(model, t, wind_jsons)
        df["model"] = model_path.parent.name
        all_rows.append(df)

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(OUT_DIR / "evaluation_metrics.csv", index=False)

    summary = all_df.groupby("model")[["coverage", "episode_reward", "steps"]].agg(["mean", "std"]) 
    summary.to_csv(OUT_DIR / "evaluation_summary_table.csv")

    plt.figure(figsize=(6, 4))
    all_df.boxplot(column="coverage", by="model")
    plt.title("Coverage Distribution")
    plt.suptitle("")
    plt.ylabel("Coverage ratio")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_coverage_boxplot.png", dpi=200)

    (OUT_DIR / "meta.json").write_text(json.dumps({"n_env": len(wind_jsons)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
