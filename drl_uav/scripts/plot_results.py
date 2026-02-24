from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main(train_log: str, eval_log: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_log)
    eval_df = pd.read_csv(eval_log)

    plt.figure(figsize=(7, 4))
    plt.plot(train_df["update"], train_df["coverage"], label="Coverage")
    plt.plot(train_df["update"], train_df["episode_reward"], label="Reward", alpha=0.7)
    plt.xlabel("Update")
    plt.title("Training curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "training_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.bar(eval_df["episode"], eval_df["coverage"])
    plt.ylim(0, 1)
    plt.xlabel("Episode")
    plt.ylabel("Coverage")
    plt.title("Evaluation coverage")
    plt.tight_layout()
    plt.savefig(out / "eval_coverage.png", dpi=180)
    plt.close()

    summary = {
        "train_final_coverage": float(train_df["coverage"].iloc[-1]),
        "train_best_coverage": float(train_df["coverage"].max()),
        "eval_mean_coverage": float(eval_df["coverage"].mean()),
        "eval_mean_reward": float(eval_df["reward"].mean()),
    }
    pd.Series(summary).to_csv(out / "summary.csv")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", default="drl_uav/outputs/train/train_log.csv")
    parser.add_argument("--eval-log", default="drl_uav/outputs/eval/eval_metrics.csv")
    parser.add_argument("--out", default="drl_uav/outputs/plots")
    args = parser.parse_args()
    main(args.train_log, args.eval_log, args.out)
