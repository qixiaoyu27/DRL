from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs" / "ppo_residual_lstm_attn"
OUT_DIR = ROOT / "paper_outputs"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_csv = OUT_DIR / "evaluation_metrics.csv"
    if eval_csv.exists():
        df = pd.read_csv(eval_csv)

        plt.figure(figsize=(7, 4))
        sns.histplot(df, x="steps", hue="model", kde=True, element="step")
        plt.title("Episode Steps Distribution")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "fig_steps_hist.png", dpi=200)

        plt.figure(figsize=(7, 4))
        sns.lineplot(data=df.groupby(["model", "wind_case"], as_index=False)["coverage"].mean(), x="wind_case", y="coverage", hue="model")
        plt.xticks([], [])
        plt.title("Coverage Across Wind Cases")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "fig_coverage_by_case.png", dpi=200)


if __name__ == "__main__":
    main()
