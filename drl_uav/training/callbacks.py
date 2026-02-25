from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


class RewardPlotCallback(BaseCallback):
    def __init__(self, monitor_csv: Path, out_png: Path, every_n_calls: int = 5000):
        super().__init__()
        self.monitor_csv = monitor_csv
        self.out_png = out_png
        self.every_n_calls = every_n_calls

    def _on_step(self) -> bool:
        if self.n_calls % self.every_n_calls != 0:
            return True
        if not self.monitor_csv.exists():
            return True
        try:
            df = pd.read_csv(self.monitor_csv, comment="#")
        except Exception:
            return True
        if len(df) < 3:
            return True

        rolling = df["r"].rolling(window=20, min_periods=1).mean()
        plt.figure(figsize=(8, 4))
        plt.plot(df.index, df["r"], alpha=0.3, label="Episode Reward")
        plt.plot(df.index, rolling, linewidth=2, label="Rolling(20)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Reward Curve")
        plt.grid(alpha=0.3)
        plt.legend()
        self.out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(self.out_png, dpi=150)
        plt.close()
        return True
