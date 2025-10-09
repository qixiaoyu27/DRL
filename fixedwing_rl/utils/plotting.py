"""Plotting helpers for training curves and wind-field heatmaps."""
from __future__ import annotations

import pathlib
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def save_training_curves(
    timesteps: Sequence[int],
    rewards: Sequence[float],
    losses: Sequence[float],
    output_path: pathlib.Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(timesteps, rewards, label="Episode reward", color="tab:blue")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(timesteps, losses, label="Loss", color="tab:orange", alpha=0.6)
    ax2.set_ylabel("Loss", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.savefig(output_path)
    plt.close(fig)


def save_wind_heatmap(
    lats: Sequence[float],
    lons: Sequence[float],
    winds: Sequence[float],
    trajectory: Iterable[Sequence[float]],
    output_path: pathlib.Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.tricontourf(lons, lats, winds, levels=20, cmap="coolwarm", alpha=0.8)
    fig.colorbar(scatter, ax=ax, label="Wind speed (m/s)")

    traj = np.array(list(trajectory))
    if traj.size:
        ax.plot(traj[:, 1], traj[:, 0], color="black", linewidth=2.0, label="Trajectory")
        ax.scatter(traj[0, 1], traj[0, 0], color="green", label="Start")
        ax.scatter(traj[-1, 1], traj[-1, 0], color="red", label="End")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Wind field heatmap with flight trajectory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
