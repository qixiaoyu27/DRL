"""Custom training callbacks for monitoring progress inside IDEs."""
from __future__ import annotations

import sys
import time
from typing import Iterable, List

from stable_baselines3.common.callbacks import BaseCallback


class ConsoleLogCallback(BaseCallback):
    """Emit periodic summaries of recent episodes to standard output.

    The default Stable-Baselines3 logging is geared toward TensorBoard files and
    stdout logs that are easier to parse from a terminal.  When running inside
    PyCharm the progress can be harder to follow, so this callback prints a short
    human-friendly summary every ``log_interval`` environment steps.  Any episodes
    that finished since the previous log are aggregated in the summary.
    """

    def __init__(self, log_interval: int = 10_000, *, stream=None) -> None:
        super().__init__(verbose=0)
        if log_interval < 1:
            raise ValueError("log_interval must be at least 1 step")
        self.log_interval = int(log_interval)
        self.stream = stream if stream is not None else sys.stdout
        self._start_time: float = 0.0
        self._last_emit_step: int = 0
        self._episode_returns: List[float] = []
        self._episode_lengths: List[int] = []

    def _init_callback(self) -> None:
        self._start_time = time.time()
        self._last_emit_step = 0
        self._episode_returns.clear()
        self._episode_lengths.clear()

    def _on_step(self) -> bool:
        infos: Iterable[dict] = self.locals.get("infos", ())
        for info in infos:
            episode = info.get("episode")
            if not episode:
                continue
            reward = float(episode.get("r", 0.0))
            length = int(episode.get("l", 0))
            self._episode_returns.append(reward)
            self._episode_lengths.append(length)

        if self.num_timesteps - self._last_emit_step >= self.log_interval:
            self._emit_summary()
        return True

    def _on_rollout_end(self) -> None:
        # Flush any buffered episode stats when a rollout finishes so the user
        # can see the final updates even if ``log_interval`` is large.
        self._emit_summary(force=True)

    def _emit_summary(self, force: bool = False) -> None:
        if not self._episode_returns:
            if not force:
                return
            # No completed episodes but we were asked to force output; emit a
            # heartbeat so the user knows training is still running.
            elapsed = time.time() - self._start_time
            fps = self.num_timesteps / elapsed if elapsed > 0 else float("nan")
            self._write(
                f"[train] steps={self.num_timesteps:,} (episodes pending) | elapsed={elapsed:,.1f}s | fps={fps:,.1f}"
            )
            self._last_emit_step = self.num_timesteps
            return

        elapsed = time.time() - self._start_time
        fps = self.num_timesteps / elapsed if elapsed > 0 else float("nan")
        mean_reward = sum(self._episode_returns) / len(self._episode_returns)
        best_reward = max(self._episode_returns)
        last_reward = self._episode_returns[-1]
        mean_length = sum(self._episode_lengths) / len(self._episode_lengths)

        self._write(
            "[train] steps={steps:,} | episodes={episodes} | "
            "reward(mean={mean:.2f}, last={last:.2f}, best={best:.2f}) | "
            "len(mean={length:.1f}) | elapsed={elapsed:,.1f}s | fps={fps:,.1f}".format(
                steps=self.num_timesteps,
                episodes=len(self._episode_returns),
                mean=mean_reward,
                last=last_reward,
                best=best_reward,
                length=mean_length,
                elapsed=elapsed,
                fps=fps,
            )
        )

        self._episode_returns.clear()
        self._episode_lengths.clear()
        self._last_emit_step = self.num_timesteps

    def _write(self, message: str) -> None:
        self.stream.write(message + "\n")
        if hasattr(self.stream, "flush"):
            self.stream.flush()

    def _on_training_end(self) -> None:
        # Ensure the last partial statistics are printed when training stops.
        self._emit_summary(force=True)
