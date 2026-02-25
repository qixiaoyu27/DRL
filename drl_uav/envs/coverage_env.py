from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .wind_field import WindField


@dataclass
class EnvConfig:
    map_size_m: float = 4000.0
    grid_n: int = 80
    dt_s: float = 1.0
    cruise_speed_mps: float = 25.0
    min_speed_mps: float = 18.0
    max_speed_mps: float = 32.0
    fov_deg: float = 120.0
    sensor_range_m: float = 1000.0
    max_steps: int = 1800
    use_residual_policy: bool = True
    use_attention_features: bool = True


class FixedWingCoverageEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: EnvConfig, wind_json: str | None = None, seed: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.wind = WindField.from_json(wind_json) if wind_json else WindField.random_field(cfg.map_size_m, rng=self.rng)

        self.observation_dim = 6 + 5 * 4 * 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32))

        self.cover = np.zeros((cfg.grid_n, cfg.grid_n), dtype=np.uint8)
        self.pos = np.zeros(2, dtype=np.float32)
        self.heading = 0.0
        self.speed = cfg.cruise_speed_mps
        self.step_count = 0
        self.last_coverage = 0.0

    def _baseline_control(self) -> np.ndarray:
        lane = int(self.pos[1] / self.cfg.map_size_m * 20)
        target_heading = 0.0 if lane % 2 == 0 else math.pi
        if self.pos[0] < 100 and lane % 2 == 1:
            target_heading = math.pi / 2
        elif self.pos[0] > self.cfg.map_size_m - 100 and lane % 2 == 0:
            target_heading = math.pi / 2
        err = math.atan2(math.sin(target_heading - self.heading), math.cos(target_heading - self.heading))
        return np.array([np.clip(err / 0.4, -1, 1), 0.0], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        self_state = np.array([
            self.pos[0] / self.cfg.map_size_m,
            self.pos[1] / self.cfg.map_size_m,
            math.cos(self.heading),
            math.sin(self.heading),
            self.speed / self.cfg.max_speed_mps,
            self.last_coverage,
        ], dtype=np.float32)

        rays = []
        half = math.radians(self.cfg.fov_deg / 2)
        for a in np.linspace(-half, half, 5):
            r_heading = self.heading + a
            for r in np.linspace(250.0, self.cfg.sensor_range_m, 4):
                sx = self.pos[0] + r * math.cos(r_heading)
                sy = self.pos[1] + r * math.sin(r_heading)
                sx = float(np.clip(sx, 0, self.cfg.map_size_m))
                sy = float(np.clip(sy, 0, self.cfg.map_size_m))
                w = self.wind.velocity(sx, sy)
                rays.extend([w[0] / 15.0, w[1] / 15.0])
        return np.concatenate([self_state, np.array(rays, dtype=np.float32)]).astype(np.float32)

    def _coverage_ratio(self) -> float:
        return float(self.cover.mean())

    def _update_cover(self):
        gx = int(np.clip(self.pos[0] / self.cfg.map_size_m * self.cfg.grid_n, 0, self.cfg.grid_n - 1))
        gy = int(np.clip(self.pos[1] / self.cfg.map_size_m * self.cfg.grid_n, 0, self.cfg.grid_n - 1))
        self.cover[gy, gx] = 1

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and options.get("wind_json"):
            self.wind = WindField.from_json(options["wind_json"])
        elif options and options.get("randomize_wind", True):
            self.wind = WindField.random_field(self.cfg.map_size_m, rng=self.rng)
        self.cover.fill(0)
        self.pos[:] = np.array([120.0, 120.0], dtype=np.float32)
        self.heading = 0.1
        self.speed = self.cfg.cruise_speed_mps
        self.step_count = 0
        self.last_coverage = 0.0
        self._update_cover()
        return self._obs(), {}

    def step(self, action):
        base = self._baseline_control()
        if self.cfg.use_residual_policy:
            ctl = np.clip(base + 0.6 * np.array(action, dtype=np.float32), -1.0, 1.0)
        else:
            ctl = np.array(action, dtype=np.float32)

        yaw_rate = float(ctl[0]) * math.radians(16.0)
        throttle = float(ctl[1])
        self.heading += yaw_rate * self.cfg.dt_s
        self.speed = float(np.clip(self.speed + throttle * 0.8, self.cfg.min_speed_mps, self.cfg.max_speed_mps))

        air_vel = np.array([self.speed * math.cos(self.heading), self.speed * math.sin(self.heading)], dtype=np.float32)
        ground_vel = air_vel + self.wind.velocity(float(self.pos[0]), float(self.pos[1]))
        self.pos += ground_vel * self.cfg.dt_s
        self.pos[:] = np.clip(self.pos, 0.0, self.cfg.map_size_m)

        prev_cov = self._coverage_ratio()
        self._update_cover()
        cov = self._coverage_ratio()
        self.last_coverage = cov

        reward_cov = (cov - prev_cov) * 400.0
        reward_step = -0.03
        reward_smooth = -0.01 * abs(yaw_rate)
        reward = reward_cov + reward_step + reward_smooth

        self.step_count += 1
        terminated = cov >= 0.995
        truncated = self.step_count >= self.cfg.max_steps

        info = {
            "coverage": cov,
            "baseline_action": base,
            "residual_action": np.array(action, dtype=np.float32),
            "ground_speed": float(np.linalg.norm(ground_vel)),
        }
        return self._obs(), float(reward), terminated, truncated, info
