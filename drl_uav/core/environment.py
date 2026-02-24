from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Vortex:
    cx: float
    cy: float
    gamma: float
    radius: float


class DynamicWindField:
    def __init__(self, map_size: int, wind_level: int = 0, num_vortices: int = 0, vortex_strength: float = 8.0):
        self.map_size = map_size
        self.wind_level = wind_level
        self.num_vortices = num_vortices
        self.vortex_strength = vortex_strength
        self.vortices: list[Vortex] = []
        self.t = 0

    def reset(self, rng: np.random.Generator) -> None:
        self.t = 0
        self.vortices = []
        for _ in range(self.num_vortices):
            self.vortices.append(
                Vortex(
                    cx=rng.uniform(0, self.map_size),
                    cy=rng.uniform(0, self.map_size),
                    gamma=rng.choice([-1.0, 1.0]) * self.vortex_strength * rng.uniform(0.6, 1.2),
                    radius=rng.uniform(3.0, 8.0),
                )
            )

    def step(self) -> None:
        self.t += 1

    def sample(self, x: float, y: float) -> np.ndarray:
        base_amp = [0.0, 2.0, 4.0, 6.0][min(self.wind_level, 3)]
        wx = base_amp * np.sin(0.015 * self.t + 0.2 * y)
        wy = base_amp * np.cos(0.013 * self.t + 0.17 * x)

        for v in self.vortices:
            dx, dy = x - v.cx, y - v.cy
            r = np.hypot(dx, dy) + 1e-6
            tangential = v.gamma * (r / v.radius if r < v.radius else v.radius / r)
            wx += -tangential * dy / r
            wy += tangential * dx / r
        return np.array([wx, wy], dtype=np.float32)


class LawnMowerBaseline:
    def __init__(self, map_size: int):
        self.map_size = map_size
        self.target_row = 0
        self.direction = 1

    def reset(self):
        self.target_row = 0
        self.direction = 1

    def action(self, state: dict) -> np.ndarray:
        x, y = state["x"], state["y"]
        desired_yaw = 0.0 if self.direction > 0 else np.pi
        if self.direction > 0 and x >= self.map_size - 1:
            desired_yaw = np.pi / 2
            if y >= self.target_row + 1:
                self.target_row += 1
                self.direction = -1
        elif self.direction < 0 and x <= 1:
            desired_yaw = np.pi / 2
            if y >= self.target_row + 1:
                self.target_row += 1
                self.direction = 1

        yaw_err = np.arctan2(np.sin(desired_yaw - state["yaw"]), np.cos(desired_yaw - state["yaw"]))
        roll_cmd = np.clip(0.8 * yaw_err, -1.0, 1.0)
        alt_cmd = np.clip((30.0 - state["alt"]) / 10.0, -1.0, 1.0)
        throttle_cmd = np.clip((18.0 - state["speed"]) / 8.0, -1.0, 1.0)
        return np.array([roll_cmd, alt_cmd, throttle_cmd], dtype=np.float32)


class FixedWingScanEnv:
    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg = cfg
        self.map_size = cfg["map_size"]
        self.max_steps = cfg["max_steps"]
        self.dt = cfg["dt"]
        self.coverage_target = cfg["coverage_target"]
        self.rng = np.random.default_rng(seed)

        self.wind = DynamicWindField(
            map_size=self.map_size,
            wind_level=cfg["wind_level"],
            num_vortices=cfg["num_vortices"],
            vortex_strength=cfg["vortex_strength"],
        )
        self.baseline = LawnMowerBaseline(self.map_size)
        self.grid = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        self.state = {}
        self.step_count = 0

    def set_difficulty(self, wind_level: int, num_vortices: int) -> None:
        self.wind.wind_level = wind_level
        self.wind.num_vortices = num_vortices

    def _observe(self) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.state["x"], self.state["y"]
        local_wind = self.wind.sample(x, y)
        coverage = self.grid.mean()
        obs = np.array(
            [
                x / self.map_size,
                y / self.map_size,
                self.state["yaw"] / np.pi,
                self.state["speed"] / 30.0,
                self.state["alt"] / 100.0,
                local_wind[0] / 20.0,
                local_wind[1] / 20.0,
                coverage,
                self.step_count / self.max_steps,
            ]
            + [0.0] * 27,
            dtype=np.float32,
        )
        grid_flat = self.grid.flatten().astype(np.float32)
        return obs, grid_flat

    def reset(self) -> tuple[np.ndarray, np.ndarray, dict]:
        self.grid.fill(0.0)
        self.state = {
            "x": 1.0,
            "y": 1.0,
            "yaw": 0.0,
            "speed": 16.0,
            "alt": 30.0,
        }
        self.step_count = 0
        self.wind.reset(self.rng)
        self.baseline.reset()
        obs, grid = self._observe()
        return obs, grid, self.state.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        self.step_count += 1
        self.wind.step()

        action = np.clip(action, -1.0, 1.0)
        roll_cmd, alt_cmd, throttle_cmd = action

        self.state["yaw"] += 0.12 * roll_cmd
        self.state["speed"] = np.clip(self.state["speed"] + 0.7 * throttle_cmd, 10.0, 28.0)
        self.state["alt"] = np.clip(self.state["alt"] + 0.8 * alt_cmd, 10.0, 80.0)

        air_vx = self.state["speed"] * np.cos(self.state["yaw"]) * 0.08
        air_vy = self.state["speed"] * np.sin(self.state["yaw"]) * 0.08
        wind_v = self.wind.sample(self.state["x"], self.state["y"]) * 0.08

        self.state["x"] = float(np.clip(self.state["x"] + air_vx + wind_v[0], 0, self.map_size - 1))
        self.state["y"] = float(np.clip(self.state["y"] + air_vy + wind_v[1], 0, self.map_size - 1))

        gx, gy = int(self.state["x"]), int(self.state["y"])
        prev_cov = self.grid.mean()
        self.grid[gy, gx] = 1.0
        coverage = self.grid.mean()

        new_cov_reward = (coverage - prev_cov) * 150.0
        smooth_penalty = -0.04 * np.linalg.norm(action)
        boundary_penalty = -0.3 if gx in (0, self.map_size - 1) or gy in (0, self.map_size - 1) else 0.0

        reward = float(new_cov_reward + smooth_penalty + boundary_penalty)
        done = self.step_count >= self.max_steps or coverage >= self.coverage_target

        obs, grid = self._observe()
        info = {
            "coverage": coverage,
            "baseline_action": self.baseline.action(self.state),
            "wind": self.wind.sample(self.state["x"], self.state["y"]),
        }
        return obs, grid, reward, done, info
