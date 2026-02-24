from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class RankineVortex:
    center: np.ndarray
    circulation: float
    core_radius: float

    def velocity(self, point: np.ndarray) -> np.ndarray:
        rel = point - self.center
        r = np.linalg.norm(rel) + 1e-6
        tangential_dir = np.array([-rel[1], rel[0]]) / r
        if r <= self.core_radius:
            vt = self.circulation * r / (2.0 * np.pi * self.core_radius**2)
        else:
            vt = self.circulation / (2.0 * np.pi * r)
        return vt * tangential_dir


class WindField:
    def __init__(self, map_size: float = 4000.0, max_bg_wind: float = 5.4, n_vortices: int = 4, seed: int | None = None):
        self.map_size = map_size
        self.max_bg_wind = max_bg_wind
        self.n_vortices = n_vortices
        self.rng = np.random.default_rng(seed)
        self.background = np.zeros(2, dtype=np.float32)
        self.vortices: List[RankineVortex] = []

    def reset_random(self) -> None:
        speed = self.rng.uniform(0.0, self.max_bg_wind)
        direction = self.rng.uniform(0.0, 2.0 * np.pi)
        self.background = speed * np.array([np.cos(direction), np.sin(direction)], dtype=np.float32)
        self.vortices = []
        margin = 400.0
        for _ in range(self.n_vortices):
            center = self.rng.uniform(margin, self.map_size - margin, size=2)
            circulation = self.rng.uniform(-2400.0, 2400.0)
            core_radius = self.rng.uniform(80.0, 240.0)
            self.vortices.append(RankineVortex(center=center.astype(np.float32), circulation=float(circulation), core_radius=float(core_radius)))

    def reset_from_scenario(self, scenario: dict) -> None:
        b = scenario["background_wind"]
        speed = float(b["speed"])
        direction = float(b["direction"])
        self.background = speed * np.array([np.cos(direction), np.sin(direction)], dtype=np.float32)
        self.vortices = []
        for item in scenario["vortices"]:
            self.vortices.append(
                RankineVortex(
                    center=np.array(item["center"], dtype=np.float32),
                    circulation=float(item["circulation"]),
                    core_radius=float(item["core_radius"]),
                )
            )

    def velocity(self, point: np.ndarray) -> np.ndarray:
        v = self.background.astype(np.float32).copy()
        for vortex in self.vortices:
            v += vortex.velocity(point).astype(np.float32)
        return v

    def nearest_vortex_features(self, point: np.ndarray, k: int = 2) -> np.ndarray:
        data: List[Tuple[float, RankineVortex]] = []
        for vortex in self.vortices:
            d = np.linalg.norm(point - vortex.center)
            data.append((d, vortex))
        data.sort(key=lambda x: x[0])
        features = []
        for d, vortex in data[:k]:
            rel = (vortex.center - point) / self.map_size
            features.extend([rel[0], rel[1], d / self.map_size, vortex.circulation / 2400.0, vortex.core_radius / 240.0])
        while len(features) < k * 5:
            features.extend([0.0, 0.0, 1.0, 0.0, 0.0])
        return np.array(features, dtype=np.float32)
