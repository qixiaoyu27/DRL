from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass
class RankineVortex:
    x: float
    y: float
    gamma: float
    core_radius: float

    def velocity(self, px: float, py: float) -> np.ndarray:
        dx = px - self.x
        dy = py - self.y
        r = np.hypot(dx, dy) + 1e-6
        if r <= self.core_radius:
            vt = self.gamma * r / (2.0 * np.pi * self.core_radius**2)
        else:
            vt = self.gamma / (2.0 * np.pi * r)
        ux = -vt * dy / r
        uy = vt * dx / r
        return np.array([ux, uy], dtype=np.float32)


class WindField:
    """Background wind + up to 4 Rankine vortices."""

    def __init__(self, background_xy: np.ndarray, vortices: list[RankineVortex]):
        self.background_xy = np.array(background_xy, dtype=np.float32)
        self.vortices = vortices

    @classmethod
    def from_json(cls, path: str | Path) -> "WindField":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vortices = [RankineVortex(**v) for v in data.get("vortices", [])]
        return cls(np.array(data["background_xy"], dtype=np.float32), vortices)

    @classmethod
    def random_field(
        cls,
        map_size_m: float = 4000.0,
        max_vortices: int = 4,
        rng: np.random.Generator | None = None,
    ) -> "WindField":
        rng = rng or np.random.default_rng()
        bg_speed = rng.uniform(0.0, 5.4)  # 3级风及以下
        bg_dir = rng.uniform(0, 2 * np.pi)
        background_xy = np.array([bg_speed * np.cos(bg_dir), bg_speed * np.sin(bg_dir)], dtype=np.float32)
        n_v = rng.integers(1, max_vortices + 1)
        vortices: list[RankineVortex] = []
        for _ in range(n_v):
            vortices.append(
                RankineVortex(
                    x=float(rng.uniform(0.1 * map_size_m, 0.9 * map_size_m)),
                    y=float(rng.uniform(0.1 * map_size_m, 0.9 * map_size_m)),
                    gamma=float(rng.uniform(-2200, 2200)),
                    core_radius=float(rng.uniform(80, 280)),
                )
            )
        return cls(background_xy=background_xy, vortices=vortices)

    def velocity(self, px: float, py: float) -> np.ndarray:
        w = self.background_xy.copy()
        for v in self.vortices:
            w += v.velocity(px, py)
        return w

    def to_json_dict(self) -> dict:
        return {
            "background_xy": self.background_xy.tolist(),
            "vortices": [
                {"x": v.x, "y": v.y, "gamma": v.gamma, "core_radius": v.core_radius}
                for v in self.vortices
            ],
        }
