"""Utilities for generating and sampling wind fields."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class WindFieldConfig:
    """Configuration for the procedural wind field.

    Attributes
    ----------
    seed:
        Random seed for reproducibility.
    grid_size:
        Number of grid cells along each axis of the square wind map.
    scale:
        Scaling factor for Perlin-like noise features. Higher values lead to
        smoother fields.
    max_speed:
        Maximum magnitude (m/s) of the wind vector that will be generated.
    altitude_layers:
        Number of independent wind layers stacked vertically. Each layer is
        blended based on the aircraft's altitude fraction within the
        configured range.
    altitude_range:
        Tuple describing the altitude range (m) covered by the wind map.
    """

    seed: int = 0
    grid_size: int = 32
    scale: float = 0.1
    max_speed: float = 20.0
    altitude_layers: int = 3
    altitude_range: Tuple[float, float] = (0.0, 3000.0)


class WindField:
    """Procedural wind field that can be queried at any position."""

    def __init__(self, config: WindFieldConfig | None = None) -> None:
        self.config = config or WindFieldConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._field = self._generate_field()

    def _generate_field(self) -> np.ndarray:
        grid = self.config.grid_size
        layers = self.config.altitude_layers
        noise = self._rng.normal(size=(layers, grid, grid, 2))
        noise = self._smooth(noise)
        # Normalize and scale to the requested maximum speed.
        norms = np.linalg.norm(noise, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = noise / norms
        return normalized * self.config.max_speed

    def _smooth(self, noise: np.ndarray) -> np.ndarray:
        """Apply a cheap separable smoothing filter to reduce sharp gradients."""
        kernel = np.array([0.25, 0.5, 0.25])
        passes = max(1, int(np.ceil(1.0 / max(self.config.scale, 1e-3))))
        for _ in range(passes):
            for axis in (-3, -2):
                noise = np.apply_along_axis(
                    lambda m: np.convolve(m, kernel, mode="same"), axis, noise
                )
        return noise

    def sample(self, x: float, y: float, altitude: float) -> np.ndarray:
        """Return the local wind vector in m/s for a given position.

        Parameters
        ----------
        x, y:
            Position in meters within the square map (0 .. grid_size).
        altitude:
            Altitude in meters above ground level.
        """
        grid = self.config.grid_size
        fx = np.clip(x, 0, grid - 1e-6)
        fy = np.clip(y, 0, grid - 1e-6)
        ix, iy = int(fx), int(fy)
        tx, ty = fx - ix, fy - iy
        # Bilinear interpolation inside the altitude layer.
        layers = self.config.altitude_layers
        alt_min, alt_max = self.config.altitude_range
        frac = np.clip((altitude - alt_min) / (alt_max - alt_min + 1e-6), 0.0, 1.0)
        layer_pos = frac * (layers - 1)
        il = int(layer_pos)
        tl = layer_pos - il

        def get(layer: int, gx: int, gy: int) -> np.ndarray:
            layer = int(np.clip(layer, 0, layers - 1))
            gx = int(np.clip(gx, 0, grid - 1))
            gy = int(np.clip(gy, 0, grid - 1))
            return self._field[layer, gx, gy]

        v00 = get(il, ix, iy)
        v10 = get(il, ix + 1, iy)
        v01 = get(il, ix, iy + 1)
        v11 = get(il, ix + 1, iy + 1)
        lower = (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11

        v00_u = get(il + 1, ix, iy)
        v10_u = get(il + 1, ix + 1, iy)
        v01_u = get(il + 1, ix, iy + 1)
        v11_u = get(il + 1, ix + 1, iy + 1)
        upper = (1 - tx) * (1 - ty) * v00_u + tx * (1 - ty) * v10_u + (1 - tx) * ty * v01_u + tx * ty * v11_u

        return (1 - tl) * lower + tl * upper

    def reset(self, seed: int | None = None) -> None:
        """Regenerate the wind field with a new random seed."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._field = self._generate_field()
