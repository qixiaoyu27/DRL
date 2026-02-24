from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from drl_uav import config
from drl_uav.scenario_manager import load_scenarios
from drl_uav.wind_field import WindField


class FixedWingCoverageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.map_size = config.MAP_SIZE_M
        self.flight_altitude = config.FLIGHT_ALTITUDE_M
        self.fov_deg = config.CAMERA_FOV_DEG
        self.cruise_speed = config.CRUISE_SPEED_MPS
        self.dt = config.DT
        self.max_steps = config.EPISODE_MAX_STEPS
        self.max_turn_rate = np.deg2rad(config.MAX_TURN_RATE_DEG_S)

        self.wind_field = WindField(
            map_size=self.map_size,
            max_bg_wind=config.MAX_BACKGROUND_WIND_MPS,
            n_vortices=config.NUM_VORTICES,
            seed=seed,
        )
        self.scenarios = load_scenarios(config.SCENARIO_JSON)
        self.scenario_idx = -1

        self._init_jsbsim()

        swath = 2.0 * self.flight_altitude * np.tan(np.deg2rad(self.fov_deg) / 2.0)
        self.grid_resolution = swath / 2.0
        self.grid_size = int(np.ceil(self.map_size / self.grid_resolution))

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_dim = 4 + 2 + 1 + 1 + 10
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.state = np.zeros(3, dtype=np.float32)
        self.coverage = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.step_count = 0

    @staticmethod
    def _candidate_root_dirs(jsbsim_module) -> list[Path]:
        dirs: list[Path] = []

        # 1) hardcoded config first
        if config.JSBSIM_ROOT_DIR:
            dirs.append(Path(config.JSBSIM_ROOT_DIR))

        # 2) env var paths
        for key in ["JSBSIM_ROOT", "JSBSIM_ROOT_DIR"]:
            val = os.environ.get(key)
            if val:
                dirs.append(Path(val))

        # 3) package-relative guesses (pip/conda installs)
        pkg_dir = Path(jsbsim_module.__file__).resolve().parent
        dirs.extend([pkg_dir, pkg_dir.parent, pkg_dir / "data"])

        # 4) common Windows/Unix install locations
        dirs.extend(
            [
                Path("./jsbsim"),
                Path("./JSBSim"),
                Path("C:/Program Files/JSBSim"),
                Path("C:/Program Files (x86)/JSBSim"),
                Path("/usr/share/jsbsim"),
                Path("/usr/local/share/jsbsim"),
            ]
        )

        uniq: list[Path] = []
        for p in dirs:
            rp = p.resolve() if p.exists() else p
            if rp not in uniq:
                uniq.append(rp)
        return uniq

    @staticmethod
    def _contains_aircraft(root: Path) -> bool:
        return (root / "aircraft").exists()

    def _configure_jsbsim_paths(self, jsbsim_module) -> tuple[list[str], list[str]]:
        configured: list[str] = []
        searched: list[str] = []

        roots = self._candidate_root_dirs(jsbsim_module)
        for root in roots:
            searched.append(str(root))
            if self._contains_aircraft(root):
                if hasattr(self.jsbsim, "set_root_dir"):
                    self.jsbsim.set_root_dir(str(root))
                    configured.append(f"root={root}")
                if hasattr(self.jsbsim, "set_aircraft_path"):
                    self.jsbsim.set_aircraft_path(str(root / "aircraft"))
                    configured.append(f"aircraft={root / 'aircraft'}")
                if hasattr(self.jsbsim, "set_engine_path") and (root / "engine").exists():
                    self.jsbsim.set_engine_path(str(root / "engine"))
                    configured.append(f"engine={root / 'engine'}")
                if hasattr(self.jsbsim, "set_systems_path") and (root / "systems").exists():
                    self.jsbsim.set_systems_path(str(root / "systems"))
                    configured.append(f"systems={root / 'systems'}")
                break

        # Explicit path overrides (highest priority at end)
        if config.JSBSIM_AIRCRAFT_DIR and hasattr(self.jsbsim, "set_aircraft_path"):
            self.jsbsim.set_aircraft_path(config.JSBSIM_AIRCRAFT_DIR)
            configured.append(f"aircraft={config.JSBSIM_AIRCRAFT_DIR}")
        if config.JSBSIM_ENGINE_DIR and hasattr(self.jsbsim, "set_engine_path"):
            self.jsbsim.set_engine_path(config.JSBSIM_ENGINE_DIR)
            configured.append(f"engine={config.JSBSIM_ENGINE_DIR}")
        if config.JSBSIM_SYSTEMS_DIR and hasattr(self.jsbsim, "set_systems_path"):
            self.jsbsim.set_systems_path(config.JSBSIM_SYSTEMS_DIR)
            configured.append(f"systems={config.JSBSIM_SYSTEMS_DIR}")

        return configured, searched

    def _init_jsbsim(self):
        try:
            import jsbsim
        except Exception as exc:
            raise RuntimeError("JSBSim is required. Please install jsbsim package.") from exc

        self.jsbsim = jsbsim.FGFDMExec(None)
        self.jsbsim.set_dt(self.dt)
        configured, searched = self._configure_jsbsim_paths(jsbsim)

        loaded_model = None
        for model in config.JSBSIM_MODEL_CANDIDATES:
            if self.jsbsim.load_model(model):
                loaded_model = model
                break

        if loaded_model is None:
            msg = (
                "JSBSim model loading failed. Tried models="
                f"{config.JSBSIM_MODEL_CANDIDATES}. "
                f"Configured paths={configured if configured else 'none'}. "
                f"Searched root candidates={searched}. "
                "Please set drl_uav/config.py JSBSIM_ROOT_DIR or JSBSIM_AIRCRAFT_DIR."
            )
            raise RuntimeError(msg)

        self.jsbsim_model = loaded_model
        print(f"[JSBSim] loaded model: {self.jsbsim_model}")

    def _apply_jsbsim_state(self, turn_cmd: float) -> None:
        psi = float(self.state[2])
        self.jsbsim["ic/h-sl-ft"] = self.flight_altitude * 3.28084
        self.jsbsim["ic/u-fps"] = self.cruise_speed * 3.28084
        self.jsbsim["fcs/aileron-cmd-norm"] = np.clip(turn_cmd, -1.0, 1.0)
        self.jsbsim["fcs/elevator-cmd-norm"] = 0.0
        self.jsbsim["fcs/rudder-cmd-norm"] = 0.0
        self.jsbsim["attitude/psi-rad"] = psi
        self.jsbsim.run()

    def _cover_cells(self) -> int:
        x, y = self.state[:2]
        radius = self.grid_resolution
        gx_min = max(int((x - radius) / self.grid_resolution), 0)
        gx_max = min(int((x + radius) / self.grid_resolution), self.grid_size - 1)
        gy_min = max(int((y - radius) / self.grid_resolution), 0)
        gy_max = min(int((y + radius) / self.grid_resolution), self.grid_size - 1)

        newly = 0
        for gx in range(gx_min, gx_max + 1):
            for gy in range(gy_min, gy_max + 1):
                if self.coverage[gx, gy] == 0:
                    cx = (gx + 0.5) * self.grid_resolution
                    cy = (gy + 0.5) * self.grid_resolution
                    if (cx - x) ** 2 + (cy - y) ** 2 <= radius**2:
                        self.coverage[gx, gy] = 1
                        newly += 1
        return newly

    def _get_obs(self) -> np.ndarray:
        x, y, psi = self.state
        wind = self.wind_field.velocity(self.state[:2])
        coverage_ratio = float(self.coverage.mean())
        nearest = self.wind_field.nearest_vortex_features(self.state[:2], k=2)
        return np.array(
            [
                2.0 * (x / self.map_size) - 1.0,
                2.0 * (y / self.map_size) - 1.0,
                np.cos(psi),
                np.sin(psi),
                np.clip(wind[0] / 20.0, -1.0, 1.0),
                np.clip(wind[1] / 20.0, -1.0, 1.0),
                2.0 * coverage_ratio - 1.0,
                2.0 * (self.step_count / self.max_steps) - 1.0,
                *np.clip(nearest, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.coverage.fill(0)
        self.step_count = 0

        self.scenario_idx = (self.scenario_idx + 1) % len(self.scenarios)
        self.wind_field.reset_from_scenario(self.scenarios[self.scenario_idx])

        self.state[0] = self.map_size * 0.5
        self.state[1] = self.map_size * 0.5
        self.state[2] = np.random.uniform(0.0, 2.0 * np.pi)

        self.jsbsim.run_ic()
        self._cover_cells()
        return self._get_obs(), {"scenario_id": self.scenario_idx, "jsbsim_model": self.jsbsim_model}

    def step(self, action):
        turn_cmd = float(np.clip(action[0], -1.0, 1.0))
        self.step_count += 1

        self._apply_jsbsim_state(turn_cmd)
        self.state[2] += turn_cmd * self.max_turn_rate * self.dt

        heading = np.array([np.cos(self.state[2]), np.sin(self.state[2])], dtype=np.float32)
        air_velocity = self.cruise_speed * heading
        wind = self.wind_field.velocity(self.state[:2])
        ground_velocity = air_velocity + wind
        self.state[:2] += ground_velocity * self.dt

        out_of_bounds = not (0.0 <= self.state[0] <= self.map_size and 0.0 <= self.state[1] <= self.map_size)
        self.state[0] = np.clip(self.state[0], 0.0, self.map_size)
        self.state[1] = np.clip(self.state[1], 0.0, self.map_size)

        newly = self._cover_cells()
        coverage_ratio = float(self.coverage.mean())

        reward = 3.0 * newly - 0.02 - 0.05 * np.linalg.norm(wind)
        if out_of_bounds:
            reward -= 8.0
        if coverage_ratio > 0.95:
            reward += 80.0

        terminated = coverage_ratio > 0.95
        truncated = self.step_count >= self.max_steps
        info = {
            "coverage_ratio": coverage_ratio,
            "position": self.state[:2].copy(),
            "wind": wind.copy(),
            "scenario_id": self.scenario_idx,
            "jsbsim_model": self.jsbsim_model,
        }
        return self._get_obs(), float(reward), terminated, truncated, info
