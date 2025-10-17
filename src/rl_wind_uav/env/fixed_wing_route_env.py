"""Gymnasium environment for routing a fixed-wing UAV in a wind field."""
from __future__ import annotations

import io
import math
import os
import subprocess
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import jsbsim  # type: ignore
except ImportError as exc:  # pragma: no cover - imported dynamically by the user
    raise ImportError(
        "JSBSim python bindings are required. Install with `pip install jsbsim`."
    ) from exc

from .wind_field import WindField, WindFieldConfig


@dataclass
class RouteConfig:
    """Configuration options controlling the environment dynamics."""

    jsbsim_root: str
    aircraft: str = "c310"
    init_lat: float = 37.628559
    init_lon: float = -122.393561
    init_altitude_m: float = 600.0
    goal_lat: float = 37.648559
    goal_lon: float = -122.363561
    goal_threshold_m: float = 150.0
    integration_dt: float = 0.1
    action_hold_time: float = 0.5
    episode_time_s: float = 600.0
    enable_flightgear: bool = False
    flightgear_path: Optional[str] = None
    flightgear_port: int = 5502
    flightgear_stream_hz: int = 60
    run_fg_headless: bool = True
    suppress_jsbsim_output: bool = True
    wind_config: WindFieldConfig = field(default_factory=WindFieldConfig)
    max_bank_deg: float = 25.0
    max_pitch_deg: float = 10.0


class FlightGearSession:
    """Helper to manage a FlightGear process for visualization."""

    def __init__(self, exe_path: str, port: int, headless: bool = True, stream_hz: int = 60) -> None:
        self.exe_path = exe_path
        self.port = port
        self.headless = headless
        self.stream_hz = max(1, int(stream_hz))
        self._process: Optional[subprocess.Popen[str]] = None

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        args = [
            self.exe_path,
            "--aircraft=c310",
            f"--generic=socket,in,{self.stream_hz},,{self.port},udp",
            "--fdm=external",
        ]
        if self.headless:
            args.extend(["--timeofday=noon", "--disable-sound", "--fog-disable", "--geometry=800x600"])
            args.append("--enable-freeze")
        self._process = subprocess.Popen(args)
        # Give FlightGear some time to boot to avoid connection errors.
        time.sleep(5)

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None


class FixedWingRouteEnv(gym.Env[np.ndarray, np.ndarray]):
    """Route-finding environment for a fixed-wing aircraft.

    The agent controls commanded bank and pitch angles. JSBSim propagates the
    aircraft dynamics, while a procedurally generated wind field is queried at
    the current aircraft position. The reward encourages reaching the goal
    quickly while staying in low-wind regions.
    """

    metadata = {"render_modes": ["flightgear", "none"], "render_fps": 10}

    def __init__(self, config: RouteConfig) -> None:
        super().__init__()
        self.config = config
        self._wind_field = WindField(config.wind_config)
        with self._maybe_suppress_jsbsim_output():
            self._sim = jsbsim.FGFDMExec(config.jsbsim_root)
            self._sim.set_debug_level(0)
            self._sim.load_model(config.aircraft)
            self._sim.set_dt(config.integration_dt)
            self._sim.disable_output()
        self._dt = config.integration_dt
        self._hold_steps = max(1, int(config.action_hold_time / self._dt))
        self._max_steps = int(config.episode_time_s / self._dt)
        self._steps = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._fg_session: Optional[FlightGearSession] = None
        self._fg_output_enabled = False
        self._origin_lat_rad = math.radians(self.config.init_lat)
        self._origin_lon_rad = math.radians(self.config.init_lon)

        self._configure_flightgear_output()

        obs_high = np.array([
            math.pi,  # heading error
            math.pi / 2,  # track angle
            1000.0,  # altitude error (m)
            120.0,  # airspeed (m/s)
            60.0,  # wind x (m/s)
            60.0,  # wind y (m/s)
            50000.0,  # downrange distance to goal (m)
            50000.0,  # crossrange distance to goal (m)
        ], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._wind_field.reset(seed)
        self._steps = 0
        self._prev_action[:] = 0.0

        self._setup_initial_conditions()
        with self._maybe_suppress_jsbsim_output():
            self._sim.run_ic()
        observation = self._observe()
        info = {"wind": self._current_wind}
        return observation, info

    def step(self, action: np.ndarray):
        clipped_action = np.clip(action, -1.0, 1.0)
        bank_cmd = clipped_action[0] * math.radians(self.config.max_bank_deg)
        pitch_cmd = clipped_action[1] * math.radians(self.config.max_pitch_deg)
        self._prev_action = clipped_action.astype(np.float32)

        for _ in range(self._hold_steps):
            self._apply_control(bank_cmd, pitch_cmd)
            if not self._sim.run():
                break
        self._steps += 1

        obs = self._observe()
        reward, terminated = self._compute_reward()
        truncated = self._steps >= self._max_steps
        info = {"wind": self._current_wind, "action": clipped_action}
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # JSBSim helpers
    # ------------------------------------------------------------------
    def _setup_initial_conditions(self) -> None:
        lat_deg = self.config.init_lat
        lon_deg = self.config.init_lon
        alt_m = self.config.init_altitude_m

        self._sim['ic/lat-geod-deg'] = lat_deg
        self._sim['ic/lon-geod-deg'] = lon_deg
        self._sim['ic/h-sl-ft'] = alt_m * 3.28084
        self._sim['ic/psi-true-deg'] = 90.0
        self._sim['ic/u-fps'] = 200.0
        self._sim['ic/v-fps'] = 0.0
        self._sim['ic/w-fps'] = 0.0
        self._sim['ic/roc-fpm'] = 0.0
        self._sim['ic/theta-deg'] = 0.0
        self._sim['ic/phi-deg'] = 0.0
        self._sim['ap/aileron-cmd-norm'] = 0.0
        self._sim['ap/elevator-cmd-norm'] = 0.0
        self._sim['ap/rudder-cmd-norm'] = 0.0
        self._sim['ap/throttle-cmd-norm'] = 0.8
        self._sim['gear/gear-pos-norm'] = 1.0
        self._sim['propulsion/set-running'] = 1

    def _apply_control(self, bank_cmd: float, pitch_cmd: float) -> None:
        self._sim['ap/phi-cmd-rad'] = bank_cmd
        self._sim['ap/theta-cmd-rad'] = pitch_cmd
        self._sim['ap/throttle-cmd-norm'] = 0.8

    # ------------------------------------------------------------------
    # Observations and rewards
    # ------------------------------------------------------------------
    def _observe(self) -> np.ndarray:
        lat = float(self._sim['position/lat-gc-rad'])
        lon = float(self._sim['position/long-gc-rad'])
        alt = float(self._sim['position/h-sl-meters'])
        true_heading = float(self._sim['attitude/psi-rad'])
        airspeed = float(self._sim['velocities/vtrue-kts']) * 0.514444

        goal_vec = self._goal_vector(lat, lon)
        track_angle = math.atan2(goal_vec[1], goal_vec[0])
        heading_error = self._wrap_angle(track_angle - true_heading)

        enu = self._enu_from_latlon(lat, lon)
        wind = self._wind_field.sample(
            enu[0] / 1000.0 + self._wind_field.config.grid_size / 2,
            enu[1] / 1000.0 + self._wind_field.config.grid_size / 2,
            alt,
        )
        self._current_wind = wind

        obs = np.array(
            [
                heading_error,
                track_angle,
                alt - self.config.init_altitude_m,
                airspeed,
                wind[0],
                wind[1],
                goal_vec[0],
                goal_vec[1],
            ],
            dtype=np.float32,
        )
        return obs

    def _goal_vector(self, lat: float, lon: float) -> np.ndarray:
        lat1 = math.radians(self.config.goal_lat)
        lon1 = math.radians(self.config.goal_lon)
        r_earth = 6371000.0
        dn = (lat1 - lat) * r_earth
        de = (lon1 - lon) * r_earth * math.cos((lat1 + lat) / 2.0)
        return np.array([de, dn], dtype=np.float32)

    def _enu_from_latlon(self, lat: float, lon: float) -> np.ndarray:
        r_earth = 6371000.0
        dn = (lat - self._origin_lat_rad) * r_earth
        de = (lon - self._origin_lon_rad) * r_earth * math.cos((lat + self._origin_lat_rad) / 2.0)
        return np.array([de, dn], dtype=np.float32)

    def _compute_reward(self) -> Tuple[float, bool]:
        pos_lat = float(self._sim['position/lat-gc-rad'])
        pos_lon = float(self._sim['position/long-gc-rad'])
        pos_alt = float(self._sim['position/h-sl-meters'])

        goal_vec = self._goal_vector(pos_lat, pos_lon)
        distance = np.linalg.norm(goal_vec)
        wind_mag = float(np.linalg.norm(self._current_wind))
        altitude_penalty = abs(pos_alt - self.config.init_altitude_m)

        reward = -0.001 * distance - 0.05 * wind_mag - 0.001 * altitude_penalty
        terminated = distance < self.config.goal_threshold_m or pos_alt < 50.0
        if terminated:
            reward += 100.0 if distance < self.config.goal_threshold_m else -100.0
        return reward, terminated

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    # ------------------------------------------------------------------
    # Rendering / FlightGear hook
    # ------------------------------------------------------------------
    def render(self):  # pragma: no cover - requires external executable
        if not self.config.enable_flightgear:
            return None
        self._configure_flightgear_output()
        if self._fg_session is None:
            exe = self.config.flightgear_path or self._detect_flightgear()
            self._fg_session = FlightGearSession(
                exe,
                self.config.flightgear_port,
                self.config.run_fg_headless,
                self.config.flightgear_stream_hz,
            )
            self._fg_session.start()
        return None

    def close(self) -> None:
        if self._fg_session is not None:
            self._fg_session.close()
        self._sim = None  # type: ignore
        self._fg_session = None
        self._fg_output_enabled = False

    def _detect_flightgear(self) -> str:  # pragma: no cover - depends on platform
        candidates = [
            os.getenv("FLIGHTGEAR_EXE"),
            "/usr/games/fgfs",
            "/Applications/FlightGear.app/Contents/MacOS/fgfs",
            str(Path.home() / "Applications/FlightGear.app/Contents/MacOS/fgfs"),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        raise FileNotFoundError(
            "Could not auto-detect FlightGear executable. Provide `flightgear_path` in RouteConfig."
        )

    @contextmanager
    def _maybe_suppress_jsbsim_output(self):
        if not getattr(self.config, "suppress_jsbsim_output", False):
            yield
            return
        buffer = io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(buffer):
            yield

    def _configure_flightgear_output(self) -> None:
        if self._fg_output_enabled or not self.config.enable_flightgear:
            return

        rate = max(1, int(self.config.flightgear_stream_hz))
        directive = f"""
<output name="flightgear">
  <type>FLIGHTGEAR</type>
  <rate>{rate}</rate>
  <protocol>udp</protocol>
  <address>127.0.0.1</address>
  <port>{self.config.flightgear_port}</port>
</output>
"""

        with self._maybe_suppress_jsbsim_output():
            self._sim.set_output_directive(directive)
        self._fg_output_enabled = True
