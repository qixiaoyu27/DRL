"""JSBSim-based fixed-wing UAV environment with ERA5 wind integration."""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import jsbsim
import numpy as np

from fixedwing_rl.utils.wind_field import WindField3D


def _deg2rad(angle_deg: float) -> float:
    return angle_deg * math.pi / 180.0


@dataclass
class FixedWingEnvConfig:
    """Configuration options for :class:`FixedWingJSBSimEnv`."""

    initial_longitude: float
    initial_latitude: float
    initial_altitude_m: float
    target_waypoint: Tuple[float, float, float]
    max_time_s: int = 600
    dt: float = 0.02
    cruise_speed_mps: float = 40.0
    energy_weight: float = 0.01
    time_weight: float = 1.0
    stability_weight: float = 0.5
    max_roll_deg: float = 45.0
    max_pitch_deg: float = 20.0
    max_throttle: float = 1.0
    jsbsim_root: Optional[pathlib.Path] = None
    aircraft: str = "c172x"
    wind_field: Optional[WindField3D] = None
    max_wind_speed_ms: float = 12.0


class FixedWingJSBSimEnv(gym.Env):
    """Gymnasium-compatible environment for fixed-wing path planning.

    The environment exposes GPS (lat, lon, altitude), IMU (Euler angles, body rates),
    and barometric altitude as part of the observation vector. Actions control
    aileron, elevator, rudder, and throttle. A configurable :class:`WindField3D`
    is injected into the JSBSim simulation, allowing training against realistic
    ERA5 wind fields. Rewards combine time-to-goal and energy consumption to
    encourage fast yet efficient trajectories. The environment is designed for
    training with PPO or SAC in a 3D space while remaining robust up to 3rd level
    (≈10.8 m/s) wind conditions.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1.0}

    def __init__(self, config: FixedWingEnvConfig):
        super().__init__()
        self.config = config
        self._wind_field = config.wind_field
        self._wind_cache: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}
        self._dt = config.dt
        self._time = 0.0
        self._trajectory: List[Tuple[float, float, float]] = []

        self._sim = jsbsim.FGFDMExec(str(config.jsbsim_root) if config.jsbsim_root else None)
        self._sim.load_model(config.aircraft)
        self._sim.set_dt(self._dt)
        self._sim.run_ic()

        # Define action and observation spaces.
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, config.max_throttle], dtype=np.float32),
            dtype=np.float32,
        )

        obs_high = np.array(
            [
                math.pi,  # heading
                math.pi / 2,  # pitch
                math.pi / 2,  # roll
                200.0,  # airspeed
                200.0,  # groundspeed
                100.0,  # climb rate
                10000.0,  # altitude
                100.0,  # roll rate
                100.0,  # pitch rate
                100.0,  # yaw rate
                math.pi,  # wind direction
                50.0,  # wind magnitude
            ],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.reset()

    def _set_initial_conditions(self) -> None:
        self._sim['ic/lat-geod-deg'] = self.config.initial_latitude
        self._sim['ic/long-gc-deg'] = self.config.initial_longitude
        self._sim['ic/h-sl-ft'] = self.config.initial_altitude_m * 3.28084
        self._sim['ic/psi-true-deg'] = 0.0
        self._sim['ic/theta-deg'] = 0.0
        self._sim['ic/phi-deg'] = 0.0
        self._sim['ic/ub-fps'] = self.config.cruise_speed_mps * 3.28084
        self._sim.run_ic()

    def _update_wind(self) -> None:
        if not self._wind_field:
            return
        position = self._current_position()
        wind_ned = self._wind_field.vector_at(*position)
        # JSBSim expects wind in ft/s.
        wind_fps = [component * 3.28084 for component in wind_ned]
        self._sim['atmosphere/wind-north-fps'] = wind_fps[0]
        self._sim['atmosphere/wind-east-fps'] = wind_fps[1]
        self._sim['atmosphere/wind-down-fps'] = -wind_fps[2]

    def _current_position(self) -> Tuple[float, float, float]:
        lat = float(self._sim['position/lat-gc-deg'])
        lon = float(self._sim['position/long-gc-deg'])
        alt = float(self._sim['position/h-sl-meters'])
        return lat, lon, alt

    def _current_orientation(self) -> Tuple[float, float, float]:
        roll = _deg2rad(float(self._sim['attitude/phi-deg']))
        pitch = _deg2rad(float(self._sim['attitude/theta-deg']))
        yaw = _deg2rad(float(self._sim['attitude/psi-deg']))
        return roll, pitch, yaw

    def _airspeeds(self) -> Tuple[float, float]:
        airspeed = float(self._sim['velocities/vtrue-kts']) * 0.514444
        groundspeed = float(self._sim['velocities/vg-kts']) * 0.514444
        return airspeed, groundspeed

    def _body_rates(self) -> Tuple[float, float, float]:
        p = float(self._sim['velocities/p-rad_sec'])
        q = float(self._sim['velocities/q-rad_sec'])
        r = float(self._sim['velocities/r-rad_sec'])
        return p, q, r

    def _wind_vector(self) -> Tuple[float, float, float]:
        if not self._wind_field:
            return (0.0, 0.0, 0.0)
        return self._wind_field.vector_at(*self._current_position())

    def _energy_proxy(self, throttle: float) -> float:
        return throttle ** 2

    def _distance_to_target(self) -> float:
        lat, lon, alt = self._current_position()
        target_lat, target_lon, target_alt = self.config.target_waypoint
        return math.sqrt(
            (lat - target_lat) ** 2 * (111_139.0 ** 2)
            + (lon - target_lon) ** 2 * (111_139.0 * math.cos(_deg2rad(lat))) ** 2
            + (alt - target_alt) ** 2
        )

    def _terminal(self) -> Tuple[bool, bool]:
        distance = self._distance_to_target()
        timeout = self._time >= self.config.max_time_s
        success = distance < 50.0
        crash = bool(self._sim['fcs/over-g']) or bool(self._sim['simulation/crashed'])
        truncated = timeout
        terminated = success or crash
        return terminated, truncated

    def _observe(self) -> np.ndarray:
        roll, pitch, yaw = self._current_orientation()
        airspeed, groundspeed = self._airspeeds()
        climb_rate = float(self._sim['velocities/h-dot-fps']) * 0.3048
        altitude = float(self._sim['position/h-sl-meters'])
        p, q, r = self._body_rates()
        wind = self._wind_vector()

        observation = np.array(
            [
                yaw,
                pitch,
                roll,
                airspeed,
                groundspeed,
                climb_rate,
                altitude,
                p,
                q,
                r,
                math.atan2(wind[1], wind[0]),
                np.linalg.norm(wind),
            ],
            dtype=np.float32,
        )
        return observation

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self._set_initial_conditions()
        self._time = 0.0
        self._trajectory.clear()
        self._update_wind()
        observation = self._observe()
        info = {"position": self._current_position()}
        self._trajectory.append(self._current_position())
        return observation, info

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        aileron, elevator, rudder, throttle = action

        self._sim['fcs/aileron-cmd-norm'] = float(np.clip(aileron, -1.0, 1.0))
        self._sim['fcs/elevator-cmd-norm'] = float(np.clip(elevator, -1.0, 1.0))
        self._sim['fcs/rudder-cmd-norm'] = float(np.clip(rudder, -1.0, 1.0))
        self._sim['fcs/throttle-cmd-norm'] = float(np.clip(throttle, 0.0, self.config.max_throttle))

        self._update_wind()
        self._sim.run()
        self._time += self._dt

        observation = self._observe()
        distance = self._distance_to_target()
        energy_penalty = self.config.energy_weight * self._energy_proxy(float(throttle))
        time_penalty = self.config.time_weight * self._dt
        stability_penalty = self.config.stability_weight * np.mean(np.square(observation[2:5]))
        reward = -distance * 1e-3 - energy_penalty - time_penalty - stability_penalty

        terminated, truncated = self._terminal()
        if terminated:
            reward += 100.0 if distance < 50.0 else -100.0

        info = {
            "position": self._current_position(),
            "distance": distance,
            "energy_penalty": energy_penalty,
            "time": self._time,
        }
        self._trajectory.append(info["position"])
        return observation, reward, terminated, truncated, info

    @property
    def trajectory(self) -> List[Tuple[float, float, float]]:
        return list(self._trajectory)

    def render(self):
        return None

    def close(self):
        if hasattr(self, "_sim"):
            self._sim = None
