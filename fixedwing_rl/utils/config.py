"""Configuration dataclasses and loader for training scripts."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional

import yaml


def _filter_kwargs(cls, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return only keyword arguments that are valid for *cls*."""
    if not data:
        return {}
    allowed = {f.name for f in fields(cls)}
    return {key: value for key, value in data.items() if key in allowed}


@dataclass
class AlgorithmConfig:
    """Algorithm-related hyper-parameters."""

    name: str = "ppo"
    total_timesteps: int = 200_000
    device: str = "cuda"
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AlgorithmConfig":
        kwargs = _filter_kwargs(cls, data)
        if kwargs.get("policy_kwargs") is None:
            kwargs["policy_kwargs"] = {}
        return cls(**kwargs)


@dataclass
class WindFieldConfig:
    """ERA5 wind field configuration."""

    era5_path: str = ""
    u_component: str = "u10"
    v_component: str = "v10"
    w_component: str = "w"
    altitude: str = "level"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "WindFieldConfig":
        kwargs = _filter_kwargs(cls, data)
        return cls(**kwargs)


@dataclass
class EnvironmentConfig:
    """Environment parameters controlling JSBSim and reward shaping."""

    initial_latitude: float = 34.5
    initial_longitude: float = -117.5
    initial_altitude_m: float = 1500.0
    target_latitude: float = 34.8
    target_longitude: float = -117.2
    target_altitude: float = 1600.0
    max_time_s: int = 900
    dt: float = 0.02
    cruise_speed_mps: float = 40.0
    energy_weight: float = 0.01
    time_weight: float = 1.0
    stability_weight: float = 0.5
    max_roll_deg: float = 45.0
    max_pitch_deg: float = 20.0
    max_throttle: float = 1.0
    jsbsim_root: Optional[str] = None
    aircraft: str = "c172x"
    max_wind_speed_ms: float = 12.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EnvironmentConfig":
        payload: Dict[str, Any] = dict(data or {})
        target = payload.pop("target", {}) or {}
        initial_altitude = payload.pop("initial_altitude", None)
        if initial_altitude is not None and "initial_altitude_m" not in payload:
            payload["initial_altitude_m"] = initial_altitude
        for key in ("latitude", "longitude", "altitude"):
            target_value = target.get(key)
            if target_value is not None:
                payload[f"target_{key}"] = target_value
        kwargs = _filter_kwargs(cls, payload)
        return cls(**kwargs)

    def to_fixedwing_config(self, wind_field) -> "FixedWingEnvConfig":  # pragma: no cover - convenience wrapper
        from fixedwing_rl.envs.fixedwing_jsbsim_env import FixedWingEnvConfig

        jsbsim_root = pathlib.Path(self.jsbsim_root).expanduser() if self.jsbsim_root else None
        return FixedWingEnvConfig(
            initial_longitude=self.initial_longitude,
            initial_latitude=self.initial_latitude,
            initial_altitude_m=self.initial_altitude_m,
            target_waypoint=(self.target_latitude, self.target_longitude, self.target_altitude),
            max_time_s=self.max_time_s,
            dt=self.dt,
            cruise_speed_mps=self.cruise_speed_mps,
            energy_weight=self.energy_weight,
            time_weight=self.time_weight,
            stability_weight=self.stability_weight,
            max_roll_deg=self.max_roll_deg,
            max_pitch_deg=self.max_pitch_deg,
            max_throttle=self.max_throttle,
            jsbsim_root=jsbsim_root,
            aircraft=self.aircraft,
            wind_field=wind_field,
            max_wind_speed_ms=self.max_wind_speed_ms,
        )


@dataclass
class EvaluationConfig:
    """Evaluation rollout parameters for visualisation."""

    steps: int = 2000

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EvaluationConfig":
        kwargs = _filter_kwargs(cls, data)
        return cls(**kwargs)


@dataclass
class OutputConfig:
    """Filesystem locations for outputs."""

    model_dir: str = "checkpoints"
    plots_dir: str = "plots"
    tensorboard_log: str = "logs"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "OutputConfig":
        kwargs = _filter_kwargs(cls, data)
        return cls(**kwargs)


@dataclass
class TrainingConfig:
    """Top-level configuration object assembled from YAML."""

    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    wind_field: WindFieldConfig = field(default_factory=WindFieldConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        data = data or {}
        return cls(
            algorithm=AlgorithmConfig.from_dict(data.get("algorithm")),
            environment=EnvironmentConfig.from_dict(data.get("environment")),
            wind_field=WindFieldConfig.from_dict(data.get("wind_field")),
            evaluation=EvaluationConfig.from_dict(data.get("evaluation")),
            output=OutputConfig.from_dict(data.get("output")),
        )


def load_training_config(path: str | pathlib.Path) -> TrainingConfig:
    """Load :class:`TrainingConfig` from a YAML file."""

    config_path = pathlib.Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    return TrainingConfig.from_dict(payload)


__all__ = [
    "AlgorithmConfig",
    "EnvironmentConfig",
    "EvaluationConfig",
    "OutputConfig",
    "TrainingConfig",
    "WindFieldConfig",
    "load_training_config",
]
