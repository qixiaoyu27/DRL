"""Helper utilities for loading default training and inference configuration."""
from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from rl_wind_uav.env.fixed_wing_route_env import RouteConfig
from rl_wind_uav.env.wind_field import WindFieldConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_TRAIN_CONFIG = CONFIG_DIR / "train.json"
DEFAULT_INFERENCE_CONFIG = CONFIG_DIR / "inference.json"

PPO_KNOWN_KEYS: Tuple[str, ...] = (
    "policy",
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "clip_range_vf",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "target_kl",
    "normalize_advantage",
    "use_sde",
    "sde_sample_freq",
    "policy_kwargs",
    "verbose",
    "tensorboard_log",
)

ROUTE_FIELD_NAMES = {field.name for field in fields(RouteConfig)} - {"jsbsim_root", "wind_config"}


class ConfigurationError(RuntimeError):
    """Raised when mandatory configuration values are missing."""


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(explicit_path: Optional[Path], fallback: Path) -> Dict[str, Any]:
    """Return configuration data from ``explicit_path`` or ``fallback`` if it exists."""

    candidate: Optional[Path] = None
    if explicit_path is not None:
        candidate = explicit_path.expanduser()
        if not candidate.is_file():
            raise ConfigurationError(f"Config file {candidate} does not exist")
    elif fallback.is_file():
        candidate = fallback

    if candidate is None:
        return {}

    data = _read_json(candidate)
    if not isinstance(data, dict):
        raise ConfigurationError(f"Config file {candidate} must contain a JSON object")
    return data


def resolve_jsbsim_root(cli_value: Optional[str], config: Dict[str, Any]) -> Path:
    root = cli_value or config.get("jsbsim_root") or os.environ.get("JSBSIM_ROOT")
    if not root:
        raise ConfigurationError(
            "JSBSim root not provided. Set JSBSIM_ROOT env, edit config/train.json, or pass --jsbsim-root."
        )

    root_path = Path(str(root)).expanduser()
    if not root_path.exists():
        raise ConfigurationError(
            f"JSBSim root {root_path} does not exist. Update your configuration to point to a valid JSBSim installation."
        )
    return root_path


def resolve_model_path(cli_value: Optional[Path], config: Dict[str, Any]) -> Path:
    model_value = cli_value or config.get("model")
    if not model_value:
        raise ConfigurationError(
            "Model path not provided. Update config/inference.json or pass --model when running inference."
        )

    model_path = Path(model_value).expanduser()
    if not model_path.exists():
        raise ConfigurationError(f"Model file {model_path} does not exist")
    return model_path


def resolve_device(cli_value: Optional[str], config: Dict[str, Any], default: str = "auto") -> str:
    """Resolve the torch device string for training or inference."""

    if cli_value:
        return cli_value

    if "device" in config and config["device"] not in (None, ""):
        return str(config["device"])

    env_value = os.environ.get("RL_WIND_UAV_DEVICE")
    if env_value:
        return env_value

    return default


def resolve_int(value: Optional[int], config: Dict[str, Any], key: str, default: int) -> int:
    if value is not None:
        return value
    if key in config:
        return int(config[key])
    return default


def resolve_bool(value: bool, config: Dict[str, Any], key: str) -> bool:
    if value:
        return True
    if key in config:
        return bool(config[key])
    return False


def resolve_path(
    value: Optional[Path], config: Dict[str, Any], key: str, default: Optional[Path]
) -> Optional[Path]:
    if value is not None:
        return value.expanduser()
    if key in config and config[key] not in (None, ""):
        return Path(str(config[key])).expanduser()
    return default.expanduser() if isinstance(default, Path) else default


def _ensure_section(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return a nested configuration section ensuring it is a dictionary."""

    section = config.get(key, {})
    if section in (None, ""):
        return {}
    if not isinstance(section, dict):
        raise ConfigurationError(f"Config section '{key}' must be a JSON object")
    return dict(section)


def gather_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Collect environment overrides from top-level and nested config entries."""

    overrides = _ensure_section(config, "env")

    for field in ROUTE_FIELD_NAMES:
        if field in config and field not in overrides:
            overrides[field] = config[field]

    # Backwards compatibility aliases
    if "flightgear" in config and "enable_flightgear" not in overrides:
        overrides["enable_flightgear"] = config["flightgear"]
    if "wind" in config and "wind_config" not in overrides:
        overrides["wind_config"] = config["wind"]

    return overrides


def build_wind_config(overrides: Dict[str, Any] | None) -> WindFieldConfig:
    """Create a :class:`WindFieldConfig` instance from configuration data."""

    if not overrides:
        return WindFieldConfig()

    kwargs: Dict[str, Any] = {}
    for field in fields(WindFieldConfig):
        if field.name in overrides:
            value = overrides[field.name]
            if field.name == "altitude_range" and isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                pair = tuple(value)
                if len(pair) != 2:
                    raise ConfigurationError("wind.altitude_range must contain exactly two values")
                value = (float(pair[0]), float(pair[1]))
            kwargs[field.name] = value
    return WindFieldConfig(**kwargs)


def build_route_config(jsbsim_root: Path, overrides: Dict[str, Any] | None) -> RouteConfig:
    """Create a :class:`RouteConfig` instance using overrides from the config file."""

    data = dict(overrides or {})
    route_kwargs: Dict[str, Any] = {"jsbsim_root": str(jsbsim_root)}

    wind_overrides = data.pop("wind", None)
    wind_overrides = data.pop("wind_config", wind_overrides)

    for field in ROUTE_FIELD_NAMES:
        if field in data:
            route_kwargs[field] = data[field]

    if "flightgear_path" in route_kwargs and route_kwargs["flightgear_path"] in ("", None):
        route_kwargs["flightgear_path"] = None

    if wind_overrides is not None:
        route_kwargs["wind_config"] = build_wind_config(wind_overrides)

    return RouteConfig(**route_kwargs)


def extract_ppo_hyperparameters(config: Dict[str, Any]) -> Tuple[str, int, Optional[str], Dict[str, Any]]:
    """Return PPO configuration split into policy, verbosity, tensorboard path, and kwargs."""

    ppo_section = _ensure_section(config, "ppo")
    if not ppo_section:
        return "MlpPolicy", 1, None, {}

    kwargs: Dict[str, Any] = {}
    for key in PPO_KNOWN_KEYS:
        if key in ppo_section:
            kwargs[key] = ppo_section[key]

    policy = str(kwargs.pop("policy", "MlpPolicy"))
    verbose = int(kwargs.pop("verbose", 1))
    tensorboard_log = kwargs.pop("tensorboard_log", None)

    return policy, verbose, tensorboard_log, kwargs

