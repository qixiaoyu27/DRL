"""Helper utilities for loading default training and inference configuration."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_TRAIN_CONFIG = CONFIG_DIR / "train.json"
DEFAULT_INFERENCE_CONFIG = CONFIG_DIR / "inference.json"


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

