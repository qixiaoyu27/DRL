"""Top-level package exports with lazy imports to avoid heavy dependencies at import time."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__all__ = [
    "ConsoleLogCallback",
    "FixedWingRouteEnv",
    "RouteConfig",
    "WindField",
    "WindFieldConfig",
]

_IMPORT_STRUCTURE = {
    "ConsoleLogCallback": ("rl_wind_uav.callbacks", "ConsoleLogCallback"),
    "FixedWingRouteEnv": ("rl_wind_uav.env.fixed_wing_route_env", "FixedWingRouteEnv"),
    "RouteConfig": ("rl_wind_uav.env.fixed_wing_route_env", "RouteConfig"),
    "WindField": ("rl_wind_uav.env.wind_field", "WindField"),
    "WindFieldConfig": ("rl_wind_uav.env.wind_field", "WindFieldConfig"),
}


def __getattr__(name: str) -> Any:
    if name in _IMPORT_STRUCTURE:
        module_name, attr_name = _IMPORT_STRUCTURE[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'rl_wind_uav' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_IMPORT_STRUCTURE.keys()))


if TYPE_CHECKING:  # pragma: no cover - used only for static analyzers
    from .callbacks import ConsoleLogCallback
    from .env.fixed_wing_route_env import FixedWingRouteEnv, RouteConfig
    from .env.wind_field import WindField, WindFieldConfig
