from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from drl_uav import config


def generate_scenarios_json(path: Path = config.SCENARIO_JSON, n_scenarios: int = config.SCENARIO_COUNT, seed: int = 2025) -> Path:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    scenarios = []
    margin = 400.0
    for idx in range(n_scenarios):
        speed = float(rng.uniform(0.0, config.MAX_BACKGROUND_WIND_MPS))
        direction = float(rng.uniform(0.0, 2.0 * np.pi))
        vortices = []
        for _ in range(config.NUM_VORTICES):
            vortices.append(
                {
                    "center": [float(v) for v in rng.uniform(margin, config.MAP_SIZE_M - margin, size=2)],
                    "circulation": float(rng.uniform(-2400.0, 2400.0)),
                    "core_radius": float(rng.uniform(80.0, 240.0)),
                }
            )

        scenarios.append(
            {
                "id": idx,
                "map_size": config.MAP_SIZE_M,
                "background_wind": {"speed": speed, "direction": direction},
                "vortices": vortices,
            }
        )

    payload = {
        "description": "100 fixed wind-field scenarios for JSBSim-based UAV coverage RL",
        "count": n_scenarios,
        "scenarios": scenarios,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_scenarios(path: Path = config.SCENARIO_JSON) -> list[dict]:
    if not path.exists():
        generate_scenarios_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["scenarios"]
