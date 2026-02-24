from __future__ import annotations

from pathlib import Path

# IDE-friendly hardcoded settings.
MAP_SIZE_M = 4000.0
FLIGHT_ALTITUDE_M = 100.0
CAMERA_FOV_DEG = 120.0
CRUISE_SPEED_MPS = 25.0
MAX_BACKGROUND_WIND_MPS = 5.4
NUM_VORTICES = 4
EPISODE_MAX_STEPS = 1400
DT = 1.0
MAX_TURN_RATE_DEG_S = 20.0

TRAIN_TIMESTEPS = 120_000
TRAIN_SEED = 42
CHECKPOINT_STEPS = 10_000
LOG_DIR = Path("outputs")
SCENARIO_JSON = Path("outputs/scenarios_100.json")
SCENARIO_COUNT = 100

# JSBSim settings (hardcoded, editable in IDE).
# If None, code auto-discovers from package path and common locations.
JSBSIM_ROOT_DIR: str | None = None
JSBSIM_AIRCRAFT_DIR: str | None = None
JSBSIM_ENGINE_DIR: str | None = None
JSBSIM_SYSTEMS_DIR: str | None = None

# Preferred model and fallbacks; code will try these in order.
JSBSIM_MODEL_CANDIDATES = ["c172p", "c172x", "f16"]
