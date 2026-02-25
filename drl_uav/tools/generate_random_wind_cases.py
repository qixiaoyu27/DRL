from __future__ import annotations

from pathlib import Path
import json
from drl_uav.envs.wind_field import WindField

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "wind_cases"
N_CASES = 100


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(N_CASES):
        wf = WindField.random_field(map_size_m=4000.0)
        (OUT_DIR / f"wind_case_{i:03d}.json").write_text(json.dumps(wf.to_json_dict(), indent=2), encoding="utf-8")
    print(f"Generated {N_CASES} wind files to {OUT_DIR}")


if __name__ == "__main__":
    main()
