"""Generate wind heatmap and trajectory using a trained policy."""
from __future__ import annotations

import pathlib
from typing import Optional

from fixedwing_rl.utils import load_training_config
from fixedwing_rl.utils.plotting import save_wind_heatmap
from train import (
    ALGOS,
    create_wind_field,
    default_model_path,
    evaluate,
    make_env,
    resolve_config_path,
)


MODEL_OVERRIDE: Optional[str] = None


def main(model_path: str | pathlib.Path | None = None, config_path: str | pathlib.Path | None = None) -> None:
    config = load_training_config(resolve_config_path(config_path))

    algo_key = config.algorithm.name.lower()
    if algo_key not in ALGOS:
        raise ValueError(
            f"Unsupported algorithm '{config.algorithm.name}'. Expected one of {sorted(ALGOS)}."
        )
    algo_cls = ALGOS[algo_key]

    wind_field = create_wind_field(config)

    env = make_env(config, wind_field, monitor=False)
    chosen_path = model_path or MODEL_OVERRIDE
    model_path = pathlib.Path(chosen_path) if chosen_path else default_model_path(config)
    model = algo_cls.load(model_path, device=config.algorithm.device)

    lats, lons, winds, trajectory = evaluate(model, env, config.evaluation.steps)
    plots_dir = pathlib.Path(config.output.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_wind_heatmap(lats, lons, winds, trajectory, plots_dir / "wind_trajectory.png")

    env.close()


if __name__ == "__main__":
    main()
