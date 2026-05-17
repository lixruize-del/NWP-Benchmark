"""GraphCast v2 runner with process-local bundle/predictor cache."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import jax
import numpy as np

from src.common.data_reader import DEFAULT_ERA5_NPY_ROOT
from src.models import graphcast_runner as base

_MODEL_BUNDLE_CACHE: dict[str, tuple] = {}
_PREDICTOR_CACHE: dict[str, object] = {}


def _get_model_bundle(weights_root: Path):
    key = str(weights_root)
    if key not in _MODEL_BUNDLE_CACHE:
        _MODEL_BUNDLE_CACHE[key] = base._load_model(weights_root)
    return _MODEL_BUNDLE_CACHE[key]


def _get_predictor(weights_root: Path):
    key = str(weights_root)
    if key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[key]
    (
        params,
        state,
        model_config,
        task_config,
        diffs_stddev,
        mean_by_level,
        stddev_by_level,
    ) = _get_model_bundle(weights_root)
    _PREDICTOR_CACHE[key] = base._build_predictor(
        params,
        state,
        model_config,
        task_config,
        diffs_stddev,
        mean_by_level,
        stddev_by_level,
    )
    return _PREDICTOR_CACHE[key]


def run_graphcast_forecast_v2(
    init_time: datetime,
    lead_times_hours: List[int],
    *,
    era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
) -> Dict[int, np.ndarray]:
    wanted = sorted({int(h) for h in lead_times_hours})
    if not wanted:
        return {}
    if any(h % base.STEP_HOURS != 0 for h in wanted):
        raise ValueError(f"lead_time_hours must be multiples of {base.STEP_HOURS}: {wanted}")

    steps = max(wanted) // base.STEP_HOURS
    (
        _params,
        _state,
        _model_config,
        task_config,
        _diffs_stddev,
        _mean_by_level,
        _stddev_by_level,
    ) = _get_model_bundle(base.DEFAULT_WEIGHTS_ROOT)
    ds_long = base._build_np25_input(init_time, era5_root)
    inputs, forcings, targets_template = base._prepare_forcings_and_targets(
        ds_long, init_time, steps, task_config
    )
    run_forward_jitted = _get_predictor(base.DEFAULT_WEIGHTS_ROOT)
    predictions = base.rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
    )

    out: Dict[int, np.ndarray] = {}
    for step_idx in range(steps):
        lead_hour = (step_idx + 1) * base.STEP_HOURS
        if lead_hour in wanted:
            step_ds = predictions.isel(time=step_idx, drop=False)
            out[lead_hour] = base._pack_output(step_ds).astype(np.float32)
    return out


class GraphcastForecastRunnerV2:
    def __init__(self, *, era5_root: Path = DEFAULT_ERA5_NPY_ROOT) -> None:
        self.era5_root = era5_root

    def run(self, init_time: datetime, lead_times_hours: List[int]) -> Dict[int, np.ndarray]:
        return run_graphcast_forecast_v2(
            init_time,
            lead_times_hours,
            era5_root=self.era5_root,
        )

