"""Compatibility wrapper around class-based Pangu runner."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.common.data_reader import DEFAULT_ERA5_NPY_ROOT
from src.models.pangu_runner_class import (
    DEFAULT_WEIGHTS_ROOT,
    PanguForecastRunner,
)

_RUNNER_CACHE: dict[Tuple[str, str], PanguForecastRunner] = {}


def _runner(era5_root: Path, weights_root: Path) -> PanguForecastRunner:
    key = (str(era5_root), str(weights_root))
    if key not in _RUNNER_CACHE:
        _RUNNER_CACHE[key] = PanguForecastRunner(
            era5_root=era5_root,
            weights_root=weights_root,
        )
    return _RUNNER_CACHE[key]


def run_pangu_forecast(
    init_time: datetime,
    lead_times_hours: List[int],
    *,
    era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
    weights_root: Path | None = None,
) -> Dict[int, np.ndarray]:
    """Function API kept for compatibility; delegates to class implementation."""
    runner = _runner(era5_root=era5_root, weights_root=weights_root or DEFAULT_WEIGHTS_ROOT)
    return runner.run(init_time, lead_times_hours)
