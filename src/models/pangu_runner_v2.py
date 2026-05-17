"""Pangu v2 runner: class-first implementation with compatibility wrapper."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.common.data_reader_ifs import DEFAULT_ERA5_NPY_ROOT
from src.models.pangu_runner_class import DEFAULT_WEIGHTS_ROOT, PanguForecastRunner

_RUNNERS: dict[tuple[str, str], PanguForecastRunner] = {}


def _get_runner(era5_root: Path, weights_root: Path) -> PanguForecastRunner:
    key = (str(era5_root), str(weights_root))
    if key not in _RUNNERS:
        _RUNNERS[key] = PanguForecastRunner(era5_root=era5_root, weights_root=weights_root)
    return _RUNNERS[key]


def run_pangu_forecast_v2(
    init_time: datetime,
    lead_times_hours: List[int],
    *,
    era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
    weights_root: Path | None = None,
) -> Dict[int, np.ndarray]:
    runner = _get_runner(era5_root, weights_root or DEFAULT_WEIGHTS_ROOT)
    return runner.run(init_time, lead_times_hours)


class PanguForecastRunnerV2(PanguForecastRunner):
    """Alias class for explicit v2 usage."""

