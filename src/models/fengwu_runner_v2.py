"""FengWu v2 runner with class wrapper and cached normalization arrays."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.common.data_reader import DEFAULT_ERA5_NPY_ROOT
from src.models.fengwu_runner import (
    NORM_DIR,
    STEP_HOURS,
    run_fengwu_forecast,
)

logger = logging.getLogger(__name__)

_NORM_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def get_fengwu_norm_constants() -> tuple[np.ndarray, np.ndarray]:
    mean_path = NORM_DIR / "data_mean.npy"
    std_path = NORM_DIR / "data_std.npy"
    key = f"{mean_path}|{std_path}"
    if key in _NORM_CACHE:
        return _NORM_CACHE[key]
    data_mean = np.load(mean_path).astype(np.float32)[:, None, None]
    data_std = np.load(std_path).astype(np.float32)[:, None, None]
    _NORM_CACHE[key] = (data_mean, data_std)
    return _NORM_CACHE[key]


class FengwuForecastRunnerV2:
    """Class-style FengWu runner; keeps long-lived process semantics."""

    def __init__(
        self,
        *,
        era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
        onnx_name: str = "fengwu_v1.onnx",
        flip_north_south: bool = False,
    ) -> None:
        self.era5_root = era5_root
        self.onnx_name = onnx_name
        self.flip_north_south = flip_north_south

    def run(self, init_time: datetime, lead_times_hours: List[int]) -> Dict[int, np.ndarray]:
        # Pre-warm normalization constants once per process.
        get_fengwu_norm_constants()
        t0 = time.perf_counter()
        out = run_fengwu_forecast(
            init_time,
            lead_times_hours,
            era5_root=self.era5_root,
            onnx_name=self.onnx_name,
            flip_north_south=self.flip_north_south,
        )
        logger.info(
            "FengWu(v2) init=%s leads=%d step_h=%d total_s=%.3f",
            init_time.strftime("%Y%m%d%H"),
            len(lead_times_hours),
            STEP_HOURS,
            time.perf_counter() - t0,
        )
        return out

