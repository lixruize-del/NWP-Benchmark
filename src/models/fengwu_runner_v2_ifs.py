"""FengWu v2 runner with class wrapper and cached normalization arrays."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort

from src.common.data_reader_ifs import DEFAULT_ERA5_NPY_ROOT, load_fengwu_init_pair
from src.models.fengwu_runner import (
    DEFAULT_WEIGHTS_ROOT,
    LAT_RES,
    LON_RES,
    NORM_DIR,
    STEP_HOURS,
)

logger = logging.getLogger(__name__)

_NORM_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_SESSION_CACHE: dict[str, ort.InferenceSession] = {}


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


def _get_session(model_path: Path) -> ort.InferenceSession:
    key = str(model_path)
    if key in _SESSION_CACHE:
        return _SESSION_CACHE[key]
    sess_options = ort.SessionOptions()
    sess_options.enable_cpu_mem_arena = False
    sess_options.enable_mem_pattern = False
    sess_options.enable_mem_reuse = False
    sess_options.intra_op_num_threads = 1
    cuda_provider_options = {"arena_extend_strategy": "kSameAsRequested"}
    try:
        providers = [("CUDAExecutionProvider", cuda_provider_options), "CPUExecutionProvider"]
        session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
    except Exception:
        logger.warning("CUDAExecutionProvider unavailable; falling back to CPUExecutionProvider.")
        session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=["CPUExecutionProvider"])
    _SESSION_CACHE[key] = session
    return session


def run_fengwu_forecast_v2_ifs(
    init_time: datetime,
    lead_times_hours: List[int],
    *,
    era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
    onnx_name: str = "fengwu_v2.onnx",
    flip_north_south: bool = False,
) -> Dict[int, np.ndarray]:
    wanted = sorted({int(h) for h in lead_times_hours})
    if not wanted:
        return {}
    if any(h % STEP_HOURS != 0 for h in wanted):
        raise ValueError(f"FengWu lead times must be multiples of {STEP_HOURS}h: {wanted}")

    model_path = DEFAULT_WEIGHTS_ROOT / "fengwu" / onnx_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file missing: {model_path}")

    data_mean, data_std = get_fengwu_norm_constants()
    input1, input2 = load_fengwu_init_pair(
        init_time,
        root=era5_root,
        flip_north_south=flip_north_south,
    )
    if input1.shape != (69, LAT_RES, LON_RES) or input2.shape != (69, LAT_RES, LON_RES):
        raise ValueError(f"Expected inputs (69,{LAT_RES},{LON_RES}), got {input1.shape}, {input2.shape}")

    input1n = (input1 - data_mean) / data_std
    input2n = (input2 - data_mean) / data_std
    curr = np.concatenate([input1n, input2n], axis=0)[None, ...].astype(np.float32)

    session = _get_session(model_path)
    max_steps = max(wanted) // STEP_HOURS
    out: Dict[int, np.ndarray] = {}
    for i in range(max_steps):
        forecast_hour = (i + 1) * STEP_HOURS
        out_arr = session.run(None, {"input": curr})[0]
        curr = np.concatenate((curr[:, 69:], out_arr[:, :69]), axis=1).astype(np.float32)
        pred_norm = out_arr[0, :69]
        pred = (pred_norm * data_std) + data_mean
        if forecast_hour in wanted:
            out[forecast_hour] = pred.astype(np.float32)
    return out


class FengwuForecastRunnerV2:
    """Class-style FengWu runner; keeps long-lived process semantics."""

    def __init__(
        self,
        *,
        era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
        onnx_name: str = "fengwu_v2.onnx",
        flip_north_south: bool = False,
    ) -> None:
        self.era5_root = era5_root
        self.onnx_name = onnx_name
        self.flip_north_south = flip_north_south

    def run(self, init_time: datetime, lead_times_hours: List[int]) -> Dict[int, np.ndarray]:
        t0 = time.perf_counter()
        out = run_fengwu_forecast_v2_ifs(
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

