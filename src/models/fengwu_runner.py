"""FengWu ERA5 (v1) runner: np.25 inputs, ONNX autoregression."""  # src:/home/NWP-Benchmark/src/fengwu/inference.py:1-20

from __future__ import annotations  # src:/home/NWP-Benchmark/src/fengwu/inference.py:1 style

import logging  # src:/home/NWP-Benchmark/src/fengwu/inference.py:4
import os  # src:/home/NWP-Benchmark/src/fengwu/inference.py:1
import time
from datetime import datetime  # src:/home/NWP-Benchmark/src/fengwu/inference.py
from pathlib import Path  # src:/home/NWP-Benchmark/src/fengwu/inference.py:6
from typing import Dict, List  # src:/home/NWP-Benchmark/src/fengwu/inference.py:7

import numpy as np  # src:/home/NWP-Benchmark/src/fengwu/inference.py:10
import onnxruntime as ort  # src:/home/NWP-Benchmark/src/fengwu/inference.py:10

from src.common.data_reader import (  # src:/vepfs-dev/.../data_reader.py — same stack as prepare channel order
    DEFAULT_ERA5_NPY_ROOT,
    load_fengwu_init_pair,
)

logger = logging.getLogger(__name__)  # src:/home/NWP-Benchmark/src/fengwu/inference.py:29

# src:/home/NWP-Benchmark/src/fengwu/inference.py:31-38
STEP_HOURS = 6  # src:/home/NWP-Benchmark/src/fengwu/inference.py:32
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]  # src:/home/NWP-Benchmark/src/fengwu/inference.py:35
LAT_RES = 721  # src:/home/NWP-Benchmark/src/fengwu/inference.py:37
LON_RES = 1440  # src:/home/NWP-Benchmark/src/fengwu/inference.py:38

NORM_DIR = Path(__file__).resolve().parents[1] / "fengwu" / "normalization_constants"  # src:/home/NWP-Benchmark/src/fengwu/inference.py:111-112 (CURRENT_DIR relative)
DEFAULT_WEIGHTS_ROOT = Path(os.environ.get("NWP_WEIGHTS_ROOT", "/ecmwf-era5-datasets/nwp_bench/assets/weights"))  # src:/home/NWP-Benchmark/src/fengwu/inference.py:20 adapted

_SESSION_CACHE: dict[str, ort.InferenceSession] = {}  # src:/optional reuse to avoid reloading ONNX in batch jobs
_NORM_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _get_session(model_path: Path) -> ort.InferenceSession:  # src:/home/NWP-Benchmark/src/fengwu/inference.py:138-152
    key = str(model_path)  # src:/cache key
    if key in _SESSION_CACHE:  # src:/cache hit
        return _SESSION_CACHE[key]  # src:/return cached
    sess_options = ort.SessionOptions()  # src:/home/NWP-Benchmark/src/fengwu/inference.py:139
    sess_options.enable_cpu_mem_arena = False  # src:/home/NWP-Benchmark/src/fengwu/inference.py:140
    sess_options.enable_mem_pattern = False  # src:/home/NWP-Benchmark/src/fengwu/inference.py:141
    sess_options.enable_mem_reuse = False  # src:/home/NWP-Benchmark/src/fengwu/inference.py:142
    sess_options.intra_op_num_threads = 1  # src:/home/NWP-Benchmark/src/fengwu/inference.py:143
    cuda_provider_options = {"arena_extend_strategy": "kSameAsRequested"}  # src:/home/NWP-Benchmark/src/fengwu/inference.py:145
    try:  # src:/home/NWP-Benchmark/src/fengwu/inference.py:147
        providers = [("CUDAExecutionProvider", cuda_provider_options), "CPUExecutionProvider"]  # src:/home/NWP-Benchmark/src/fengwu/inference.py:148
        session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)  # src:/home/NWP-Benchmark/src/fengwu/inference.py:149
    except Exception:  # src:/home/NWP-Benchmark/src/fengwu/inference.py:150
        logger.warning("CUDAExecutionProvider unavailable; falling back to CPU.")  # src:/home/NWP-Benchmark/src/fengwu/inference.py:151
        session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=["CPUExecutionProvider"])  # src:/home/NWP-Benchmark/src/fengwu/inference.py:152
    _SESSION_CACHE[key] = session  # src:/store cache
    return session  # src:/return


def _get_norm(mean_path: Path, std_path: Path) -> tuple[np.ndarray, np.ndarray]:
    key = f"{mean_path}|{std_path}"
    if key in _NORM_CACHE:
        return _NORM_CACHE[key]
    data_mean = np.load(mean_path).astype(np.float32)[:, None, None]
    data_std = np.load(std_path).astype(np.float32)[:, None, None]
    _NORM_CACHE[key] = (data_mean, data_std)
    return _NORM_CACHE[key]


def run_fengwu_forecast(  # src:/home/NWP-Benchmark/src/fengwu/inference.py:89-183 adapted for API + np.25
    init_time: datetime,
    lead_times_hours: List[int],
    *,
    era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
    onnx_name: str = "fengwu_v1.onnx",
    flip_north_south: bool = False,
) -> Dict[int, np.ndarray]:
    """ERA5-trained FengWu (v1 ONNX). Uses same 69-channel layout as ``prepare.py`` / ``load_fengwu_init_pair``."""
    wanted = sorted({int(h) for h in lead_times_hours})  # src:/vepfs contract
    if not wanted:  # src:/empty
        return {}  # src:/empty
    if any(h % STEP_HOURS != 0 for h in wanted):  # src:/home/NWP-Benchmark/src/fengwu/inference.py:97-98
        raise ValueError(f"FengWu lead times must be multiples of {STEP_HOURS}h: {wanted}")  # src:/home/NWP-Benchmark/src/fengwu/inference.py:98
    if "v2" in onnx_name.lower():  # src:/home prepare: hres -> fengwu_v2; ERA5 -> fengwu_v1
        raise ValueError(
            "This runner targets FengWu v1 (ERA5). For v2 / HRES inputs use the separate operational pipeline."
        )

    model_path = DEFAULT_WEIGHTS_ROOT / "fengwu" / onnx_name  # src:/home/NWP-Benchmark/src/fengwu/inference.py:107
    if not model_path.exists():  # src:/home/NWP-Benchmark/src/fengwu/inference.py:108-109
        raise FileNotFoundError(f"Model file missing: {model_path}")  # src:/home/NWP-Benchmark/src/fengwu/inference.py:109

    mean_path = NORM_DIR / "data_mean.npy"  # src:/home/NWP-Benchmark/src/fengwu/inference.py:111
    std_path = NORM_DIR / "data_std.npy"  # src:/home/NWP-Benchmark/src/fengwu/inference.py:112
    if not mean_path.exists() or not std_path.exists():  # src:/home/NWP-Benchmark/src/fengwu/inference.py:113-114
        raise FileNotFoundError(f"Missing mean/std: {mean_path}, {std_path}")  # src:/home/NWP-Benchmark/src/fengwu/inference.py:114

    data_mean, data_std = _get_norm(mean_path, std_path)

    # src:/home/NWP-Benchmark/src/fengwu/prepare.py:291-295 input1=T-6h, input2=T0 from same ERA5 stacks
    input1, input2 = load_fengwu_init_pair(init_time, root=era5_root, flip_north_south=flip_north_south)  # src:/vepfs data_reader mirrors prepare order
    if input1.shape != (69, LAT_RES, LON_RES) or input2.shape != (69, LAT_RES, LON_RES):  # src:/home/NWP-Benchmark/src/fengwu/inference.py:131-132
        raise ValueError(f"Expected inputs (69,{LAT_RES},{LON_RES}), got {input1.shape}, {input2.shape}")  # src:/home/NWP-Benchmark/src/fengwu/inference.py:132

    input1n = (input1 - data_mean) / data_std  # src:/home/NWP-Benchmark/src/fengwu/inference.py:134
    input2n = (input2 - data_mean) / data_std  # src:/home/NWP-Benchmark/src/fengwu/inference.py:135
    curr = np.concatenate([input1n, input2n], axis=0)[None, ...].astype(np.float32)  # src:/home/NWP-Benchmark/src/fengwu/inference.py:136

    session = _get_session(model_path)  # src:/home/NWP-Benchmark/src/fengwu/inference.py:149

    max_steps = max(wanted) // STEP_HOURS  # src:/home/NWP-Benchmark/src/fengwu/inference.py:100 + loop extent
    out: Dict[int, np.ndarray] = {}  # src:/vepfs multi-lead output
    for i in range(max_steps):  # src:/home/NWP-Benchmark/src/fengwu/inference.py:163
        forecast_hour = (i + 1) * STEP_HOURS  # src:/home/NWP-Benchmark/src/fengwu/inference.py:164
        out_arr = session.run(None, {"input": curr})[0]  # src:/home/NWP-Benchmark/src/fengwu/inference.py:167
        curr = np.concatenate((curr[:, 69:], out_arr[:, :69]), axis=1).astype(np.float32)  # src:/home/NWP-Benchmark/src/fengwu/inference.py:169
        pred_norm = out_arr[0, :69]  # src:/home/NWP-Benchmark/src/fengwu/inference.py:171
        pred = (pred_norm * data_std) + data_mean  # src:/home/NWP-Benchmark/src/fengwu/inference.py:172
        if forecast_hour in wanted:  # src:/only store requested leads
            out[forecast_hour] = pred.astype(np.float32)  # src:/home/NWP-Benchmark/src/fengwu/inference.py:174-172 single step tensor
    return out  # src:/return mapping lead -> (69,721,1440)


class FengwuForecastRunner:
    """Class-style FengWu runner with cached ONNX session and normalization arrays."""

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
        t0 = time.perf_counter()
        out = run_fengwu_forecast(
            init_time,
            lead_times_hours,
            era5_root=self.era5_root,
            onnx_name=self.onnx_name,
            flip_north_south=self.flip_north_south,
        )
        logger.info(
            "FengWu(class) init=%s leads=%d total_s=%.3f",
            init_time.strftime("%Y%m%d%H"),
            len(lead_times_hours),
            time.perf_counter() - t0,
        )
        return out
