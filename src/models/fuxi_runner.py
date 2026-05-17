"""FuXi runner: np.25 → ONNX cascade (short/medium/long), 6h steps."""  # src:/home/NWP-Benchmark/src/fuxi/inference.py:1-40

from __future__ import annotations  # src:/home/NWP-Benchmark/src/fuxi/inference.py style

import logging  # src:/home/NWP-Benchmark/src/fuxi/inference.py:4
import os  # src:/home/NWP-Benchmark/src/fuxi/inference.py:1
import time  # wall timing for diagnostics
from datetime import datetime  # src:/home/NWP-Benchmark/src/fuxi/inference.py time_encoding
from pathlib import Path  # src:/home/NWP-Benchmark/src/fuxi/inference.py:6
from typing import Dict, List, Tuple  # src:/home/NWP-Benchmark/src/fuxi/inference.py typing

import numpy as np  # src:/home/NWP-Benchmark/src/fuxi/inference.py:8
import onnxruntime as ort  # src:/home/NWP-Benchmark/src/fuxi/inference.py:11
import pandas as pd  # src:/home/NWP-Benchmark/src/fuxi/inference.py:10

from src.common.data_reader import (  # src:/vepfs-dev/.../data_reader.py
    DEFAULT_ERA5_NPY_ROOT,
    fuxi_channel_names,
    load_fuxi_input_tensor,
)

logger = logging.getLogger(__name__)  # src:/home/NWP-Benchmark/src/fuxi/inference.py:34

STEP_HOURS = 6  # src:/home/NWP-Benchmark/src/fuxi/inference.py:40
STAGES = ("short", "medium", "long")  # src:/home/NWP-Benchmark/src/fuxi/inference.py:164-165
DEFAULT_WEIGHTS_ROOT = Path(os.environ.get("NWP_WEIGHTS_ROOT", "/ecmwf-era5-datasets/nwp_bench/assets/weights"))  # src:/home/NWP-Benchmark/src/fuxi/inference.py:21 adapted

_SESSIONS: dict[str, ort.InferenceSession] = {}  # src:/cache ONNX sessions


def _ort_intra_threads() -> int:
    raw = os.environ.get("FUXI_ORT_INTRA_OP_THREADS", "4").strip()
    try:
        return max(1, min(int(raw), 64))
    except ValueError:
        return 4


def _ort_inter_threads() -> int:
    raw = os.environ.get("FUXI_ORT_INTER_OP_THREADS", "1").strip()
    try:
        return max(1, min(int(raw), 64))
    except ValueError:
        return 1


def _time_encoding(init_time: pd.Timestamp, total_step: int, freq: int = 6) -> np.ndarray:  # src:/home/NWP-Benchmark/src/fuxi/inference.py:88-101
    init_time = np.array([init_time])  # src:/home/NWP-Benchmark/src/fuxi/inference.py:90
    tembs: List[np.ndarray] = []  # src:/home/NWP-Benchmark/src/fuxi/inference.py:91
    for i in range(total_step):  # src:/home/NWP-Benchmark/src/fuxi/inference.py:92
        hours = np.array([pd.Timedelta(hours=t * freq) for t in [i - 1, i, i + 1]])  # src:/home/NWP-Benchmark/src/fuxi/inference.py:93
        times = init_time[:, None] + hours[None]  # src:/home/NWP-Benchmark/src/fuxi/inference.py:94
        times = [pd.Period(t, "h") for t in times.reshape(-1)]  # src:/home/NWP-Benchmark/src/fuxi/inference.py:95
        times = [(p.day_of_year / 366, p.hour / 24) for p in times]  # src:/home/NWP-Benchmark/src/fuxi/inference.py:96
        temb = np.array(times, dtype=np.float32)  # src:/home/NWP-Benchmark/src/fuxi/inference.py:97
        temb = np.concatenate([np.sin(temb), np.cos(temb)], axis=-1)  # src:/home/NWP-Benchmark/src/fuxi/inference.py:98
        temb = temb.reshape(1, -1)  # src:/home/NWP-Benchmark/src/fuxi/inference.py:99 shape (1,6)
        tembs.append(temb)  # src:/home/NWP-Benchmark/src/fuxi/inference.py:100
    return np.stack(tembs)  # src:/home/NWP-Benchmark/src/fuxi/inference.py:101 shape (total_step,1,6)


def _session(model_path: Path) -> ort.InferenceSession:  # src:/home/NWP-Benchmark/src/fuxi/inference.py:104-119
    key = str(model_path)  # src:/cache key
    if key in _SESSIONS:  # src:/hit
        return _SESSIONS[key]  # src:/return
    options = ort.SessionOptions()  # src:/home/NWP-Benchmark/src/fuxi/inference.py:106
    options.enable_cpu_mem_arena = False  # src:/home/NWP-Benchmark/src/fuxi/inference.py:107
    options.enable_mem_pattern = False  # src:/home/NWP-Benchmark/src/fuxi/inference.py:108
    options.enable_mem_reuse = False  # src:/home/NWP-Benchmark/src/fuxi/inference.py:109
    options.intra_op_num_threads = _ort_intra_threads()
    options.inter_op_num_threads = _ort_inter_threads()
    cuda_options = {"arena_extend_strategy": "kSameAsRequested"}  # src:/home/NWP-Benchmark/src/fuxi/inference.py:112
    providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]  # src:/home/NWP-Benchmark/src/fuxi/inference.py:113
    try:  # src:/home/NWP-Benchmark/src/fuxi/inference.py:114
        sess = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)  # src:/home/NWP-Benchmark/src/fuxi/inference.py:115
    except Exception:  # src:/home/NWP-Benchmark/src/fuxi/inference.py:116
        logger.warning("FuXi CUDA unavailable; using CPU.")  # src:/home/NWP-Benchmark/src/fuxi/inference.py:117
        sess = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])  # src:/home/NWP-Benchmark/src/fuxi/inference.py:118
    provs = sess.get_providers()
    primary = provs[0] if provs else "?"
    if primary == "CPUExecutionProvider":
        logger.warning(
            "FuXi ORT %s: primary EP is CPU (very slow). providers=%s intra_threads=%d",
            model_path.name,
            provs,
            options.intra_op_num_threads,
        )
    else:
        logger.info(
            "FuXi ORT %s: providers=%s intra_threads=%d inter_threads=%d",
            model_path.name,
            provs,
            options.intra_op_num_threads,
            options.inter_op_num_threads,
        )
    _SESSIONS[key] = sess  # src:/store
    return sess  # src:/return


def _default_stage_steps(total_steps: int) -> Tuple[int, int, int]:  # src:/home/NWP-Benchmark/src/fuxi/inference.py:177-207 allocation
    raw = os.environ.get("FUXI_STAGE_STEPS", "").strip()  # src:/optional override e.g. 20,20,20
    if raw:  # src:/user-specified
        parts = [int(x) for x in raw.split(",")]  # src:/parse
        if len(parts) != 3:  # src:/validate
            raise ValueError("FUXI_STAGE_STEPS must be three integers: short,medium,long")  # src:/msg
        a, b, c = parts  # src:/unpack
        if a + b + c != total_steps:  # src:/sum check
            raise ValueError(f"FUXI_STAGE_STEPS {a},{b},{c} sums to {a+b+c}, expected {total_steps}")  # src:/msg
        return a, b, c  # src:/return
    if total_steps <= 20:  # src:/short-only horizon
        return total_steps, 0, 0  # src:/e.g. 6h -> 1,0,0
    if total_steps <= 40:  # src:/short+medium
        return 20, total_steps - 20, 0  # src:/split
    return 20, 20, total_steps - 40  # src:/all three; long may exceed 20 — user should set FUXI_STAGE_STEPS for very long runs


def run_fuxi_forecast(  # src:/home/NWP-Benchmark/src/fuxi/inference.py:141-212 adapted API
    init_time: datetime,
    lead_times_hours: List[int],
    *,
    era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
    flip_north_south: bool = False,
) -> Dict[int, np.ndarray]:
    """Return denormalized FuXi fields ``(70,721,1440)`` per lead (multiples of 6h)."""
    t_run0 = time.perf_counter()
    wanted = sorted({int(h) for h in lead_times_hours})  # src:/vepfs contract
    if not wanted:  # src:/empty
        return {}  # src:/empty
    if any(h % STEP_HOURS != 0 for h in wanted):  # src:/validate
        raise ValueError(f"FuXi lead times must be multiples of {STEP_HOURS}h: {wanted}")  # src:/msg

    total_steps = max(wanted) // STEP_HOURS  # src:/home/NWP-Benchmark/src/fuxi/inference.py:156 sum(num_steps)
    num_steps = _default_stage_steps(total_steps)  # src:/[short,medium,long] counts
    if sum(num_steps) != total_steps:  # src:/sanity
        raise RuntimeError(f"Stage steps {num_steps} do not sum to {total_steps}")  # src:/should not happen

    init_ts = pd.Timestamp(init_time)  # src:/home/NWP-Benchmark/src/fuxi/inference.py:144 pandas init
    tembs = _time_encoding(init_ts, total_steps)  # src:/home/NWP-Benchmark/src/fuxi/inference.py:157

    t0 = time.perf_counter()
    input_np, _ch = load_fuxi_input_tensor(  # src:/home/NWP-Benchmark/src/fuxi/inference.py:160 → (1,2,70,H,W)
        init_time,
        root=era5_root,
        flip_north_south=flip_north_south,
    )
    load_input_s = time.perf_counter() - t0
    if input_np.shape != (1, 2, 70, 721, 1440):  # src:/home/NWP-Benchmark/src/fuxi/inference.py:132
        raise ValueError(f"FuXi input shape {input_np.shape}")  # src:/msg

    weight_dir = DEFAULT_WEIGHTS_ROOT / "fuxi"  # src:/home/NWP-Benchmark/src/fuxi/inference.py:167
    models: Dict[str, ort.InferenceSession] = {}  # src:/home/NWP-Benchmark/src/fuxi/inference.py:165-171
    t1 = time.perf_counter()
    for stage in STAGES:  # src:/home/NWP-Benchmark/src/fuxi/inference.py:166
        path = weight_dir / f"{stage}.onnx"  # src:/home/NWP-Benchmark/src/fuxi/inference.py:167
        if not path.exists():  # src:/home/NWP-Benchmark/src/fuxi/inference.py:168-169
            raise FileNotFoundError(f"FuXi model missing: {path}")  # src:/home/NWP-Benchmark/src/fuxi/inference.py:169
        models[stage] = _session(path)  # src:/home/NWP-Benchmark/src/fuxi/inference.py:171
    load_models_s = time.perf_counter() - t1

    out: Dict[int, np.ndarray] = {}  # src:/lead -> array
    step = 0  # src:/home/NWP-Benchmark/src/fuxi/inference.py:176 global step index
    t2 = time.perf_counter()
    ort_run_s = 0.0
    for stage_idx, n_sub in enumerate(num_steps):  # src:/home/NWP-Benchmark/src/fuxi/inference.py:177
        if n_sub == 0:  # src:/skip empty stage
            continue  # src:/next
        stage = STAGES[stage_idx]  # src:/home/NWP-Benchmark/src/fuxi/inference.py:178
        session = models[stage]  # src:/home/NWP-Benchmark/src/fuxi/inference.py:179
        for _ in range(n_sub):  # src:/home/NWP-Benchmark/src/fuxi/inference.py:184
            temb = tembs[step]  # src:/(1, 6); np.stack(tembs) -> (total_step, 1, 6)
            tr = time.perf_counter()
            new_input = session.run(None, {"input": input_np, "temb": temb})[0]  # src:/home/NWP-Benchmark/src/fuxi/inference.py:187
            ort_run_s += time.perf_counter() - tr
            output = new_input[0, -1].astype(np.float32)  # src:/home/NWP-Benchmark/src/fuxi/inference.py:189 (70,H,W)
            step += 1  # src:/home/NWP-Benchmark/src/fuxi/inference.py:207
            lead_h = step * STEP_HOURS  # src:/home/NWP-Benchmark/src/fuxi/inference.py:191
            if lead_h in wanted:  # src:/store only requested
                out[lead_h] = output  # src:/home/NWP-Benchmark/src/fuxi/inference.py:196-203 payload
            input_np = new_input  # src:/home/NWP-Benchmark/src/fuxi/inference.py:206
    loop_overhead_s = (time.perf_counter() - t2) - ort_run_s
    total_s = time.perf_counter() - t_run0
    init_label = init_time.strftime("%Y%m%d%H") if hasattr(init_time, "strftime") else str(init_time)
    logger.info(
        "FuXi timing init=%s total_s=%.3f load_npy_s=%.3f load_ort_sessions_s=%.3f "
        "ort_session_run_sum_s=%.3f loop_other_s=%.3f steps=%d wanted_leads=%d",
        init_label,
        total_s,
        load_input_s,
        load_models_s,
        ort_run_s,
        max(0.0, loop_overhead_s),
        step,
        len(wanted),
    )
    return out  # src:/done


class FuxiForecastRunner:
    """Class-style FuXi runner; reuses module-level ONNX session cache."""

    def __init__(
        self,
        *,
        era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
        flip_north_south: bool = False,
    ) -> None:
        self.era5_root = era5_root
        self.flip_north_south = flip_north_south

    def run(self, init_time: datetime, lead_times_hours: List[int]) -> Dict[int, np.ndarray]:
        return run_fuxi_forecast(
            init_time,
            lead_times_hours,
            era5_root=self.era5_root,
            flip_north_south=self.flip_north_south,
        )


# re-export for callers that import from runner  # src:/convenience
__all__ = [
    "run_fuxi_forecast",
    "FuxiForecastRunner",
    "fuxi_channel_names",
    "STEP_HOURS",
]  # src:/public API
