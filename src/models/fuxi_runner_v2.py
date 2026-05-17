"""FuXi v2 runner with faster ONNX session initialization controls."""

from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort

from src.common.data_reader_ifs import DEFAULT_ERA5_NPY_ROOT
import src.models.fuxi_runner as base

logger = logging.getLogger(__name__)

_SESSIONS_V2: dict[str, ort.InferenceSession] = {}


def _ort_graph_opt_level() -> ort.GraphOptimizationLevel:
    raw = os.environ.get("FUXI_ORT_GRAPH_OPT_LEVEL", "basic").strip().lower()
    if raw in ("disable", "disabled", "none"):
        return ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if raw in ("extended",):
        return ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    if raw in ("all", "enable_all"):
        return ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.GraphOptimizationLevel.ORT_ENABLE_BASIC


def _session_v2(model_path: Path) -> ort.InferenceSession:
    key = str(model_path)
    if key in _SESSIONS_V2:
        return _SESSIONS_V2[key]

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = base._ort_intra_threads()
    options.inter_op_num_threads = base._ort_inter_threads()
    options.graph_optimization_level = _ort_graph_opt_level()

    cuda_options = {"arena_extend_strategy": "kSameAsRequested"}
    providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    logger.info(
        "FuXi(v2) session %s providers=%s graph_opt=%s",
        model_path.name,
        sess.get_providers(),
        os.environ.get("FUXI_ORT_GRAPH_OPT_LEVEL", "basic"),
    )
    _SESSIONS_V2[key] = sess
    return sess


def run_fuxi_forecast_v2(
    init_time: datetime,
    lead_times_hours: List[int],
    *,
    era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
    flip_north_south: bool = False,
) -> Dict[int, np.ndarray]:
    wanted = sorted({int(h) for h in lead_times_hours})
    if not wanted:
        return {}
    if any(h % base.STEP_HOURS != 0 for h in wanted):
        raise ValueError(f"FuXi lead times must be multiples of {base.STEP_HOURS}h: {wanted}")

    total_steps = max(wanted) // base.STEP_HOURS
    num_steps = base._default_stage_steps(total_steps)
    if sum(num_steps) != total_steps:
        raise RuntimeError(f"Stage steps {num_steps} do not sum to {total_steps}")

    init_ts = base.pd.Timestamp(init_time)
    tembs = base._time_encoding(init_ts, total_steps)
    input_np, _ = base.load_fuxi_input_tensor(
        init_time,
        root=era5_root,
        flip_north_south=flip_north_south,
    )

    weight_dir = base.DEFAULT_WEIGHTS_ROOT / "fuxi"
    models: Dict[str, ort.InferenceSession] = {}
    for stage in base.STAGES:
        path = weight_dir / f"{stage}.onnx"
        if not path.exists():
            raise FileNotFoundError(f"FuXi model missing: {path}")
        models[stage] = _session_v2(path)

    out: Dict[int, np.ndarray] = {}
    step = 0
    for stage_idx, n_sub in enumerate(num_steps):
        if n_sub == 0:
            continue
        stage = base.STAGES[stage_idx]
        session = models[stage]
        for _ in range(n_sub):
            temb = tembs[step]
            new_input = session.run(None, {"input": input_np, "temb": temb})[0]
            output = new_input[0, -1].astype(np.float32)
            step += 1
            lead_h = step * base.STEP_HOURS
            if lead_h in wanted:
                out[lead_h] = output
            input_np = new_input
    return out


class FuxiForecastRunnerV2:
    """Class-style FuXi runner reusing existing ONNX session cache behavior."""

    def __init__(
        self,
        *,
        era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
        flip_north_south: bool = False,
    ) -> None:
        self.era5_root = era5_root
        self.flip_north_south = flip_north_south

    def run(self, init_time: datetime, lead_times_hours: List[int]) -> Dict[int, np.ndarray]:
        return run_fuxi_forecast_v2(
            init_time,
            lead_times_hours,
            era5_root=self.era5_root,
            flip_north_south=self.flip_north_south,
        )

