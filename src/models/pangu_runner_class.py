"""Class-based Pangu runner with behavior aligned to pangu_runner.py."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort

from src.common.data_reader import DEFAULT_ERA5_NPY_ROOT, load_pangu_inputs

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_ROOT = Path(
    os.environ.get("NWP_WEIGHTS_ROOT", "/ecmwf-era5-datasets/nwp_bench/assets/weights")
)

MODEL_STEPS: Dict[int, str] = {
    1: "pangu_weather_1.onnx",
    3: "pangu_weather_3.onnx",
    6: "pangu_weather_6.onnx",
    24: "pangu_weather_24.onnx",
}

PANGU_ROLL_STEP_H = 6


def _session_options() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.enable_mem_reuse = False
    opts.intra_op_num_threads = 4
    return opts


def _providers():
    cuda_provider_options = {"arena_extend_strategy": "kSameAsRequested"}
    return [("CUDAExecutionProvider", cuda_provider_options), "CPUExecutionProvider"]


def _step_sequence_6h_only(wanted_sorted: List[int]) -> List[int]:
    prev = 0
    steps: List[int] = []
    for target in wanted_sorted:
        delta = target - prev
        if delta < 0:
            raise ValueError(f"Non-monotonic lead targets after sort: {wanted_sorted}")
        if delta == 0:
            continue
        if delta % PANGU_ROLL_STEP_H != 0:
            raise ValueError(
                f"Lead gap {delta}h from {prev}h to {target}h is not a multiple of {PANGU_ROLL_STEP_H}h"
            )
        steps.extend([PANGU_ROLL_STEP_H] * (delta // PANGU_ROLL_STEP_H))
        prev = target
    return steps


def _run_step(
    session: ort.InferenceSession, upper: np.ndarray, surface: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pred_upper, pred_surface = session.run(None, {"input": upper, "input_surface": surface})
    return pred_upper.astype(np.float32), pred_surface.astype(np.float32)


class PanguForecastRunner:
    """
    Class-based equivalent of `run_pangu_forecast`.

    Key behavior guarantees:
    - Input/output contract and channel stacking order unchanged.
    - Rollout policy unchanged: 6-hour-only iterative inference.
    - ONNX session is cached on the instance and reused across runs.
    """

    def __init__(
        self,
        *,
        era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
        weights_root: Path | None = None,
    ) -> None:
        self.era5_root = era5_root
        self.weights_root = weights_root or DEFAULT_WEIGHTS_ROOT
        self._sessions: Dict[int, ort.InferenceSession] = {}

    def _get_session(self, step_h: int) -> ort.InferenceSession:
        if step_h in self._sessions:
            return self._sessions[step_h]

        model_path = self.weights_root / "pangu" / MODEL_STEPS[step_h]
        if not model_path.exists():
            raise FileNotFoundError(f"Required model file missing: {model_path}")

        t0 = time.perf_counter()
        logger.info("Loading Pangu ONNX %sh from %s", step_h, model_path)
        sess = ort.InferenceSession(
            str(model_path),
            sess_options=_session_options(),
            providers=_providers(),
        )
        self._sessions[step_h] = sess
        logger.info("Pangu session init step=%sh took %.3fs", step_h, time.perf_counter() - t0)
        return sess

    def run(self, init_time: datetime, lead_times_hours: List[int]) -> Dict[int, np.ndarray]:
        t_all = time.perf_counter()
        t_load = time.perf_counter()
        upper, surface = load_pangu_inputs(init_time, root=self.era5_root, flip_north_south=False)
        upper = upper.astype(np.float32)
        surface = surface.astype(np.float32)
        logger.info("Pangu input loading took %.3fs", time.perf_counter() - t_load)

        wanted = sorted({int(h) for h in lead_times_hours})
        if not wanted:
            return {}

        wanted_list = sorted(wanted)
        max_lead = max(wanted_list)
        step_sequence = _step_sequence_6h_only(wanted_list)
        logger.info(
            "Pangu(class) init=%s max_lead=%sh roll_step=%sh num_steps=%d",
            init_time.strftime("%Y%m%d%H"),
            max_lead,
            PANGU_ROLL_STEP_H,
            len(step_sequence),
        )

        curr_upper = upper
        curr_surface = surface
        elapsed = 0
        out: Dict[int, np.ndarray] = {}
        session = self._get_session(PANGU_ROLL_STEP_H)

        for step_idx, _ in enumerate(step_sequence, start=1):
            t_step = time.perf_counter()
            curr_upper, curr_surface = _run_step(session, curr_upper, curr_surface)
            elapsed += PANGU_ROLL_STEP_H
            logger.info(
                "Pangu(class) step=%d lead=%sh infer_s=%.3f",
                step_idx,
                elapsed,
                time.perf_counter() - t_step,
            )

            if elapsed in wanted:
                upper_flat = curr_upper.reshape(-1, curr_upper.shape[-2], curr_upper.shape[-1])
                stacked = np.concatenate([upper_flat, curr_surface], axis=0).astype(np.float32)
                out[elapsed] = stacked

        missing = [h for h in wanted if h not in out]
        if missing:
            raise RuntimeError(f"Failed to produce leads: {missing}")

        logger.info("Pangu(class) run total took %.3fs", time.perf_counter() - t_all)
        return out

