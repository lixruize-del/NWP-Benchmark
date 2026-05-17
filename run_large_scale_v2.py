#!/usr/bin/env python3
"""v2 entrypoint: keeps base flow, swaps selected models to class-based runners."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict

import numpy as np

import run_large_scale as base
from src.common.data_reader import DEFAULT_ERA5_NPY_ROOT
from src.models.aifs_runner import AifsForecastRunner, aifs_channel_names
from src.models.fengwu_runner_v2 import FengwuForecastRunnerV2
from src.models.fuxi_runner_v2 import FuxiForecastRunnerV2
from src.models.graphcast_runner_v2 import GraphcastForecastRunnerV2
from src.models.neuralgcm_runner import NeuralGCMForecastRunnerV2, neuralgcm_channel_names
from src.models.pangu_runner_v2 import PanguForecastRunnerV2
from src.models.stormer_runner_v2 import StormerForecastRunnerV2, stormer_channel_names_v2

_ORIGINAL_BUILD_ADAPTER = base.build_adapter
_ORIGINAL_ITER_INIT_TIMES = base.iter_init_times
_MIN_INIT_TIME: datetime | None = None


def build_adapter_v2(
    model: str,
    era5_root: Path,
    ifs_hres_root: Path,
    fengwu_onnx: str,
) -> base.ModelAdapter:
    if model == "pangu":
        from src.common.data_reader import pangu_channel_names

        runner = PanguForecastRunnerV2(era5_root=era5_root)

        def gt(t: datetime) -> np.ndarray:
            arr, _ = base.load_pangu_ground_truth_stack(t, root=era5_root, flip_north_south=False)
            return arr

        return base.ModelAdapter("pangu", runner.run, pangu_channel_names(), base._lat_721(), gt)

    if model == "stormer":
        names = stormer_channel_names_v2()
        runner = StormerForecastRunnerV2(
            era5_root=era5_root,
            list_intervals=[6],
        )

        def gt(t: datetime) -> np.ndarray:
            return base.load_stormer_stack(t, root=era5_root, flip_north_south=False)

        return base.ModelAdapter("stormer", runner.run, names, base._lat_stormer(), gt)

    if model == "fengwu":
        from src.common.data_reader import fengwu_channel_names

        runner = FengwuForecastRunnerV2(
            era5_root=era5_root,
            onnx_name=fengwu_onnx,
            flip_north_south=False,
        )

        def gt(t: datetime) -> np.ndarray:
            return base.load_fengwu_snapshot(t, root=era5_root, flip_north_south=False)

        return base.ModelAdapter("fengwu", runner.run, fengwu_channel_names(), base._lat_721(), gt)

    if model == "fuxi":
        from src.common.data_reader import fuxi_channel_names

        runner = FuxiForecastRunnerV2(era5_root=era5_root, flip_north_south=False)

        def gt(t: datetime) -> np.ndarray:
            return base.load_fuxi_snapshot(t, root=era5_root, flip_north_south=False)

        return base.ModelAdapter("fuxi", runner.run, fuxi_channel_names(), base._lat_721(), gt)

    if model in ("graphcast", "graphcast_operational"):
        from src.models.graphcast_runner import graphcast_channel_names

        is_operational = model == "graphcast_operational"
        names = graphcast_channel_names() if not is_operational else []
        runner = GraphcastForecastRunnerV2(era5_root=era5_root)

        def _run(it, leads, **_unused):
            del _unused
            if is_operational:
                raise NotImplementedError("graphcast_operational is not wired in this workspace.")
            return runner.run(it, leads)

        def _gt(t: datetime) -> np.ndarray:
            if not names:
                return np.zeros((1, 721, 1440), dtype=np.float32)
            return base.load_snapshot_by_channel_names(t, names, root=era5_root, flip_north_south=False)

        return base.ModelAdapter("graphcast", _run, names, base._lat_721(), _gt)

    if model == "aifs":
        names = aifs_channel_names()
        runner = AifsForecastRunner(era5_root=era5_root)

        def _gt(t: datetime) -> np.ndarray:
            return base.load_snapshot_by_channel_names(t, names, root=era5_root, flip_north_south=False)

        return base.ModelAdapter("aifs", runner.run, names, base._lat_721(), _gt)

    if model == "neuralgcm":
        runner = NeuralGCMForecastRunnerV2(era5_root=era5_root)
        model_obj = runner.model
        names = neuralgcm_channel_names(weights_root=runner.weights_root)
        lat_nodes = int(model_obj.data_coords.horizontal.latitude_nodes)
        lon_nodes = int(model_obj.data_coords.horizontal.longitude_nodes)
        lat = np.linspace(90.0, -90.0, lat_nodes, dtype=np.float64)

        def _gt(t: datetime) -> np.ndarray:
            return base.load_snapshot_by_channel_names(
                t, names, root=era5_root, flip_north_south=False
            )

        return base.ModelAdapter("neuralgcm", runner.run, names, lat, _gt)

    return _ORIGINAL_BUILD_ADAPTER(model, era5_root, ifs_hres_root, fengwu_onnx)


def _iter_init_times_v2(
    start: datetime,
    end: datetime,
    hours: list[int],
) -> list[datetime]:
    out = _ORIGINAL_ITER_INIT_TIMES(start, end, hours)
    if _MIN_INIT_TIME is None:
        return out
    return [x for x in out if x >= _MIN_INIT_TIME]


def _extract_v2_args(argv: list[str]) -> tuple[list[str], datetime | None]:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--min_init_time", type=str, default=None)
    ns, rest = ap.parse_known_args(argv[1:])
    min_init: datetime | None = None
    if ns.min_init_time:
        min_init = datetime.strptime(ns.min_init_time, "%Y%m%d%H")
    return [argv[0], *rest], min_init


def main() -> None:
    global _MIN_INIT_TIME
    argv_for_base, min_init = _extract_v2_args(sys.argv)
    _MIN_INIT_TIME = min_init
    sys.argv = argv_for_base
    original = base.build_adapter
    original_iter = base.iter_init_times
    try:
        base.build_adapter = build_adapter_v2  # type: ignore[assignment]
        base.iter_init_times = _iter_init_times_v2  # type: ignore[assignment]
        base.main()
    finally:
        base.build_adapter = original  # type: ignore[assignment]
        base.iter_init_times = original_iter  # type: ignore[assignment]
        _MIN_INIT_TIME = None


if __name__ == "__main__":
    main()

