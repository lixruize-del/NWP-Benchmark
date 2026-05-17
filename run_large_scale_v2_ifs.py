#!/usr/bin/env python3
"""v2 entrypoint: keeps base flow, swaps selected models to class-based runners."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict

import numpy as np

import run_large_scale_ifs as base
from src.common.data_reader_ifs import DEFAULT_ERA5_NPY_ROOT
from src.models.aifs_runner import AifsForecastRunner, aifs_channel_names
from src.models.fengwu_runner_v2_ifs import FengwuForecastRunnerV2
from src.models.fuxi_runner_v2 import FuxiForecastRunnerV2
from src.models.graphcast_runner_v2 import GraphcastForecastRunnerV2
from src.models.pangu_runner_v2 import PanguForecastRunnerV2
from src.models.stormer_runner_v2 import StormerForecastRunnerV2, stormer_channel_names_v2

_ORIGINAL_BUILD_ADAPTER = base.build_adapter
_ORIGINAL_ITER_INIT_TIMES = base.iter_init_times
_ORIGINAL_EXPECTED_NC_PATH_FOR_INIT = getattr(base, "_expected_nc_path_for_init", None)
_MIN_INIT_TIME: datetime | None = None
_NC_ROOT_FOR_SKIP: Path | None = None
_INIT_TIME_FOR_SKIP: datetime | None = None
_LEAD_TIMES_FOR_SKIP: list[int] = []
_SAVE_LEAD_RANGE_FOR_SKIP: tuple[int, int] | None = None


def build_adapter_v2(
    model: str,
    era5_root: Path,
    ifs_hres_root: Path,
    fengwu_onnx: str,
) -> base.ModelAdapter:
    if model == "pangu":
        from src.common.data_reader_ifs import pangu_channel_names

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
        from src.common.data_reader_ifs import fengwu_channel_names

        runner = FengwuForecastRunnerV2(
            era5_root=era5_root,
            onnx_name=fengwu_onnx,
            flip_north_south=False,
        )

        def gt(t: datetime) -> np.ndarray:
            return base.load_fengwu_snapshot(t, root=era5_root, flip_north_south=False)

        return base.ModelAdapter("fengwu", runner.run, fengwu_channel_names(), base._lat_721(), gt)

    if model == "fuxi":
        from src.common.data_reader_ifs import fuxi_channel_names

        runner = FuxiForecastRunnerV2(era5_root=era5_root, flip_north_south=False)

        def gt(t: datetime) -> np.ndarray:
            return base.load_fuxi_snapshot(t, root=era5_root, flip_north_south=False)

        return base.ModelAdapter("fuxi", runner.run, fuxi_channel_names(), base._lat_721(), gt)

    if model == "graphcast":
        from src.models.graphcast_runner import graphcast_channel_names

        names = graphcast_channel_names()
        runner = GraphcastForecastRunnerV2(era5_root=era5_root)

        def _run(it, leads, **_unused):
            del _unused
            return runner.run(it, leads)

        def _gt(t: datetime) -> np.ndarray:
            return base.load_snapshot_by_channel_names(t, names, root=era5_root, flip_north_south=False)

        return base.ModelAdapter("graphcast", _run, names, base._lat_721(), _gt)

    if model == "graphcast_operational":
        from src.models.graphcast_runner_ifs import (
            GraphcastOperationalForecastRunner,
            graphcast_operational_channel_names,
        )

        names = graphcast_operational_channel_names()
        runner = GraphcastOperationalForecastRunner(era5_root=era5_root)

        def _run_oper(it, leads, **_unused):
            del _unused
            return runner.run(it, leads)

        def _gt_oper(t: datetime) -> np.ndarray:
            return base.load_snapshot_by_channel_names(t, names, root=era5_root, flip_north_south=False)

        return base.ModelAdapter("graphcast_operational", _run_oper, names, base._lat_721(), _gt_oper)

    if model == "aifs":
        names = aifs_channel_names()
        runner = AifsForecastRunner(era5_root=era5_root)

        def _gt(t: datetime) -> np.ndarray:
            return base.load_snapshot_by_channel_names(t, names, root=era5_root, flip_north_south=False)

        return base.ModelAdapter("aifs", runner.run, names, base._lat_721(), _gt)

    return _ORIGINAL_BUILD_ADAPTER(model, era5_root, ifs_hres_root, fengwu_onnx)


def _iter_init_times_v2(
    start: datetime,
    end: datetime,
    hours: list[int],
) -> list[datetime]:
    out = _ORIGINAL_ITER_INIT_TIMES(start, end, hours)
    if _MIN_INIT_TIME is None:
        filtered = out
    else:
        filtered = [x for x in out if x >= _MIN_INIT_TIME]
    if _NC_ROOT_FOR_SKIP is None:
        return filtered
    kept: list[datetime] = []
    for it in filtered:
        expected_nc = _expected_target_nc_for_skip(it)
        if expected_nc is not None and expected_nc.exists():
            base.logger.info(
                "Skip init %s (existing forecast found: %s).",
                it,
                expected_nc,
            )
            continue
        kept.append(it)
    return kept


def _extract_v2_args(
    argv: list[str],
) -> tuple[list[str], datetime | None, Path | None, datetime | None, list[int], tuple[int, int] | None]:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--min_init_time", type=str, default=None)
    ns, rest = ap.parse_known_args(argv[1:])
    min_init: datetime | None = None
    if ns.min_init_time:
        min_init = datetime.strptime(ns.min_init_time, "%Y%m%d%H")
    argv_for_base = [argv[0], *rest]
    model: str | None = None
    has_fengwu_onnx = False
    nc_dir: Path | None = None
    init_time: datetime | None = None
    lead_times: list[int] = []
    save_lead_range: tuple[int, int] | None = None
    for i, tok in enumerate(rest):
        if tok == "--model" and i + 1 < len(rest):
            model = rest[i + 1]
        elif tok.startswith("--model="):
            model = tok.split("=", 1)[1]
        elif tok == "--nc_dir" and i + 1 < len(rest):
            nc_dir = Path(rest[i + 1])
        elif tok.startswith("--nc_dir="):
            nc_dir = Path(tok.split("=", 1)[1])
        elif tok == "--init_time" and i + 1 < len(rest):
            init_time = datetime.strptime(rest[i + 1], "%Y%m%d%H")
        elif tok.startswith("--init_time="):
            init_time = datetime.strptime(tok.split("=", 1)[1], "%Y%m%d%H")
        elif tok == "--lead_times":
            j = i + 1
            vals: list[int] = []
            while j < len(rest) and not rest[j].startswith("--"):
                try:
                    vals.append(int(rest[j]))
                except ValueError:
                    break
                j += 1
            if vals:
                lead_times = vals
        elif tok.startswith("--lead_times="):
            raw = tok.split("=", 1)[1].replace(",", " ")
            try:
                lead_times = [int(x) for x in raw.split() if x.strip()]
            except ValueError:
                pass
        elif tok == "--save_lead_range" and i + 2 < len(rest):
            try:
                save_lead_range = (int(rest[i + 1]), int(rest[i + 2]))
            except ValueError:
                pass
        elif tok.startswith("--save_lead_range="):
            raw = tok.split("=", 1)[1].replace(",", " ")
            parts = [x for x in raw.split() if x.strip()]
            if len(parts) >= 2:
                try:
                    save_lead_range = (int(parts[0]), int(parts[1]))
                except ValueError:
                    pass
        if tok == "--fengwu_onnx" or tok.startswith("--fengwu_onnx="):
            has_fengwu_onnx = True
    # IFS flow should default FengWu to v2 weights unless user explicitly overrides.
    if model == "fengwu" and not has_fengwu_onnx:
        argv_for_base.extend(["--fengwu_onnx", "fengwu_v2.onnx"])
    if nc_dir is None and model:
        nc_dir = base.REPO_ROOT / "outputs" / model / "nc"
    return argv_for_base, min_init, nc_dir, init_time, lead_times, save_lead_range


def _expected_target_nc_for_skip(init_time: datetime) -> Path | None:
    if _ORIGINAL_EXPECTED_NC_PATH_FOR_INIT is None or _NC_ROOT_FOR_SKIP is None:
        return None
    if not _LEAD_TIMES_FOR_SKIP:
        return None
    return _ORIGINAL_EXPECTED_NC_PATH_FOR_INIT(
        init_time,
        _LEAD_TIMES_FOR_SKIP,
        _SAVE_LEAD_RANGE_FOR_SKIP,
        _NC_ROOT_FOR_SKIP,
    )


def _expected_nc_path_for_init_v2(
    init_time: datetime,
    lead_times: list[int],
    save_lead_range: tuple[int, int] | None,
    nc_root: Path,
) -> Path | None:
    """
    Keep base behavior for representative-target detection.
    """
    if _ORIGINAL_EXPECTED_NC_PATH_FOR_INIT is not None:
        return _ORIGINAL_EXPECTED_NC_PATH_FOR_INIT(init_time, lead_times, save_lead_range, nc_root)
    return None


def main() -> None:
    global _MIN_INIT_TIME, _NC_ROOT_FOR_SKIP, _INIT_TIME_FOR_SKIP, _LEAD_TIMES_FOR_SKIP, _SAVE_LEAD_RANGE_FOR_SKIP
    argv_for_base, min_init, nc_root_for_skip, init_time_for_skip, lead_times_for_skip, save_lead_range_for_skip = _extract_v2_args(sys.argv)
    _MIN_INIT_TIME = min_init
    _NC_ROOT_FOR_SKIP = nc_root_for_skip
    _INIT_TIME_FOR_SKIP = init_time_for_skip
    _LEAD_TIMES_FOR_SKIP = sorted(set(int(x) for x in lead_times_for_skip))
    _SAVE_LEAD_RANGE_FOR_SKIP = save_lead_range_for_skip
    sys.argv = argv_for_base
    original = base.build_adapter
    original_iter = base.iter_init_times
    original_expected_nc = getattr(base, "_expected_nc_path_for_init", None)
    try:
        if _NC_ROOT_FOR_SKIP is not None and _INIT_TIME_FOR_SKIP is not None:
            expected_nc = _expected_target_nc_for_skip(_INIT_TIME_FOR_SKIP)
            if expected_nc is not None and expected_nc.exists():
                base.logger.info(
                    "Skip init %s (existing forecast found: %s).",
                    _INIT_TIME_FOR_SKIP,
                    expected_nc,
                )
                return
        base.build_adapter = build_adapter_v2  # type: ignore[assignment]
        base.iter_init_times = _iter_init_times_v2  # type: ignore[assignment]
        if original_expected_nc is not None:
            base._expected_nc_path_for_init = _expected_nc_path_for_init_v2  # type: ignore[attr-defined]
        base.main()
    finally:
        base.build_adapter = original  # type: ignore[assignment]
        base.iter_init_times = original_iter  # type: ignore[assignment]
        if original_expected_nc is not None:
            base._expected_nc_path_for_init = original_expected_nc  # type: ignore[attr-defined]
        _MIN_INIT_TIME = None
        _NC_ROOT_FOR_SKIP = None
        _INIT_TIME_FOR_SKIP = None
        _LEAD_TIMES_FOR_SKIP = []
        _SAVE_LEAD_RANGE_FOR_SKIP = None


if __name__ == "__main__":
    main()

