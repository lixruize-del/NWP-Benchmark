#!/usr/bin/env python3
"""
Large-scale ERA5 inference + evaluation (single GPU).

Usage
-----
See repository README / plan: online mode streams metrics to CSV; offline mode
writes NetCDF for selected leads and variables via ``Saver``.

Climatology / ACC
-----------------
Anomaly Correlation is left for you to wire in ``src/common/evaluator.py`` once
a climatology path is available (no ``--climatology`` CLI flag by design).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.data_reader import (  # noqa: E402
    DEFAULT_ERA5_NPY_ROOT,
    Era5NpyLayout,
    load_fengwu_snapshot,
    load_fuxi_snapshot,
    load_npy_2d,
    load_pangu_ground_truth_stack,
    load_snapshot_by_channel_names,
    load_stormer_stack,
    snapshot_has_pressure_files,
)
from src.common.saver import Saver  # noqa: E402
from src.models.aifs_runner import run_aifs_forecast  # noqa: E402
from src.models.fengwu_runner import FengwuForecastRunner  # noqa: E402
from src.models.fuxi_runner import FuxiForecastRunner  # noqa: E402
from src.models.pangu_runner_class import PanguForecastRunner  # noqa: E402
from metrics import Metrics as WeightedMetrics  # noqa: E402

logger = logging.getLogger("run_large_scale")
DEFAULT_IFS_HRES_ROOT = Path("/ecmwf-era5-datasets/ifs-latest")
from src.common.repo_paths import debug_log_path  # noqa: E402

_DEBUG_LOG_PATH = debug_log_path("debug-bd4a8b.log")
_DEBUG_SESSION_ID = "bd4a8b"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": _DEBUG_SESSION_ID,
        "runId": os.environ.get("NWP_DEBUG_RUN_ID", "aifs-debug"),
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


@dataclass
class ModelAdapter:
    name: str
    run: Callable[..., Dict[int, np.ndarray]]
    channel_names: List[str]
    lat: np.ndarray

    load_gt: Callable[[datetime], np.ndarray]
    """Return ``(V, H, W)`` ground truth (ERA5 0.25° for most models; native only where noted)."""


def _lat_stormer() -> np.ndarray:
    # Stormer native grid (1.40625 deg) is cell-centered with latitude
    # increasing from south to north: [-89.296875, ..., 89.296875].
    ddeg = 1.40625
    lat_start = -90.0 + ddeg / 2.0
    lat_stop = 90.0 - ddeg / 2.0
    return np.linspace(lat_start, lat_stop, 128, dtype=np.float64)


def _lat_721() -> np.ndarray:
    return np.linspace(90.0, -90.0, 721, dtype=np.float64)

def _lat_aurora_720() -> np.ndarray:
    # Aurora 0.25° feeds (721,1440); inside the model, Batch.crop(patch_size=4)
    # drops the south-pole row → (720,1440). Latitudes are linspace(90,-90,721)[:-1],
    # NOT cell-centered 89.875…-89.875. See aurora.batch.Batch.crop and
    # https://microsoft.github.io/aurora/batch.html
    return np.linspace(90.0, -90.0, 721, dtype=np.float64)[:-1]


def _lon_1440() -> np.ndarray:
    return np.linspace(0.0, 360.0, 1440, endpoint=False, dtype=np.float64)


def _lon_256() -> np.ndarray:
    return np.linspace(0.0, 360.0, 256, endpoint=False, dtype=np.float64)


def _lon_512() -> np.ndarray:
    return np.linspace(0.0, 360.0, 512, endpoint=False, dtype=np.float64)


_CLIM_SINGLE_SHORTS = {"msl", "sp", "t2m", "u10", "v10"}
_CLIM_PRESSURE_SHORTS = {"z", "q", "t", "u", "v", "r", "w"}
_CLIM_LONG_TO_SHORT = {
    "geopotential": "z",
    "specific_humidity": "q",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "relative_humidity": "r",
    "vertical_velocity": "w",
}


def _load_climatology_721(
    valid_time: datetime,
    channel_names: List[str],
    *,
    era5_root: Path,
    flip_north_south: bool = False,
) -> np.ndarray:
    """Load daily climatology (1993-2016) on 721x1440 for requested channels.

    Use ``flip_north_south=True`` when stacking metrics vs fields oriented with row 0 at
    90°N (same convention as ``WeightedMetrics`` / ``metrics.lat``).
    """
    mmdd = valid_time.strftime("%m-%d")
    clim_root = era5_root / "climate_mean_day"
    pdir = clim_root / "1993-2016" / mmdd
    sdir = clim_root / "single" / "1993-2016" / mmdd
    out = np.full((len(channel_names), 721, 1440), np.nan, dtype=np.float32)

    for i, name in enumerate(channel_names):
        base = name
        lev: float | None = None
        if "_" in name:
            maybe_base, maybe_lev = name.rsplit("_", 1)
            if maybe_lev.isdigit():
                base = maybe_base
                lev = float(maybe_lev)
        else:
            # Support compact level names used by FuXi/others, e.g. Z500, T850.
            m = re.match(r"^([A-Za-z]+)(\d{2,4})$", name.strip())
            if m:
                base = m.group(1)
                lev = float(m.group(2))

        base = _CLIM_LONG_TO_SHORT.get(base.lower(), base.lower())
        path: Path | None = None
        if lev is None:
            if base in _CLIM_SINGLE_SHORTS:
                path = sdir / f"{base}.npy"
        else:
            if base in _CLIM_PRESSURE_SHORTS:
                path = pdir / f"{base}-{lev:.1f}.npy"
        if path is None or not path.exists():
            continue
        out[i] = load_npy_2d(path, flip_north_south=flip_north_south).astype(np.float32)
    return out


def build_adapter(
    model: str,
    era5_root: Path,
    ifs_hres_root: Path,
    fengwu_onnx: str,
) -> ModelAdapter:
    if model == "pangu":
        from src.common.data_reader import pangu_channel_names

        runner = PanguForecastRunner(era5_root=era5_root)

        def gt(t: datetime) -> np.ndarray:
            arr, _ = load_pangu_ground_truth_stack(t, root=era5_root, flip_north_south=False)
            return arr

        return ModelAdapter(
            "pangu",
            runner.run,
            pangu_channel_names(),
            _lat_721(),
            gt,
        )
    if model == "stormer":
        from src.models.stormer_runner import run_stormer_forecast, stormer_channel_names

        names = stormer_channel_names()

        def gt(t: datetime) -> np.ndarray:
            return load_stormer_stack(t, root=era5_root, flip_north_south=False)

        return ModelAdapter(
            "stormer",
            lambda it, leads, **kw: run_stormer_forecast(
                it,
                leads,
                era5_root=era5_root,
                list_intervals=[6],
                **kw,
            ),
            names,
            _lat_stormer(),
            gt,
        )
    if model == "fengwu":
        from src.common.data_reader import fengwu_channel_names

        runner = FengwuForecastRunner(
            era5_root=era5_root,
            onnx_name=fengwu_onnx,
            flip_north_south=False,
        )

        def gt(t: datetime) -> np.ndarray:
            return load_fengwu_snapshot(t, root=era5_root, flip_north_south=False)

        return ModelAdapter(
            "fengwu",
            runner.run,
            fengwu_channel_names(),
            _lat_721(),
            gt,
        )
    if model == "fengwu_v2":
        raise NotImplementedError(
            "fengwu_v2 uses IFS HRES inputs and is not implemented in this workspace; "
            "use --model fengwu with ERA5 np.25 (default weights fengwu_v1.onnx)."
        )
    if model == "fuxi":
        from src.common.data_reader import fuxi_channel_names

        runner = FuxiForecastRunner(era5_root=era5_root, flip_north_south=False)

        def gt(t: datetime) -> np.ndarray:
            return load_fuxi_snapshot(t, root=era5_root, flip_north_south=False)

        return ModelAdapter(
            "fuxi",
            runner.run,
            fuxi_channel_names(),
            _lat_721(),
            gt,
        )
    if model in ("graphcast", "graphcast_operational"):
        from src.models.graphcast_runner import GraphcastForecastRunner, graphcast_channel_names

        is_operational = model == "graphcast_operational"
        names = graphcast_channel_names() if not is_operational else []
        gc_runner = GraphcastForecastRunner(era5_root=era5_root)

        def _run_graphcast(it, leads, **_unused):
            del _unused
            if is_operational:
                raise NotImplementedError("graphcast_operational is not wired in this workspace.")
            return gc_runner.run(it, leads)

        def _gc_gt(t: datetime) -> np.ndarray:
            if not names:
                return np.zeros((1, 721, 1440), dtype=np.float32)
            return load_snapshot_by_channel_names(
                t, names, root=era5_root, flip_north_south=False
            )

        return ModelAdapter(
            "graphcast",
            _run_graphcast,
            names,
            _lat_721(),
            _gc_gt,
        )
    if model == "aurora":
        from src.models.aurora_runner import aurora_channel_names, run_aurora_forecast

        names = aurora_channel_names()
        lat720 = _lat_aurora_720()

        def gt(t: datetime) -> np.ndarray:
            return load_snapshot_by_channel_names(
                t, names, root=era5_root, flip_north_south=False
            )

        return ModelAdapter(
            "aurora",
            lambda it, leads, **kw: run_aurora_forecast(it, leads, era5_root=era5_root, **kw),
            names,
            lat720,
            gt,
        )
    if model == "neuralgcm":
        from src.models.neuralgcm_runner import (
            neuralgcm_channel_names,
            run_neuralgcm_forecast,
            DEFAULT_WEIGHTS_ROOT,
            _load_model_cached,
        )

        model_obj = _load_model_cached(DEFAULT_WEIGHTS_ROOT)
        names = neuralgcm_channel_names(weights_root=DEFAULT_WEIGHTS_ROOT)
        lat_nodes = int(model_obj.data_coords.horizontal.latitude_nodes)
        lon_nodes = int(model_obj.data_coords.horizontal.longitude_nodes)
        def _ngt(t: datetime) -> np.ndarray:
            return load_snapshot_by_channel_names(
                t, names, root=era5_root, flip_north_south=False
            )

        return ModelAdapter(
            "neuralgcm",
            lambda it, leads, **kw: run_neuralgcm_forecast(it, leads, era5_root=era5_root, **kw),
            names,
            np.linspace(90.0, -90.0, lat_nodes, dtype=np.float64),
            _ngt,
        )
    if model == "aifs":
        return ModelAdapter(
            "aifs",
            lambda it, leads, **kw: run_aifs_forecast(
                it, leads, ifs_hres_root=ifs_hres_root, **kw
            ),
            [],
            _lat_721(),
            lambda t: np.zeros((1, 721, 1440), dtype=np.float32),
        )
    raise ValueError(f"Unknown model {model}")


def iter_init_times(
    start: datetime,
    end: datetime,
    hours: List[int],
) -> List[datetime]:
    out: List[datetime] = []
    cur = start.date()
    end_date = end.date()
    while cur <= end_date:
        for h in sorted(hours):
            dt = datetime(cur.year, cur.month, cur.day, h, 0, 0)
            if start <= dt <= end:
                out.append(dt)
        cur += timedelta(days=1)
    return out


def sufficient_era5_snapshot(model: str, init: datetime, layout: Era5NpyLayout) -> bool:
    if model in ("aifs", "graphcast_operational", "fengwu_v2"):
        return True
    if not snapshot_has_pressure_files(layout, init):
        return False
    if model in ("fengwu", "fuxi"):
        t0 = init - timedelta(hours=6)
        if not snapshot_has_pressure_files(layout, t0):
            return False
    return True


def _norm_channel_name(name: str) -> str:
    return name.replace("_", "").strip().lower()


def subset_channels(
    data: np.ndarray,
    all_names: List[str],
    wanted: Optional[List[str]],
) -> tuple[np.ndarray, List[str]]:
    if not wanted:
        return data, all_names
    norm_to_idx: dict[str, int] = {}
    for i, ch in enumerate(all_names):
        norm_to_idx.setdefault(_norm_channel_name(ch), i)
    idx: List[int] = []
    names: List[str] = []
    missing: List[str] = []
    for w in wanted:
        i = norm_to_idx.get(_norm_channel_name(w))
        if i is None:
            missing.append(w)
            continue
        idx.append(i)
        names.append(all_names[i])
    if missing:
        raise ValueError(f"Unknown save_vars / channel names: {sorted(set(missing))}")
    return data[idx, ...], names


def resolve_channel_subset(
    all_names: List[str], wanted: Optional[List[str]], *, arg_name: str
) -> tuple[List[int], List[str]]:
    if not wanted:
        idx = list(range(len(all_names)))
        return idx, list(all_names)
    norm_to_idx: dict[str, int] = {}
    for i, ch in enumerate(all_names):
        norm_to_idx.setdefault(_norm_channel_name(ch), i)
    idx: List[int] = []
    names: List[str] = []
    missing: List[str] = []
    for w in wanted:
        i = norm_to_idx.get(_norm_channel_name(w))
        if i is None:
            missing.append(w)
            continue
        idx.append(i)
        names.append(all_names[i])
    if missing:
        raise ValueError(f"Unknown {arg_name} channel names: {sorted(set(missing))}")
    return idx, names


def _normalize_fuxi_eval_name(name: str) -> str:
    s = name.strip()
    low = s.lower()
    if low in {"msl", "t2m", "u10", "v10", "tp", "tp6h"}:
        return "tp" if low in {"tp", "tp6h"} else low
    m = re.match(r"^([a-zA-Z]+)[_]?(\d{2,4})$", s)
    if m:
        return f"{m.group(1).lower()}_{m.group(2)}"
    return low


def _fuxi_drop_rh_eval_channels(
    eval_idx: List[int], eval_names: List[str]
) -> tuple[List[int], List[str]]:
    """
    FuXi outputs ``r_*`` (RH). Metrics exclude RH so scores align with the
    standard ERA5 ``q_*`` moisture convention and avoid missing ``r_*`` analysis files.
    """
    out_idx: List[int] = []
    out_names: List[str] = []
    for i, n in zip(eval_idx, eval_names):
        if _normalize_fuxi_eval_name(n).startswith("r_"):
            continue
        out_idx.append(i)
        out_names.append(n)
    return out_idx, out_names


def load_gt_subset_by_model(
    model: str,
    valid_time: datetime,
    eval_names: List[str],
    *,
    era5_root: Path,
    adapter: ModelAdapter,
) -> np.ndarray:
    if not eval_names:
        return np.zeros((0, adapter.lat.shape[0], 1440), dtype=np.float32)
    load_names = eval_names
    if model == "fuxi":
        load_names = [_normalize_fuxi_eval_name(n) for n in eval_names]
    if model == "aifs" and "z_500" in eval_names:
        try:
            layout = Era5NpyLayout(era5_root)
            p = layout.pressure_path(valid_time, "z", 500.0)
            # region agent log
            _debug_log(
                "H12",
                "run_large_scale.py:load_gt_subset_by_model",
                "aifs z500 gt file diagnostics",
                {
                    "valid_time": valid_time.strftime("%Y%m%d%H"),
                    "gt_path": str(p),
                    "exists": bool(p.exists()),
                    "size_bytes": int(p.stat().st_size) if p.exists() else -1,
                },
            )
            # endregion
        except Exception as e:
            _debug_log(
                "H12",
                "run_large_scale.py:load_gt_subset_by_model",
                "aifs z500 gt file diagnostics failed",
                {"valid_time": valid_time.strftime("%Y%m%d%H"), "error": f"{type(e).__name__}:{e}"},
            )
    return load_snapshot_by_channel_names(
        valid_time, load_names, root=era5_root, flip_north_south=False
    )


def _compute_weighted_metrics(
    pred_eval: np.ndarray,
    gt_stack: np.ndarray,
    climatology_stack: np.ndarray | None,
    *,
    use_eval_wrmse: bool = False,
) -> dict[str, np.ndarray]:
    """
    Compute weighted metrics via metrics.py with output shape (C,).

    When ``use_eval_wrmse`` is True, the ``wrmse`` vector matches
    ``src.common.era5_eval_regrid.wrmse_stack_eval_style`` (numpy float64 diff + cosine weights / mean),
    instead of ``metrics.WeightedMetrics.WRMSE`` (torch + approximate pi).
    """
    meter = WeightedMetrics()
    pred_t = torch.from_numpy(np.asarray(pred_eval, dtype=np.float32))[None, ...]
    gt_t = torch.from_numpy(np.asarray(gt_stack, dtype=np.float32))[None, ...]
    clim_t = (
        torch.from_numpy(np.asarray(climatology_stack, dtype=np.float32))[None, ...]
        if climatology_stack is not None
        else None
    )

    with torch.no_grad():
        wrmse = meter.WRMSE(
            pred_t, gt_t, data_mask=None, clim_time_mean_daily=clim_t
        ).squeeze(0)
        bias = meter.Bias(
            pred_t, gt_t, data_mask=None, clim_time_mean_daily=clim_t
        ).squeeze(0)
        mae = meter.MAE(
            pred_t, gt_t, data_mask=None, clim_time_mean_daily=clim_t
        ).squeeze(0)
        if clim_t is None:
            activity = torch.full_like(wrmse, torch.nan)
            acc = torch.full_like(wrmse, torch.nan)
        else:
            activity = meter.Activity(
                pred_t, gt_t, data_mask=None, clim_time_mean_daily=clim_t
            ).squeeze(0)
            acc = meter.WACC(
                pred_t, gt_t, data_mask=None, clim_time_mean_daily=clim_t
            ).squeeze(0)

    def _to_np(x: torch.Tensor) -> np.ndarray:
        arr = x.detach().cpu().numpy().astype(np.float64, copy=False)
        arr[~np.isfinite(arr)] = np.nan
        return arr

    out_wrmse = _to_np(wrmse)
    if use_eval_wrmse:
        from src.common.era5_eval_regrid import wrmse_stack_eval_style

        out_wrmse = wrmse_stack_eval_style(pred_eval, gt_stack)

    return {
        "wrmse": out_wrmse,
        "bias": _to_np(bias),
        "mae": _to_np(mae),
        "activity": _to_np(activity),
        "acc": _to_np(acc),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NWP-Benchmark large-scale driver")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "pangu",
            "stormer",
            "fengwu",
            "fengwu_v2",
            "fuxi",
            "graphcast",
            "graphcast_operational",
            "aurora",
            "neuralgcm",
            "aifs",
        ],
    )
    parser.add_argument("--init_time", type=str, default=None, help="Single init time YYYYMMDDHH")
    parser.add_argument("--start", type=str, required=False, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=False, help="YYYY-MM-DD (inclusive)")
    parser.add_argument(
        "--init_hours",
        type=int,
        nargs="+",
        default=[0, 12],
        help="UTC hours for both daily initializations",
    )
    parser.add_argument(
        "--lead_times",
        type=int,
        nargs="+",
        required=True,
        help="Lead times in hours (e.g. 6 12 ... 240, multiples of 6 for most models)",
    )
    parser.add_argument("--mode", choices=("online", "offline", "both"), default="online")
    parser.add_argument(
        "--era5_root",
        type=Path,
        default=DEFAULT_ERA5_NPY_ROOT,
        help="ERA5 np.25 root directory",
    )
    parser.add_argument(
        "--ifs_hres_root",
        type=Path,
        default=DEFAULT_IFS_HRES_ROOT,
        help="IFS-HRES root (for models using IFS inputs)",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Online mode: append metrics here",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=5,
        help="Write CSV every N completed initialisation times",
    )
    parser.add_argument(
        "--fengwu_onnx",
        type=str,
        default="fengwu_v1.onnx",
        help="FengWu checkpoint file name under weights/fengwu/",
    )
    parser.add_argument(
        "--save_lead_range",
        type=int,
        nargs=2,
        default=None,
        metavar=("MIN_H", "MAX_H"),
        help="Offline: only save leads in this inclusive range (hours)",
    )
    parser.add_argument(
        "--save_vars",
        type=str,
        nargs="*",
        default=None,
        help="Offline: subset of channel short names (e.g. z_500 t_850)",
    )
    parser.add_argument("--nc_dir", type=Path, default=None, help="Offline NetCDF root")
    parser.add_argument(
        "--eval_vars",
        type=str,
        nargs="*",
        default=None,
        help="Online metrics: evaluate only this channel subset (default: all model channels)",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.init_time:
        inits = [datetime.strptime(args.init_time, "%Y%m%d%H")]
    else:
        if not args.start or not args.end:
            raise ValueError("--start/--end are required when --init_time is not provided.")
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(hour=23, minute=59)
        inits = iter_init_times(start, end, args.init_hours)
    layout = Era5NpyLayout(args.era5_root)

    if args.model == "fengwu_v2" and args.fengwu_onnx == "fengwu_v1.onnx":
        args.fengwu_onnx = "fengwu_v2.onnx"
    adapter = build_adapter(args.model, args.era5_root, args.ifs_hres_root, args.fengwu_onnx)
    eval_idx, eval_names = resolve_channel_subset(
        adapter.channel_names, args.eval_vars, arg_name="eval_vars"
    )
    if args.model == "fuxi":
        n_before = len(eval_names)
        eval_idx, eval_names = _fuxi_drop_rh_eval_channels(eval_idx, eval_names)
        dropped = n_before - len(eval_names)
        if dropped:
            logger.info(
                "FuXi: excluded %d relative-humidity channel(s) (r_*) from ERA5 metrics.",
                dropped,
            )
        if not eval_names:
            logger.error(
                "FuXi: no variables left for metrics after excluding r_*; "
                "use non-r channels in --eval_vars."
            )
            sys.exit(2)
    lead_times = sorted(set(args.lead_times))
    logger.info(
        "Model=%s | inits=%d | leads=%s | eval_vars=%d",
        args.model,
        len(inits),
        lead_times,
        len(eval_names),
    )

    rows: List[dict] = []
    done = 0
    out_csv = args.output_csv or (
        REPO_ROOT / "outputs" / args.model / "metrics.csv"
    )

    saver: Optional[Saver] = None
    nc_root = args.nc_dir or (REPO_ROOT / "outputs" / args.model / "nc")
    if args.mode in ("offline", "both"):
        saver = Saver(str(nc_root))
        if args.save_lead_range is None:
            raise ValueError("offline/both mode requires --save_lead_range MIN MAX")

    for init_time in inits:
        if not sufficient_era5_snapshot(args.model, init_time, layout):
            logger.warning("Skip init %s (missing NPY).", init_time)
            continue
        try:
            t_infer0 = time.perf_counter() if args.model == "fuxi" else None
            preds = adapter.run(init_time, lead_times)
        except FileNotFoundError as e:
            logger.warning("Skip init %s: %s", init_time, e)
            continue
        except NotImplementedError as e:
            logger.error("%s", e)
            sys.exit(2)

        infer_s = (
            time.perf_counter() - t_infer0 if args.model == "fuxi" and t_infer0 is not None else 0.0
        )
        init_str = init_time.strftime("%Y%m%d%H")
        save_nc_s = 0.0
        nc_writes = 0

        for lead in lead_times:
            if lead not in preds:
                continue
            pred = preds[lead]
            valid_time = init_time + timedelta(hours=int(lead))

            if args.mode in ("offline", "both"):
                assert saver is not None and args.save_lead_range is not None
                lo, hi = args.save_lead_range
                if lo <= lead <= hi:
                    sub, sub_names = subset_channels(pred, adapter.channel_names, args.save_vars)
                    lat = adapter.lat
                    # Select longitude coordinates by actual grid width to avoid
                    # mismatches on models like Aurora (720x1440).
                    if pred.shape[-1] == 1440:
                        lon = _lon_1440()
                    elif pred.shape[-1] == 256:
                        lon = _lon_256()
                    elif pred.shape[-1] == 512:
                        lon = _lon_512()
                    else:
                        raise ValueError(
                            f"Unsupported forecast grid width {pred.shape[-1]} for save lon coords"
                        )
                    if args.model == "fuxi":
                        t_save0 = time.perf_counter()
                    saver.save(
                        data=sub,
                        channel_mapping=sub_names,
                        init_time_str=init_str,
                        lead_time_hours=int(lead),
                        lat_values=lat.astype(np.float64),
                        lon_values=lon,
                    )
                    if args.model == "fuxi":
                        save_nc_s += time.perf_counter() - t_save0
                        nc_writes += 1
                if args.mode == "offline":
                    continue

            # online metrics
            if not adapter.channel_names:
                logger.warning("Model %s has no metric channel mapping yet; skip metrics.", args.model)
                continue
            pred_eval = pred[eval_idx, ...]
            if args.model in ("stormer", "aurora", "neuralgcm"):
                from src.common.era5_eval_regrid import regrid_model_pred_eval_to_era5_025

                pred_eval = regrid_model_pred_eval_to_era5_025(args.model, pred_eval)
            try:
                gt_stack = load_gt_subset_by_model(
                    args.model,
                    valid_time,
                    eval_names,
                    era5_root=args.era5_root,
                    adapter=adapter,
                )
            except FileNotFoundError:
                logger.warning(
                    "No ERA5 analysis on disk for valid %s (lead %sh) — skip metrics (forecast still ran).",
                    valid_time,
                    lead,
                )
                continue

            if gt_stack.shape != pred_eval.shape:
                logger.error(
                    "Shape mismatch pred %s vs gt %s at lead %sh",
                    pred_eval.shape,
                    gt_stack.shape,
                    lead,
                )
                continue

            if args.model == "aifs" and lead == 6 and "z_500" in eval_names:
                zi = eval_names.index("z_500")
                pz = pred_eval[zi].astype(np.float64, copy=False)
                gz = gt_stack[zi].astype(np.float64, copy=False)
                rmse_raw = float(np.sqrt(np.mean((pz - gz) ** 2)))
                rmse_lon_roll = float(np.sqrt(np.mean((np.roll(pz, 720, axis=1) - gz) ** 2)))
                rmse_lat_flip = float(np.sqrt(np.mean((pz[::-1, :] - gz) ** 2)))
                rmse_vs_init_gt = float("nan")
                try:
                    gt_init = load_gt_subset_by_model(
                        args.model,
                        init_time,
                        eval_names,
                        era5_root=args.era5_root,
                        adapter=adapter,
                    )
                    gz_init = gt_init[zi].astype(np.float64, copy=False)
                    rmse_vs_init_gt = float(np.sqrt(np.mean((pz - gz_init) ** 2)))
                except Exception:
                    pass
                rmse_gh_unweighted = float(np.sqrt(np.mean(((pz / 9.80665) - (gz / 9.80665)) ** 2)))
                lat_w = np.cos(np.deg2rad(adapter.lat.astype(np.float64)))
                lat_w = np.clip(lat_w, 0.0, None)
                lat_w = lat_w / lat_w.mean()
                w2d = lat_w[:, None] * np.ones((1, pz.shape[1]), dtype=np.float64)
                wrmse_gh_weighted = float(
                    np.sqrt(np.sum(w2d * (((pz / 9.80665) - (gz / 9.80665)) ** 2)) / np.sum(w2d))
                )
                lat = adapter.lat.astype(np.float64)
                lat_abs = np.abs(lat)
                belt_20_80 = (lat_abs >= 20.0) & (lat_abs <= 80.0)
                nh_20_80 = (lat >= 20.0) & (lat <= 80.0)
                wrmse_global_z = float(np.sqrt(np.sum(w2d * ((pz - gz) ** 2)) / np.sum(w2d)))
                wrmse_belt_20_80_z = float("nan")
                wrmse_nh_20_80_z = float("nan")
                if np.any(belt_20_80):
                    wb = lat_w[belt_20_80][:, None] * np.ones((1, pz.shape[1]), dtype=np.float64)
                    wrmse_belt_20_80_z = float(
                        np.sqrt(
                            np.sum(wb * ((pz[belt_20_80] - gz[belt_20_80]) ** 2)) / np.sum(wb)
                        )
                    )
                if np.any(nh_20_80):
                    wn = lat_w[nh_20_80][:, None] * np.ones((1, pz.shape[1]), dtype=np.float64)
                    wrmse_nh_20_80_z = float(
                        np.sqrt(np.sum(wn * ((pz[nh_20_80] - gz[nh_20_80]) ** 2)) / np.sum(wn))
                    )
                # region agent log
                _debug_log(
                    "H4",
                    "run_large_scale.py:main",
                    "aifs z500 pre-metric alignment diagnostics",
                    {
                        "init_time": init_str,
                        "valid_time": valid_time.strftime("%Y%m%d%H"),
                        "lead_h": int(lead),
                        "rmse_raw": rmse_raw,
                        "rmse_lon_roll_180": rmse_lon_roll,
                        "rmse_lat_flip": rmse_lat_flip,
                        "rmse_vs_init_gt": rmse_vs_init_gt,
                        "rmse_gh_unweighted": rmse_gh_unweighted,
                        "wrmse_gh_weighted": wrmse_gh_weighted,
                        "wrmse_global_z_weighted": wrmse_global_z,
                        "wrmse_abslat20_80_z_weighted": wrmse_belt_20_80_z,
                        "wrmse_nh20_80_z_weighted": wrmse_nh_20_80_z,
                        "rmse_pred_div_g_vs_gt": float(np.sqrt(np.mean(((pz / 9.80665) - gz) ** 2))),
                        "rmse_pred_mul_g_vs_gt": float(np.sqrt(np.mean(((pz * 9.80665) - gz) ** 2))),
                        "pred_nan_ratio": float(np.isnan(pz).mean()),
                        "gt_nan_ratio": float(np.isnan(gz).mean()),
                        "pred_min": float(np.nanmin(pz)),
                        "pred_max": float(np.nanmax(pz)),
                        "gt_min": float(np.nanmin(gz)),
                        "gt_max": float(np.nanmax(gz)),
                    },
                )
                # endregion

            climatology_stack: np.ndarray | None = None
            try:
                clim721 = _load_climatology_721(
                    valid_time, eval_names, era5_root=args.era5_root
                )
                if pred_eval.shape[-2:] == (721, 1440):
                    climatology_stack = clim721
                else:
                    climatology_stack = None
            except Exception as e:
                logger.warning("Climatology load failed at %s: %s", valid_time, e)
                climatology_stack = None

            # Compare against the evaluated prediction subset shape. Using the
            # full-model `pred` shape here would incorrectly force ACC to NaN
            # whenever eval_vars is a strict subset of model channels.
            if climatology_stack is not None and climatology_stack.shape != pred_eval.shape:
                logger.warning(
                    "Climatology shape mismatch %s vs pred %s at lead %sh; ACC set NaN.",
                    climatology_stack.shape,
                    pred_eval.shape,
                    lead,
                )
                climatology_stack = None

            m = _compute_weighted_metrics(pred_eval, gt_stack, climatology_stack)
            if args.model == "aifs" and lead == 6 and "z_500" in eval_names:
                zi = eval_names.index("z_500")
                pz = pred_eval[zi].astype(np.float64, copy=False)
                gz = gt_stack[zi].astype(np.float64, copy=False)
                lat_w = np.cos(np.deg2rad(adapter.lat.astype(np.float64)))
                lat_w = np.clip(lat_w, 0.0, None)
                lat_w = lat_w / lat_w.mean()
                w2d = lat_w[:, None] * np.ones((1, pz.shape[1]), dtype=np.float64)
                wrmse_manual = float(np.sqrt(np.sum(w2d * (pz - gz) ** 2) / np.sum(w2d)))
                mae_manual = float(np.sum(w2d * np.abs(pz - gz)) / np.sum(w2d))
                bias_manual = float(np.sum(w2d * (pz - gz)) / np.sum(w2d))
                # region agent log
                _debug_log(
                    "H7",
                    "run_large_scale.py:main",
                    "aifs z500 metrics cross-check",
                    {
                        "init_time": init_str,
                        "valid_time": valid_time.strftime("%Y%m%d%H"),
                        "lead_h": int(lead),
                        "wrmse_metrics_py": float(m["wrmse"][zi]),
                        "wrmse_manual_coslat": wrmse_manual,
                        "mae_metrics_py": float(m["mae"][zi]),
                        "mae_manual_coslat": mae_manual,
                        "bias_metrics_py": float(m["bias"][zi]),
                        "bias_manual_coslat": bias_manual,
                    },
                )
                # endregion
            for vi, var in enumerate(eval_names):
                rows.append(
                    {
                        "init_time": init_str,
                        "valid_time": valid_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "lead_hours": lead,
                        "variable": var,
                        "wrmse": float(m["wrmse"][vi]),
                        "bias": float(m["bias"][vi]),
                        "mae": float(m["mae"][vi]),
                        "activity": float(m["activity"][vi]),
                        "acc": float(m["acc"][vi]),
                    }
                )

        if args.model == "fuxi":
            logger.info(
                "FuXi run_large_scale init=%s infer_adapter_run_s=%.3f save_nc_s=%.3f nc_writes=%d",
                init_str,
                infer_s,
                save_nc_s,
                nc_writes,
            )
        done += 1
        if args.mode in ("online", "both") and rows and done % args.flush_every == 0:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            logger.info("Flushed %d rows → %s", len(rows), out_csv)

    if args.mode in ("online", "both") and rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        logger.info("Wrote %d rows → %s", len(rows), out_csv)
    elif args.mode in ("online", "both"):
        logger.warning("No metrics rows produced (check data availability).")


if __name__ == "__main__":
    main()
