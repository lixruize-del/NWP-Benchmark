#!/usr/bin/env python3
"""Build local DOY percentile baselines (e.g. heatwave p90, cold-surge p10) from historical daily t2m."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from heatwave_common import (
    detect_t2m_var,
    iter_climate_dated_files,
    parse_ymd,
    select_dated_files,
    to_celsius,
)

import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from src.common.repo_paths import debug_log_path  # noqa: E402

DEBUG_LOG_PATH = debug_log_path("debug-dce7a8.log")
DEBUG_SESSION_ID = "dce7a8"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build per-grid DOY percentile thresholds from historical daily max t2m "
            "(default: p90 for heatwave-like extremes, p10 for cold-surge-like lows)."
        )
    )
    p.add_argument(
        "--climate-dir",
        type=Path,
        default=Path("/ecmwf-era5-datasets/climate/2_metre_temperature"),
    )
    p.add_argument("--start-date", type=str, default="1979-01-01")
    p.add_argument("--end-date", type=str, default="2024-12-31")
    p.add_argument("--window-days", type=int, default=15)
    p.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Deprecated: single percentile only; prefer --percentiles.",
    )
    p.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=None,
        metavar="P",
        help=(
            "One or more percentiles to compute on the pooled calendar window "
            "(e.g. 90 10 for heatwave threshold and cold baseline). "
            "Default if neither --percentiles nor --percentile is given: 90 10."
        ),
    )
    p.add_argument(
        "--var-name",
        type=str,
        default="",
        help="Optional source variable name. Auto-detected if empty.",
    )
    p.add_argument("--doy-start", type=int, default=1, help="Start DOY to compute.")
    p.add_argument("--doy-end", type=int, default=366, help="End DOY to compute.")
    p.add_argument(
        "--time-chunk",
        type=int,
        default=90,
        help="Time chunk size passed to xarray open_mfdataset.",
    )
    p.add_argument("--lat-chunk", type=int, default=90)
    p.add_argument("--lon-chunk", type=int, default=180)
    p.add_argument(
        "--out-dir", type=Path, default=Path("reports/heatwave_baseline")
    )
    p.add_argument(
        "--out-name",
        type=str,
        default="t2m_p10_p90_doy_001_366.nc",
        help="Output NetCDF filename (multiple variables if multiple percentiles).",
    )
    p.add_argument(
        "--split-output",
        action="store_true",
        help=(
            "Write separate files for upper-tail (heatwave) and lower-tail "
            "(coldwave) baselines after one shared computation pass."
        ),
    )
    p.add_argument(
        "--heatwave-out-name",
        type=str,
        default="heatwave_baseline_p90_doy_001_366.nc",
        help="Output filename for percentiles >= 50 when --split-output is used.",
    )
    p.add_argument(
        "--coldwave-out-name",
        type=str,
        default="coldwave_baseline_p10_doy_001_366.nc",
        help="Output filename for percentiles < 50 when --split-output is used.",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Optional log file path. If empty, only print to stdout.",
    )
    p.add_argument(
        "--progress-every-doy",
        type=int,
        default=10,
        help="Print progress every N DOYs during percentile loop.",
    )
    p.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Resume from existing output files by skipping fully finite DOY slices "
            "and recomputing DOYs that still contain NaN."
        ),
    )
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute all DOYs even if --resume-existing outputs already exist.",
    )
    p.add_argument(
        "--output-dtype",
        type=str,
        default="float32",
        choices=("float32", "float64"),
        help="Numeric dtype used for stored baseline fields.",
    )
    p.add_argument(
        "--checkpoint-file",
        type=str,
        default="",
        help=(
            "Checkpoint json path for resume. Default: <out-dir>/baseline_build_checkpoint.json"
        ),
    )
    p.add_argument(
        "--open-engine",
        type=str,
        default="h5netcdf",
        choices=("h5netcdf", "netcdf4"),
        help="Engine for opening climate NetCDF files with xarray.open_mfdataset.",
    )
    return p.parse_args()


def wrap_doy(doy: int) -> int:
    return ((doy - 1) % 366) + 1


def build_window_doys(center_doy: int, half_window: int) -> list[int]:
    return [wrap_doy(center_doy + offset) for offset in range(-half_window, half_window + 1)]


def resolve_percentiles(args: argparse.Namespace) -> list[float]:
    if args.percentile is not None and args.percentiles is not None:
        raise SystemExit("Use either --percentile or --percentiles, not both.")
    if args.percentiles is not None:
        raw = list(args.percentiles)
    elif args.percentile is not None:
        raw = [args.percentile]
    else:
        raw = [90.0, 10.0]
    out: list[float] = []
    seen: set[float] = set()
    for p in raw:
        if not (0.0 < p < 100.0):
            raise ValueError(f"percentile must be in (0, 100), got {p}")
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    frac = max(0.0, min(1.0, done / total))
    fill = int(round(frac * width))
    return "[" + ("#" * fill) + ("-" * (width - fill)) + "]"


def _log(msg: str, log_path: Path | None) -> None:
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    line = f"{ts} {msg}"
    print(line, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# region agent log
def _agent_log(
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
) -> None:
    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sessionId": DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        # Never let debug instrumentation break the actual build task.
        pass
# endregion


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)
    tmp.replace(path)


def _ensure_file_and_vars(
    out_file: Path,
    lat_name: str,
    lon_name: str,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    var_names: list[str],
    dtype: str,
    global_attrs: dict[str, str | int | float | list[float]],
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fill = np.nan
    with Dataset(str(out_file), "a", format="NETCDF4") as ds:
        if "doy" not in ds.dimensions:
            ds.createDimension("doy", 366)
        if lat_name not in ds.dimensions:
            ds.createDimension(lat_name, int(lat_vals.size))
        if lon_name not in ds.dimensions:
            ds.createDimension(lon_name, int(lon_vals.size))
        if "doy" not in ds.variables:
            v = ds.createVariable("doy", "i4", ("doy",))
            v[:] = np.arange(1, 367, dtype=np.int32)
        if lat_name not in ds.variables:
            v = ds.createVariable(lat_name, "f8", (lat_name,))
            v[:] = lat_vals.astype(np.float64)
        if lon_name not in ds.variables:
            v = ds.createVariable(lon_name, "f8", (lon_name,))
            v[:] = lon_vals.astype(np.float64)
        for key, val in global_attrs.items():
            setattr(ds, key, val)
        for vn in var_names:
            if vn not in ds.variables:
                v = ds.createVariable(
                    vn,
                    "f4" if dtype == "float32" else "f8",
                    ("doy", lat_name, lon_name),
                    zlib=True,
                    complevel=2,
                    fill_value=fill,
                )
                v.setncattr("units", "degC")


def _compute_completed_doys(out_file: Path, var_name: str) -> np.ndarray:
    done = np.zeros(366, dtype=bool)
    if not out_file.exists():
        return done
    with Dataset(str(out_file), "r") as ds:
        if var_name not in ds.variables:
            return done
        var = ds.variables[var_name]
        if var.shape[0] != 366:
            return done
        for i in range(366):
            slab = var[i, :, :]
            done[i] = bool(np.isfinite(slab).all())
    return done


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()
    run_id = f"run_{int(time.time())}"
    log_path = Path(args.log_file.strip()) if args.log_file.strip() else None
    percentiles = resolve_percentiles(args)
    checkpoint_path = (
        Path(args.checkpoint_file.strip())
        if args.checkpoint_file.strip()
        else (args.out_dir / "baseline_build_checkpoint.json")
    )
    if args.window_days <= 0:
        raise ValueError("--window-days must be positive.")
    if args.doy_start < 1 or args.doy_end > 366 or args.doy_start > args.doy_end:
        raise ValueError("--doy-start/--doy-end must satisfy 1<=start<=end<=366.")
    start_dt = parse_ymd(args.start_date)
    end_dt = parse_ymd(args.end_date)
    if end_dt < start_dt:
        raise ValueError("--end-date must be >= --start-date.")

    t_scan0 = time.perf_counter()
    files = list(iter_climate_dated_files(args.climate_dir))
    selected = select_dated_files(files, start_dt, end_dt)
    if not selected:
        raise SystemExit("No files found in selected baseline range.")
    _log(
        f"start baseline build period={args.start_date}..{args.end_date} "
        f"window_days={args.window_days} percentiles={percentiles} files={len(selected)}",
        log_path,
    )
    _log(
        f"stage files_scan done elapsed_s={time.perf_counter() - t_scan0:.3f} "
        f"all_files={len(files)} selected={len(selected)}",
        log_path,
    )

    t_probe0 = time.perf_counter()
    probe = xr.open_dataset(selected[0].path)
    try:
        var_name = detect_t2m_var(probe, args.var_name)
        units_hint = str(probe[var_name].attrs.get("units", "")).strip()
    finally:
        probe.close()
    _log(
        f"stage probe done elapsed_s={time.perf_counter() - t_probe0:.3f} "
        f"var={var_name} units_hint={units_hint!r}",
        log_path,
    )

    paths = [str(x.path) for x in selected]
    chunks = {"time": args.time_chunk, "latitude": args.lat_chunk, "longitude": args.lon_chunk}
    # region agent log
    _agent_log(
        run_id,
        "H4",
        "build_heatwave_baseline_percentile.py:310",
        "run configuration before open_mfdataset",
        {
            "selected_files": len(selected),
            "percentiles": percentiles,
            "window_days": int(args.window_days),
            "chunks": chunks,
            "open_engine": args.open_engine,
            "resume_existing": bool(args.resume_existing),
            "force_recompute": bool(args.force_recompute),
            "output_dtype": args.output_dtype,
        },
    )
    # endregion
    t_open0 = time.perf_counter()
    ds = xr.open_mfdataset(
        paths,
        combine="nested",
        concat_dim="time",
        chunks=chunks,
        engine=args.open_engine,
    )
    _log(
        f"stage open_mfdataset done elapsed_s={time.perf_counter() - t_open0:.3f}",
        log_path,
    )
    try:
        _log(f"opened dataset variable={var_name} source_units={units_hint!r}", log_path)
        t_pre0 = time.perf_counter()
        arr = ds[var_name]
        # Units: ERA5 / many CDS archives store 2t in Kelvin (units 'K').
        # to_celsius only subtracts 273.15 when attrs say K/kelvin.
        # For any linear statistic including quantiles:
        #   quantile(T_C) = quantile(T_K) - 273.15  (same sample set).
        # So conversion order does not change the math; °C is used so outputs match
        # evaluation thresholds stated in degC and human-readable baselines.
        arr = to_celsius(arr)
        _log(
            f"stage to_celsius done elapsed_s={time.perf_counter() - t_pre0:.3f}",
            log_path,
        )
        # Build day-level source series from sub-daily timesteps.
        # Rule:
        #   - upper-tail thresholds (e.g. p90) come from daily max t2m
        #   - lower-tail thresholds (e.g. p10) come from daily min t2m
        # This must aggregate across intraday samples; otherwise quantiles are
        # computed over hourly data (24x bigger sample axis and wrong definition).
        t_reduce0 = time.perf_counter()
        for extra_dim in ("step", "hour", "valid_time"):
            if extra_dim in arr.dims:
                arr = arr.max(dim=extra_dim, skipna=True)
        _log(
            f"stage intraday_reduce done elapsed_s={time.perf_counter() - t_reduce0:.3f} dims={list(arr.dims)}",
            log_path,
        )

        t_group0 = time.perf_counter()
        if "time" in arr.coords:
            # region agent log
            _agent_log(
                run_id,
                "H7",
                "build_heatwave_baseline_percentile.py:420",
                "before day coord creation",
                {
                    "arr_dims": list(arr.dims),
                    "arr_shape": [int(arr.sizes[k]) for k in arr.dims],
                    "arr_chunks": str(getattr(arr.data, "chunks", None)),
                },
            )
            # endregion
            _log("stage day_groupby step=before_day_coord", log_path)
            t_daycoord0 = time.perf_counter()
            day_coord = xr.DataArray(arr["time"].dt.floor("D"), dims=("time",), name="day")
            _log(
                f"stage day_groupby step=day_coord_created elapsed_s={time.perf_counter() - t_daycoord0:.3f}",
                log_path,
            )
            # region agent log
            _agent_log(
                run_id,
                "H7",
                "build_heatwave_baseline_percentile.py:427",
                "day coord created",
                {
                    "day_coord_dims": list(day_coord.dims),
                    "day_coord_size": int(day_coord.sizes.get("time", 0)),
                    "day_coord_chunks": str(getattr(day_coord.data, "chunks", None)),
                },
            )
            # endregion
            t_assign0 = time.perf_counter()
            arr = arr.assign_coords(day=day_coord)
            _log(
                f"stage day_groupby step=assign_day_coord elapsed_s={time.perf_counter() - t_assign0:.3f}",
                log_path,
            )
            t_gmax0 = time.perf_counter()
            daily_max = arr.resample(time="1D").max(skipna=True)
            _log(
                f"stage day_groupby step=daily_max_resample_graph elapsed_s={time.perf_counter() - t_gmax0:.3f}",
                log_path,
            )
            t_gmin0 = time.perf_counter()
            daily_min = arr.resample(time="1D").min(skipna=True)
            _log(
                f"stage day_groupby step=daily_min_resample_graph elapsed_s={time.perf_counter() - t_gmin0:.3f}",
                log_path,
            )
            # region agent log
            _agent_log(
                run_id,
                "H7",
                "build_heatwave_baseline_percentile.py:446",
                "daily max/min graph created",
                {
                    "daily_max_shape": [int(daily_max.sizes[k]) for k in daily_max.dims],
                    "daily_min_shape": [int(daily_min.sizes[k]) for k in daily_min.dims],
                    "daily_max_chunks": str(getattr(daily_max.data, "chunks", None)),
                    "daily_min_chunks": str(getattr(daily_min.data, "chunks", None)),
                },
            )
            # endregion
        else:
            # Fallback for unusual inputs without time coordinate.
            times = np.array([np.datetime64(x.dt.isoformat()) for x in selected])
            arr = arr.assign_coords(time=("time", times))
            daily_max = arr
            daily_min = arr
        _log(
            f"stage day_groupby done elapsed_s={time.perf_counter() - t_group0:.3f}",
            log_path,
        )
        # Materialize daily fields into in-memory NumPy arrays once.
        # This avoids rebuilding/dispatching dask task graphs inside the DOY loop.
        t_load0 = time.perf_counter()
        daily_max = daily_max.load()
        daily_min = daily_min.load()
        _log(
            f"stage daily_load done elapsed_s={time.perf_counter() - t_load0:.3f}",
            log_path,
        )
        # region agent log
        _agent_log(
            run_id,
            "H8",
            "build_heatwave_baseline_percentile.py:daily_load",
            "daily max/min materialized by load",
            {
                "daily_max_dtype": str(daily_max.dtype),
                "daily_min_dtype": str(daily_min.dtype),
                "daily_max_is_dask": bool(hasattr(daily_max.data, "chunks")),
                "daily_min_is_dask": bool(hasattr(daily_min.data, "chunks")),
            },
        )
        # endregion
        doy = xr.DataArray(daily_max["time"].dt.dayofyear, dims=("time",), name="doy")
        daily_max = daily_max.assign_coords(doy=doy)
        daily_min = daily_min.assign_coords(doy=doy)
        doy_values_np = np.asarray(doy.values, dtype=np.int32)
        # region agent log
        _agent_log(
            run_id,
            "H1",
            "build_heatwave_baseline_percentile.py:347",
            "prepared daily fields for quantile loop",
            {
                "daily_max_dims": list(daily_max.dims),
                "daily_max_shape": [int(daily_max.sizes[k]) for k in daily_max.dims],
                "daily_max_chunks": str(getattr(daily_max.data, "chunks", None)),
            },
        )
        # endregion

        lat_name = "latitude" if "latitude" in daily_max.dims else "lat"
        lon_name = "longitude" if "longitude" in daily_max.dims else "lon"
        lat_vals = daily_max[lat_name].values.astype(np.float64)
        lon_vals = daily_max[lon_name].values.astype(np.float64)

        half_window = args.window_days // 2
        total_doy = args.doy_end - args.doy_start + 1
        t_loop = time.perf_counter()
        _log(f"begin DOY loop total={total_doy}", log_path)

        out_file = args.out_dir / args.out_name
        p_to_var = {p: f"t2m_p{int(p)}_c" for p in percentiles}
        p_to_file: dict[float, Path] = {}
        if args.split_output:
            for p in percentiles:
                p_to_file[p] = (
                    args.out_dir / args.heatwave_out_name
                    if p >= 50.0
                    else args.out_dir / args.coldwave_out_name
                )
        else:
            for p in percentiles:
                p_to_file[p] = out_file

        file_to_vars: dict[Path, list[str]] = {}
        for p in percentiles:
            file_to_vars.setdefault(p_to_file[p], []).append(p_to_var[p])

        global_attrs = {
            "description": "Local DOY percentile baselines (degC)",
            "source_units_hint": units_hint,
            "stored_in": "degC",
            "baseline_period": f"{args.start_date}..{args.end_date}",
            "window_days": int(args.window_days),
            "percentiles": percentiles,
            "tail_source_rule": "Percentiles >= 50 use daily max t2m; percentiles < 50 use daily min t2m.",
        }
        t_outprep0 = time.perf_counter()
        for fpath, vnames in file_to_vars.items():
            _ensure_file_and_vars(
                out_file=fpath,
                lat_name=lat_name,
                lon_name=lon_name,
                lat_vals=lat_vals,
                lon_vals=lon_vals,
                var_names=vnames,
                dtype=args.output_dtype,
                global_attrs=global_attrs,
            )
        _log(
            f"stage output_prepare done elapsed_s={time.perf_counter() - t_outprep0:.3f} files={len(file_to_vars)}",
            log_path,
        )

        completed_by_p: dict[float, np.ndarray] = {}
        ckpt = _load_checkpoint(checkpoint_path)
        completed_from_ckpt: dict[float, set[int]] = {}
        try:
            ckpt_completed = ckpt.get("completed_doys", {})
            if isinstance(ckpt_completed, dict):
                for k, v in ckpt_completed.items():
                    p = float(k)
                    if isinstance(v, list):
                        completed_from_ckpt[p] = {int(x) for x in v if 1 <= int(x) <= 366}
        except Exception:
            completed_from_ckpt = {}
        t_resume_all0 = time.perf_counter()
        for p in percentiles:
            t_resume0 = time.perf_counter()
            if args.resume_existing and not args.force_recompute:
                done = _compute_completed_doys(p_to_file[p], p_to_var[p])
            else:
                done = np.zeros(366, dtype=bool)
            for d in sorted(completed_from_ckpt.get(p, set())):
                done[d - 1] = True
            completed_by_p[p] = done
            _log(
                f"resume map percentile={p} complete_doys={int(done.sum())}/366 "
                f"file={p_to_file[p]} var={p_to_var[p]}",
                log_path,
            )
            # region agent log
            _agent_log(
                run_id,
                "H3",
                "build_heatwave_baseline_percentile.py:415",
                "resume scan finished for percentile",
                {
                    "percentile": p,
                    "completed_doys": int(done.sum()),
                    "resume_scan_s": round(time.perf_counter() - t_resume0, 3),
                    "target_file": str(p_to_file[p]),
                },
            )
            # endregion
        _log(
            f"stage resume_scan_all done elapsed_s={time.perf_counter() - t_resume_all0:.3f}",
            log_path,
        )
        _log(f"checkpoint file: {checkpoint_path}", log_path)
        quantile_sums: dict[float, float] = {p: 0.0 for p in percentiles}
        write_sums: dict[float, float] = {p: 0.0 for p in percentiles}
        quantile_counts: dict[float, int] = {p: 0 for p in percentiles}

        for idx, center_doy in enumerate(range(args.doy_start, args.doy_end + 1), start=1):
            t_subset0 = time.perf_counter()
            window_doys = build_window_doys(center_doy, half_window)
            subset_max = daily_max.where(daily_max["doy"].isin(window_doys), drop=True)
            subset_min = daily_min.where(daily_min["doy"].isin(window_doys), drop=True)
            subset_build_s = time.perf_counter() - t_subset0
            if subset_max.sizes.get("time", 0) == 0:
                raise ValueError(f"No samples for doy={center_doy}, window={window_doys}")
            if idx <= 3:
                # region agent log
                _agent_log(
                    run_id,
                    "H6",
                    "build_heatwave_baseline_percentile.py:DOY_SUBSET",
                    "subset prepared for early DOY",
                    {
                        "doy": center_doy,
                        "idx": idx,
                        "subset_build_s": round(subset_build_s, 6),
                        "window_size": len(window_doys),
                        "window_match_count": int(np.isin(doy_values_np, np.asarray(window_doys, dtype=np.int32)).sum()),
                    },
                )
                # endregion
            for p in percentiles:
                doy_idx = center_doy - 1
                if completed_by_p[p][doy_idx]:
                    continue
                source = subset_max if p >= 50.0 else subset_min
                t_q0 = time.perf_counter()
                if idx <= 3:
                    # region agent log
                    _agent_log(
                        run_id,
                        "H6",
                        "build_heatwave_baseline_percentile.py:DOY_Q_START",
                        "before quantile call",
                        {"doy": center_doy, "idx": idx, "percentile": p},
                    )
                    # endregion
                q = source.quantile(p / 100.0, dim="time", skipna=True)
                if idx <= 3:
                    # region agent log
                    _agent_log(
                        run_id,
                        "H6",
                        "build_heatwave_baseline_percentile.py:DOY_Q_OBJ",
                        "quantile graph/object created",
                        {"doy": center_doy, "idx": idx, "percentile": p},
                    )
                    # endregion
                quantile_sums[p] += time.perf_counter() - t_q0
                quantile_counts[p] += 1
                t_vals0 = time.perf_counter()
                if idx <= 3:
                    # region agent log
                    _agent_log(
                        run_id,
                        "H6",
                        "build_heatwave_baseline_percentile.py:DOY_VALUES_START",
                        "before q.values materialization",
                        {"doy": center_doy, "idx": idx, "percentile": p},
                    )
                    # endregion
                vals = q.values.astype(np.float32 if args.output_dtype == "float32" else np.float64)
                if idx <= 3:
                    # region agent log
                    _agent_log(
                        run_id,
                        "H6",
                        "build_heatwave_baseline_percentile.py:DOY_VALUES_DONE",
                        "q.values materialized",
                        {
                            "doy": center_doy,
                            "idx": idx,
                            "percentile": p,
                            "values_s": round(time.perf_counter() - t_vals0, 6),
                        },
                    )
                    # endregion
                t_w0 = time.perf_counter()
                with Dataset(str(p_to_file[p]), "a", format="NETCDF4") as ds_out:
                    ds_out.variables[p_to_var[p]][doy_idx, :, :] = vals
                    ds_out.sync()
                write_sums[p] += time.perf_counter() - t_w0
                if idx <= 3:
                    # region agent log
                    _agent_log(
                        run_id,
                        "H6",
                        "build_heatwave_baseline_percentile.py:DOY_WRITE_DONE",
                        "early DOY write+sync done",
                        {
                            "doy": center_doy,
                            "idx": idx,
                            "percentile": p,
                            "write_s": round(write_sums[p], 6),
                        },
                    )
                    # endregion
                completed_by_p[p][doy_idx] = True

            # Persist checkpoint at DOY granularity for robust resume after kill/OOM.
            ckpt_obj = {
                "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "config": {
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                    "window_days": int(args.window_days),
                    "doy_start": int(args.doy_start),
                    "doy_end": int(args.doy_end),
                    "percentiles": percentiles,
                    "out_dir": str(args.out_dir),
                },
                "completed_doys": {
                    str(p): [i + 1 for i, ok in enumerate(done) if ok]
                    for p, done in completed_by_p.items()
                },
            }
            _atomic_write_json(checkpoint_path, ckpt_obj)
            if (
                idx == 1
                or idx == total_doy
                or (args.progress_every_doy > 0 and idx % args.progress_every_doy == 0)
            ):
                elapsed = time.perf_counter() - t_loop
                _log(
                    f"progress doy={center_doy} ({idx}/{total_doy}) "
                    f"{_bar(idx, total_doy)} elapsed_s={elapsed:.1f}",
                    log_path,
                )
                # region agent log
                _agent_log(
                    run_id,
                    "H2",
                    "build_heatwave_baseline_percentile.py:472",
                    "periodic DOY timing snapshot",
                    {
                        "doy": center_doy,
                        "idx": idx,
                        "total_doy": total_doy,
                        "elapsed_s": round(elapsed, 3),
                        "quantile_avg_s": {
                            str(p): round(quantile_sums[p] / max(1, quantile_counts[p]), 5)
                            for p in percentiles
                        },
                        "write_avg_s": {
                            str(p): round(write_sums[p] / max(1, quantile_counts[p]), 5)
                            for p in percentiles
                        },
                    },
                )
                # endregion

        output_files: list[str] = []
        if args.split_output:
            heatwave_file = args.out_dir / args.heatwave_out_name
            coldwave_file = args.out_dir / args.coldwave_out_name
            heatwave_vars = [p_to_var[p] for p in sorted(percentiles, reverse=True) if p >= 50.0]
            coldwave_vars = [p_to_var[p] for p in sorted(percentiles, reverse=True) if p < 50.0]
            if not heatwave_vars:
                raise ValueError("--split-output requested but no percentile >= 50 provided.")
            if not coldwave_vars:
                raise ValueError("--split-output requested but no percentile < 50 provided.")
            output_files.extend([str(heatwave_file), str(coldwave_file)])
            _log(
                f"wrote split outputs heatwave={heatwave_file} coldwave={coldwave_file}",
                log_path,
            )
        else:
            output_files.append(str(out_file))
            _log(f"wrote output file={out_file}", log_path)

        metadata = {
            "source_dir": str(args.climate_dir),
            "baseline_start": args.start_date,
            "baseline_end": args.end_date,
            "window_days": int(args.window_days),
            "doy_start": int(args.doy_start),
            "doy_end": int(args.doy_end),
            "percentiles": percentiles,
            "source_file_count": len(selected),
            "source_variable_name": var_name,
            "source_units_probe": units_hint,
            "output_variables": [p_to_var[p] for p in sorted(percentiles, reverse=True)],
            "output_file": str(out_file),
            "output_files": output_files,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "celsius_note": (
                "Values stored in degC. If input had units K, conversion was applied before "
                "quantiles; order is immaterial for quantiles (affine transform)."
            ),
            "tail_source_rule": (
                "Percentiles >= 50 use daily max t2m; percentiles < 50 use daily min t2m."
            ),
            "checkpoint_file": str(checkpoint_path),
        }
        with open(args.out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=True)

        print("Baseline generated:")
        if args.split_output:
            print(f"  - heatwave output: {args.out_dir / args.heatwave_out_name}")
            print(f"  - coldwave output: {args.out_dir / args.coldwave_out_name}")
        else:
            print(f"  - output: {out_file}")
        print(f"  - variables: {[p_to_var[p] for p in sorted(percentiles, reverse=True)]}")
        print(f"  - files_used: {len(selected)}")
        print(f"  - source variable: {var_name} (units attr: {units_hint!r})")
        print(f"  - percentiles: {percentiles}")
        _log(
            f"done wall_s={time.perf_counter() - t0:.1f} outputs={output_files}",
            log_path,
        )
        # region agent log
        _agent_log(
            run_id,
            "H1",
            "build_heatwave_baseline_percentile.py:541",
            "run completed",
            {
                "wall_s": round(time.perf_counter() - t0, 3),
                "outputs": output_files,
                "quantile_calls": {str(p): quantile_counts[p] for p in percentiles},
                "quantile_total_s": {str(p): round(quantile_sums[p], 3) for p in percentiles},
                "write_total_s": {str(p): round(write_sums[p], 3) for p in percentiles},
            },
        )
        # endregion
        # Mark checkpoint complete for quick status checks.
        done_ckpt = _load_checkpoint(checkpoint_path)
        done_ckpt["status"] = "done"
        done_ckpt["finished_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        _atomic_write_json(checkpoint_path, done_ckpt)
    finally:
        ds.close()


if __name__ == "__main__":
    main()
