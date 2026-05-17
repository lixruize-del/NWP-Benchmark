#!/usr/bin/env python3
"""Build fused lead-time pseudo-timeseries for object-based verification (v2).

One pass over forecast files for one model, producing for each lead day:
  - pred_tmax_daily.nc
  - pred_tmin_daily.nc

And one shared GT cache (yearly):
  - gt_tmax_c
  - gt_tmin_c
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
from typing import Any

import numpy as np
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # noqa: E402
    from heatwave_common import FORECAST_FILENAME_PATTERN, to_celsius
except ModuleNotFoundError:  # pragma: no cover
    from scripts.heatwave_common import FORECAST_FILENAME_PATTERN, to_celsius
from src.common.data_reader import Era5NpyLayout, load_npy_2d  # noqa: E402
from src.common.repo_paths import nwp_outputs_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read model forecast files once and simultaneously build multiple lead-day "
            "daily Tmax/Tmin pseudo-timeseries."
        )
    )
    p.add_argument(
        "--forecast-root",
        type=Path,
        default=nwp_outputs_dir() / "era5_monthly_202506_v2/forecasts",
    )
    p.add_argument("--model", type=str, default="")
    p.add_argument("--lead-days", type=int, nargs="+", default=[1, 3, 7, 10])
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--start-date", type=str, default="", help="Inclusive target date YYYY-MM-DD")
    p.add_argument("--end-date", type=str, default="", help="Inclusive target date YYYY-MM-DD")
    p.add_argument("--init-hours", type=int, nargs="+", default=[0, 6, 12, 18])
    p.add_argument("--gt-root", type=Path, default=Path("/ecmwf-era5-datasets/era5_np.25"))
    p.add_argument(
        "--gt-cache-file",
        type=Path,
        default=Path(""),
        help="Yearly GT cache with both tmax/tmin.",
    )
    p.add_argument(
        "--no-write-local-gt",
        action="store_true",
        help="Do not copy gt_tmax_daily.nc/gt_tmin_daily.nc into each lead directory.",
    )
    p.add_argument("--skip-missing", action="store_true")
    p.add_argument(
        "--gt-only",
        action="store_true",
        help="Only build/read yearly GT Tmax/Tmin cache; skip forecast scanning.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("reports/heatwave_object_v2"))
    return p.parse_args()


def _date_range(year: int) -> list[date]:
    out: list[date] = []
    cur = date(year, 1, 1)
    end = date(year, 12, 31)
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def _parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _read_forecast_t2m_c_and_coords(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(path)
    try:
        if "t2m" not in ds.data_vars:
            raise KeyError(f"t2m not found in {path}")
        arr = to_celsius(ds["t2m"]).squeeze(drop=True).values.astype(np.float32)
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        lat = ds[lat_name].values.astype(np.float64)
        lon = ds[lon_name].values.astype(np.float64)
    finally:
        ds.close()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D t2m after squeeze, got shape={arr.shape} from {path}")
    return arr, lat, lon


def _normalize_lon_360(lon: np.ndarray) -> np.ndarray:
    out = np.mod(np.asarray(lon, dtype=np.float64), 360.0)
    out[np.isclose(out, 360.0)] = 0.0
    return out


def _regrid_to_target_025(
    arr: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    """Regrid one 2D field to target 0.25-degree grid with robust coord alignment."""
    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_lon = _normalize_lon_360(src_lon)
    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = _normalize_lon_360(target_lon)

    # Fast path: same shape + same coordinates.
    if (
        arr.shape == (target_lat.shape[0], target_lon.shape[0])
        and np.allclose(src_lat, target_lat, atol=1e-10)
        and np.allclose(src_lon, target_lon, atol=1e-10)
    ):
        return arr.astype(np.float32, copy=False)

    da = xr.DataArray(
        arr.astype(np.float32, copy=False),
        dims=("latitude", "longitude"),
        coords={"latitude": src_lat, "longitude": src_lon},
    )

    # Remove duplicate longitudes if present after normalization.
    lon_vals = np.asarray(da["longitude"].values, dtype=np.float64)
    _, uniq_idx = np.unique(lon_vals, return_index=True)
    if uniq_idx.size != lon_vals.size:
        uniq_idx = np.sort(uniq_idx)
        da = da.isel(longitude=uniq_idx)

    # Interpolation requires monotonic coordinates.
    da = da.sortby("latitude").sortby("longitude")
    target_lon_sorted = np.sort(target_lon)
    target_lat_sorted = np.sort(target_lat)

    # Interpolate in sorted coordinates, then restore target orientation.
    interp = da.interp(
        latitude=target_lat_sorted,
        longitude=target_lon_sorted,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

    # Restore exact target ordering (e.g., latitude 90 -> -90).
    interp = interp.sel(latitude=target_lat, longitude=target_lon)
    out = np.asarray(interp.values, dtype=np.float32)
    if out.shape != (target_lat.shape[0], target_lon.shape[0]):
        raise ValueError(f"Regridded shape mismatch: got={out.shape}, target={(target_lat.shape[0], target_lon.shape[0])}")
    return out


def _read_daily_gt_tmax_tmin_c(layout: Era5NpyLayout, dt: date) -> tuple[np.ndarray, np.ndarray]:
    slabs: list[np.ndarray] = []
    for hh in (0, 6, 12, 18):
        t = datetime(dt.year, dt.month, dt.day, hh, 0, 0)
        p = layout.single_path(t, "t2m")
        if not p.exists():
            continue
        slabs.append(load_npy_2d(p, flip_north_south=False).astype(np.float32))
    if not slabs:
        raise FileNotFoundError(f"No GT t2m npy files found for date={dt.isoformat()}")
    stack = np.stack(slabs, axis=0)
    return np.nanmax(stack, axis=0) - 273.15, np.nanmin(stack, axis=0) - 273.15


def _init_pred_buffers(
    lead_days: list[int],
    n_days: int,
    shape: tuple[int, int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    pred_max: dict[int, np.ndarray] = {}
    pred_min: dict[int, np.ndarray] = {}
    filled: dict[int, np.ndarray] = {}
    for ld in lead_days:
        pred_max[ld] = np.full((n_days, shape[0], shape[1]), np.nan, dtype=np.float32)
        pred_min[ld] = np.full((n_days, shape[0], shape[1]), np.nan, dtype=np.float32)
        filled[ld] = np.zeros(n_days, dtype=bool)
    return pred_max, pred_min, filled


def _load_or_build_gt_cache(
    dates: list[date],
    lat: np.ndarray,
    lon: np.ndarray,
    gt_cache_file: Path | None,
    layout: Era5NpyLayout,
    skip_missing: bool,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    n_days = len(dates)
    from_cache = False
    missing_gt_days = 0

    if gt_cache_file is not None and gt_cache_file.exists():
        ds = xr.open_dataset(gt_cache_file)
        try:
            if "tmax_gt_c" in ds.data_vars and "tmin_gt_c" in ds.data_vars:
                tmax = np.asarray(ds["tmax_gt_c"].values, dtype=np.float32)
                tmin = np.asarray(ds["tmin_gt_c"].values, dtype=np.float32)
                if tmax.shape[0] == n_days and tmin.shape[0] == n_days:
                    return tmax, tmin, 0, True
        finally:
            ds.close()

    tmax_list: list[np.ndarray] = []
    tmin_list: list[np.ndarray] = []
    for d in dates:
        try:
            tx, tn = _read_daily_gt_tmax_tmin_c(layout, d)
        except FileNotFoundError:
            if not skip_missing:
                raise
            tx = np.full((lat.shape[0], lon.shape[0]), np.nan, dtype=np.float32)
            tn = np.full((lat.shape[0], lon.shape[0]), np.nan, dtype=np.float32)
            missing_gt_days += 1
        tmax_list.append(tx.astype(np.float32))
        tmin_list.append(tn.astype(np.float32))
    tmax_cube = np.stack(tmax_list, axis=0).astype(np.float32)
    tmin_cube = np.stack(tmin_list, axis=0).astype(np.float32)

    if gt_cache_file is not None:
        gt_cache_file.parent.mkdir(parents=True, exist_ok=True)
        ds_cache = xr.Dataset(
            data_vars={
                "tmax_gt_c": (("time", "latitude", "longitude"), tmax_cube),
                "tmin_gt_c": (("time", "latitude", "longitude"), tmin_cube),
            },
            coords={
                "time": np.asarray([np.datetime64(d.isoformat()) for d in dates], dtype="datetime64[ns]"),
                "latitude": lat,
                "longitude": lon,
            },
            attrs={"description": "Yearly ERA5 GT daily Tmax/Tmin cache", "units": "degC"},
        )
        ds_cache.to_netcdf(gt_cache_file)
    return tmax_cube, tmin_cube, missing_gt_days, from_cache


def main() -> None:
    args = parse_args()
    lead_days = sorted({int(x) for x in args.lead_days})
    invalid = [x for x in lead_days if x <= 0]
    if invalid:
        raise ValueError(f"lead-days must be positive, got {invalid}")

    dates_full = _date_range(int(args.year))
    if args.start_date.strip():
        d0 = _parse_ymd(args.start_date.strip())
    else:
        d0 = dates_full[0]
    if args.end_date.strip():
        d1 = _parse_ymd(args.end_date.strip())
    else:
        d1 = dates_full[-1]
    if d1 < d0:
        raise ValueError("--end-date must be >= --start-date")
    dates = [d for d in dates_full if d0 <= d <= d1]
    if not dates:
        raise ValueError("No target dates selected after applying --start-date/--end-date")
    date_to_idx = {d: i for i, d in enumerate(dates)}
    init_hour_set = {int(h) for h in args.init_hours}

    if not args.gt_only and not args.model.strip():
        raise ValueError("--model is required unless --gt-only is set")

    if args.gt_only:
        gt_cache_file = Path(args.gt_cache_file) if str(args.gt_cache_file).strip() else (args.out_dir / f"gt_tmax_daily_{int(args.year)}.nc")
        lat_vals = np.linspace(90.0, -90.0, 721, dtype=np.float64)
        lon_vals = np.linspace(0.0, 360.0, 1440, endpoint=False, dtype=np.float64)
        gt_layout = Era5NpyLayout(args.gt_root)
        _, _, missing_gt_days, gt_from_cache = _load_or_build_gt_cache(
            dates=dates,
            lat=lat_vals,
            lon=lon_vals,
            gt_cache_file=gt_cache_file,
            layout=gt_layout,
            skip_missing=bool(args.skip_missing),
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with open(args.out_dir / "gt_cache_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": "gt_only",
                    "year": int(args.year),
                    "start_date": d0.isoformat(),
                    "end_date": d1.isoformat(),
                    "gt_root": str(args.gt_root),
                    "gt_cache_file": str(gt_cache_file),
                    "gt_from_cache": bool(gt_from_cache),
                    "missing_gt_days": int(missing_gt_days),
                    "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                },
                f,
                ensure_ascii=True,
                indent=2,
            )
        print("GT cache ready:")
        print(f"  - gt_cache: {gt_cache_file}")
        print(f"  - from_cache: {gt_from_cache}")
        print(f"  - missing_gt_days: {missing_gt_days}")
        return

    model_dir = args.forecast_root / args.model
    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir}")

    lat_vals: np.ndarray | None = None
    lon_vals: np.ndarray | None = None
    pred_max: dict[int, np.ndarray] = {}
    pred_min: dict[int, np.ndarray] = {}
    filled: dict[int, np.ndarray] = {}

    used_forecast_files = 0
    missing_init_dirs = 0

    # Target grid: prefer GT cache coordinates when available.
    gt_cache_file = Path(args.gt_cache_file) if str(args.gt_cache_file).strip() else None
    if gt_cache_file is not None and gt_cache_file.exists():
        ds_cache = xr.open_dataset(gt_cache_file)
        try:
            c_lat_name = "latitude" if "latitude" in ds_cache.coords else "lat"
            c_lon_name = "longitude" if "longitude" in ds_cache.coords else "lon"
            lat_vals = ds_cache[c_lat_name].values.astype(np.float64)
            lon_vals = _normalize_lon_360(ds_cache[c_lon_name].values.astype(np.float64))
        finally:
            ds_cache.close()

    # Read each forecast file at most once for this model.
    def gather_init_dirs(base_dir: Path | None) -> dict[str, Path]:
        if not base_dir or not base_dir.exists():
            return {}
        res = {}
        for d in base_dir.iterdir():
            if not d.is_dir():
                continue
            name = d.name.strip()
            if len(name) != 10 or not name.isdigit():
                continue
            res[name] = d
        return res

    init_dirs = gather_init_dirs(model_dir)

    for name in sorted(init_dirs.keys()):
        init_dt = datetime.strptime(name, "%Y%m%d%H")
        if init_dt.hour not in init_hour_set:
            continue
        init_date = init_dt.date()

        init_dir = init_dirs[name]
        for f in sorted(init_dir.glob("*.nc")):
            m = FORECAST_FILENAME_PATTERN.match(f.name)
            if not m:
                continue
            fname = f.name

            lead_hours = int(m.group(3))
            valid_dt = init_dt + timedelta(hours=lead_hours)
            valid_date = valid_dt.date()
            if valid_date.year != int(args.year):
                continue
            lead_day = (valid_date - init_date).days + 1
            if lead_day not in lead_days:
                continue
            if valid_date not in date_to_idx:
                continue
            day_idx = date_to_idx[valid_date]

            arr, lat, lon = _read_forecast_t2m_c_and_coords(f)
            used_forecast_files += 1
            if lat_vals is None or lon_vals is None:
                # Default 0.25° grid when GT cache is not pre-existing.
                lat_vals = np.linspace(90.0, -90.0, 721, dtype=np.float64)
                lon_vals = np.linspace(0.0, 360.0, 1440, endpoint=False, dtype=np.float64)
            arr = _regrid_to_target_025(arr, lat, lon, lat_vals, lon_vals)
            if not pred_max:
                pred_max, pred_min, filled = _init_pred_buffers(lead_days, len(dates), arr.shape)

            if not filled[lead_day][day_idx]:
                pred_max[lead_day][day_idx, :, :] = arr
                pred_min[lead_day][day_idx, :, :] = arr
                filled[lead_day][day_idx] = True
            else:
                np.maximum(pred_max[lead_day][day_idx, :, :], arr, out=pred_max[lead_day][day_idx, :, :])
                np.minimum(pred_min[lead_day][day_idx, :, :], arr, out=pred_min[lead_day][day_idx, :, :])

    if lat_vals is None or lon_vals is None:
        raise RuntimeError(f"No usable forecast files found for model={args.model}")

    # Build/read GT once (both Tmax and Tmin).
    gt_cache_file = Path(args.gt_cache_file) if str(args.gt_cache_file).strip() else None
    gt_layout = Era5NpyLayout(args.gt_root)
    gt_tmax, gt_tmin, missing_gt_days, gt_from_cache = _load_or_build_gt_cache(
        dates=dates,
        lat=lat_vals,
        lon=lon_vals,
        gt_cache_file=gt_cache_file,
        layout=gt_layout,
        skip_missing=bool(args.skip_missing),
    )

    base_out = args.out_dir / args.model
    base_out.mkdir(parents=True, exist_ok=True)
    time_vals = np.asarray([np.datetime64(d.isoformat()) for d in dates], dtype="datetime64[ns]")

    per_lead_meta: list[dict[str, Any]] = []
    for ld in lead_days:
        if ld not in pred_max:
            # No data for this lead; initialize NaN cube.
            shape = (len(dates), lat_vals.shape[0], lon_vals.shape[0])
            pred_max_ld = np.full(shape, np.nan, dtype=np.float32)
            pred_min_ld = np.full(shape, np.nan, dtype=np.float32)
            filled_days = 0
        else:
            pred_max_ld = pred_max[ld]
            pred_min_ld = pred_min[ld]
            filled_days = int(filled[ld].sum())

        missing_pred_days = int(len(dates) - filled_days)
        if missing_pred_days > 0 and not args.skip_missing:
            raise RuntimeError(
                f"Lead {ld}: missing pred days={missing_pred_days}. Re-run with --skip-missing if expected."
            )

        out_dir = base_out / f"lead_day_{ld}" / "step1"
        out_dir.mkdir(parents=True, exist_ok=True)

        pred_tmax_file = out_dir / "pred_tmax_daily.nc"
        pred_tmin_file = out_dir / "pred_tmin_daily.nc"
        gt_tmax_file = out_dir / "gt_tmax_daily.nc"
        gt_tmin_file = out_dir / "gt_tmin_daily.nc"
        meta_file = out_dir / "meta.json"

        xr.Dataset(
            data_vars={"tmax_pred_c": (("time", "latitude", "longitude"), pred_max_ld)},
            coords={"time": time_vals, "latitude": lat_vals, "longitude": lon_vals},
            attrs={"description": "Pred daily Tmax", "model": args.model, "lead_day": int(ld), "units": "degC"},
        ).to_netcdf(pred_tmax_file)
        xr.Dataset(
            data_vars={"tmin_pred_c": (("time", "latitude", "longitude"), pred_min_ld)},
            coords={"time": time_vals, "latitude": lat_vals, "longitude": lon_vals},
            attrs={"description": "Pred daily Tmin", "model": args.model, "lead_day": int(ld), "units": "degC"},
        ).to_netcdf(pred_tmin_file)

        if not args.no_write_local_gt:
            xr.Dataset(
                data_vars={"tmax_gt_c": (("time", "latitude", "longitude"), gt_tmax)},
                coords={"time": time_vals, "latitude": lat_vals, "longitude": lon_vals},
                attrs={"description": "GT daily Tmax", "year": int(args.year), "units": "degC"},
            ).to_netcdf(gt_tmax_file)
            xr.Dataset(
                data_vars={"tmin_gt_c": (("time", "latitude", "longitude"), gt_tmin)},
                coords={"time": time_vals, "latitude": lat_vals, "longitude": lon_vals},
                attrs={"description": "GT daily Tmin", "year": int(args.year), "units": "degC"},
            ).to_netcdf(gt_tmin_file)

        meta = {
            "model": args.model,
            "lead_day": int(ld),
            "year": int(args.year),
            "start_date": d0.isoformat(),
            "end_date": d1.isoformat(),
            "used_forecast_files_total": int(used_forecast_files),
            "filled_pred_days": int(filled_days),
            "missing_pred_days": int(missing_pred_days),
            "missing_gt_days": int(missing_gt_days),
            "pred_tmax_file": str(pred_tmax_file),
            "pred_tmin_file": str(pred_tmin_file),
            "gt_tmax_file": str(gt_tmax_file) if not args.no_write_local_gt else "",
            "gt_tmin_file": str(gt_tmin_file) if not args.no_write_local_gt else "",
            "gt_cache_file": str(gt_cache_file) if gt_cache_file is not None else "",
            "gt_from_cache": bool(gt_from_cache),
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)
        per_lead_meta.append(meta)

    run_meta = {
        "model": args.model,
        "year": int(args.year),
        "start_date": d0.isoformat(),
        "end_date": d1.isoformat(),
        "lead_days": lead_days,
        "forecast_root": str(args.forecast_root),
        "gt_root": str(args.gt_root),
        "init_hours": sorted(init_hour_set),
        "used_forecast_files_total": int(used_forecast_files),
        "gt_cache_file": str(gt_cache_file) if gt_cache_file is not None else "",
        "gt_from_cache": bool(gt_from_cache),
        "missing_gt_days": int(missing_gt_days),
        "per_lead": per_lead_meta,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with open(base_out / "step1_fused_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=True, indent=2)

    print("Fused Step1 built:")
    print(f"  - model: {args.model}")
    print(f"  - lead_days: {lead_days}")
    print(f"  - out_root: {base_out}")
    if gt_cache_file is not None:
        print(f"  - gt_cache: {gt_cache_file}")
    print(f"  - used_forecast_files_total: {used_forecast_files}")


if __name__ == "__main__":
    main()

