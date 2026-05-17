#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.data_reader_ifs import Era5NpyLayout, load_npy_2d
from src.common.repo_paths import nwp_outputs_dir

_LEAD_RE = re.compile(r"-(\d{2,3})(?:_\d+)?\.nc$")
_STATION_LATLON_CACHE: dict[str, dict[str, dict[str, float]]] = {}
METRIC_COLS = ["init_time", "valid_time", "lead_hours", "variable", "wrmse", "bias", "mae", "activity", "acc"]
CLIM_LAT_721 = np.linspace(90.0, -90.0, 721, dtype=np.float64)
CLIM_LON_1440 = np.linspace(0.0, 360.0, 1440, endpoint=False, dtype=np.float64)


def _parse_linux_cpu_list(count_str: str) -> int:
    """Parse sysfs ranges like '0-52' or '0-7,16-23' -> number of CPUs."""
    total = 0
    for part in count_str.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            total += int(b) - int(a) + 1
        else:
            total += 1
    return total


def usable_cpu_count(env_override: int | None = None) -> int:
    """
    CPUs for sizing parallel pools: optional override/env, then min(affinity, sysfs online)
    so cgroup limits and physically offline cores both apply; fallback os.cpu_count().
    """
    if env_override is not None and env_override > 0:
        return env_override
    env_n = os.environ.get("STATION_EVAL_CPU_CAP")
    if env_n is not None:
        try:
            v = int(env_n.strip())
            if v > 0:
                return v
        except ValueError:
            pass

    aff: int | None = None
    try:
        na = len(os.sched_getaffinity(0))
        if na > 0:
            aff = na
    except Exception:
        pass

    onl: int | None = None
    try:
        online_path = Path("/sys/devices/system/cpu/online")
        if online_path.exists():
            no = _parse_linux_cpu_list(online_path.read_text())
            if no > 0:
                onl = no
    except Exception:
        pass

    if aff is not None and onl is not None:
        return max(1, min(aff, onl))
    if aff is not None:
        return max(1, aff)
    if onl is not None:
        return max(1, onl)
    return max(1, os.cpu_count() or 8)


@dataclass(frozen=True)
class WorkerConfig:
    station_root: str
    era5_root: str
    eval_vars: tuple[str, ...]
    use_climatology: bool
    station_latlon_json: str | None


@dataclass(frozen=True)
class StationEvalModelJob:
    """One model worth of work; safe to run in a separate process."""

    model: str
    forecast_roots: tuple[str, ...]
    out_root: str
    by_init_dir: str | None
    write_by_init: bool
    station_root: str
    era5_root: str
    eval_vars: tuple[str, ...]
    use_climatology: bool
    station_latlon_json: str | None
    expected_inits: tuple[datetime, ...]
    max_lead: int
    max_workers: int
    resume: bool
    replace_vars_list: tuple[str, ...]
    flush_every_samples: int


def _default_model_parallel_jobs(n_models: int, max_workers: int, cpu_cap: int | None) -> int:
    cpu = usable_cpu_count(cpu_cap)
    # Bound concurrent models so total worker processes stay near usable CPUs.
    return min(n_models, max(1, cpu // max(1, max_workers)))


def _station_eval_run_one_model(job: StationEvalModelJob) -> None:
    cfg = WorkerConfig(
        station_root=job.station_root,
        era5_root=job.era5_root,
        eval_vars=job.eval_vars,
        use_climatology=job.use_climatology,
        station_latlon_json=job.station_latlon_json,
    )
    replace_vars_list = list(job.replace_vars_list)
    model = job.model
    forecast_roots = [Path(r) for r in job.forecast_roots]
    out_root = Path(job.out_root)
    out_csv = out_root / f"{model}_station_metrics.csv"
    by_init_dir = (Path(job.by_init_dir) if job.by_init_dir else None) if job.write_by_init else None

    if replace_vars_list:
        df_all = _load_existing_metrics(out_csv)
        df_all = df_all[~df_all["variable"].isin(replace_vars_list)].reset_index(drop=True)
        done_keys: set[tuple[str, int]] = set()
    elif job.resume:
        df_all = _load_existing_metrics(out_csv)
        done_keys = _completed_sample_keys(df_all)
    else:
        df_all = pd.DataFrame(columns=METRIC_COLS)
        done_keys = set()

    expected_inits = list(job.expected_inits)
    tasks = _collect_model_tasks_from_roots(forecast_roots, model, expected_inits, job.max_lead)
    if job.resume and done_keys and not replace_vars_list:
        total_before = len(tasks)
        tasks = [t for t in tasks if (t[0], int(t[1])) not in done_keys]
        skipped = total_before - len(tasks)
        if skipped > 0:
            print(f"[station-eval] model={model} resume_skip_samples={skipped}")
    if not tasks:
        print(f"[station-eval] model={model}: no matching forecast nc tasks")
        return

    print(
        f"[station-eval] model={model} pending_tasks={len(tasks)} "
        f"workers={job.max_workers} resume={job.resume} "
        f"replace_vars={replace_vars_list or None} eval_vars={list(cfg.eval_vars)}"
    )
    flush_every = max(1, int(job.flush_every_samples))
    completed_now = 0
    touched_since_flush: set[str] = set()
    with ProcessPoolExecutor(max_workers=max(1, int(job.max_workers))) as executor:
        fut_map = {
            executor.submit(
                _worker_one_file,
                model,
                init_str,
                init_str,
                lead,
                str(nc_path),
                cfg,
            ): (init_str, int(lead))
            for init_str, lead, nc_path in tasks
        }
        for fut in as_completed(fut_map):
            init_str, _lead = fut_map[fut]
            try:
                rows = fut.result()
            except Exception as e:
                print(f"[station-eval] model={model} sample_failed init={init_str} err={type(e).__name__}: {e}")
                continue
            if rows:
                df_new = pd.DataFrame(rows, columns=METRIC_COLS)
                if df_all.empty:
                    df_all = df_new.copy()
                else:
                    df_all = pd.concat([df_all, df_new], ignore_index=True)
                df_all = _normalize_metrics_df(df_all)
                touched_since_flush.add(init_str)
            completed_now += 1
            if completed_now % flush_every == 0:
                _flush_outputs(model, df_all, out_csv, by_init_dir, touched_since_flush)
                print(
                    f"[station-eval] model={model} flushed samples={completed_now}/{len(tasks)} "
                    f"rows={len(df_all)}"
                )
                touched_since_flush.clear()

    _flush_outputs(model, df_all, out_csv, by_init_dir, touched_since_flush if touched_since_flush else None)
    print(f"[station-eval] model={model} done rows={len(df_all)} out={out_csv}")


def _iter_inits(start: str, end: str, init_hours: Iterable[int]) -> list[datetime]:
    st = datetime.strptime(start, "%Y-%m-%d")
    ed = datetime.strptime(end, "%Y-%m-%d").replace(hour=23, minute=59)
    out: list[datetime] = []
    cur = st.date()
    while cur <= ed.date():
        for h in sorted(set(int(x) for x in init_hours)):
            dt = datetime(cur.year, cur.month, cur.day, h, 0, 0)
            if st <= dt <= ed:
                out.append(dt)
        cur += timedelta(days=1)
    return out


def _lead_from_name(name: str) -> int | None:
    m = _LEAD_RE.search(name)
    if not m:
        return None
    return int(m.group(1))


def _station_file_path(station_root: Path, valid_time: datetime) -> Path:
    return station_root / f"{valid_time.strftime('%Y-%m-%d %H:%M:%S')}.nc"


def _load_station_latlon_map(path: str | None) -> dict[str, dict[str, float]] | None:
    if not path:
        return None
    cached = _STATION_LATLON_CACHE.get(path)
    if cached is not None:
        return cached
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        return None
    _STATION_LATLON_CACHE[path] = obj
    return obj


def _normalize_query_lon_for_grid(qlon: np.ndarray, lon_1d: np.ndarray) -> np.ndarray:
    """
    Align station/query longitudes with the forecast/climatology longitude axis.

    Station catalogs and many observation files use [-180, 180]. ERA5 and many
    saved model grids use [0, 360). Negative western longitudes must be mapped
    (e.g. -120° -> 240°) before bilinear indexing on a 0–360° axis.
    """
    qlon = np.asarray(qlon, dtype=np.float64)
    lon_min = float(np.nanmin(lon_1d))
    lon_max = float(np.nanmax(lon_1d))
    # ERA5-style [0, 360), including 0 … 357.5 with lon_min == 0
    if lon_max > 180.0 or lon_min >= 0.0:
        q = np.mod(qlon, 360.0)
    else:
        # Signed [-180, 180) grid (wrap inputs into the same range)
        q = np.mod(qlon + 180.0, 360.0) - 180.0

    lon0 = float(lon_1d[0])
    if not np.isclose(lon0, 0.0):
        q = np.mod(q - lon0, 360.0) + lon0
    return q


def _resolve_station_latlon(
    ds_stn: xr.Dataset,
    station_latlon_map: dict[str, dict[str, float]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    stn_ids = np.asarray(ds_stn["station"].values).astype(str)
    lat_nc = _to_numpy_1d(ds_stn["latitude"])
    lon_nc = _to_numpy_1d(ds_stn["longitude"])

    if station_latlon_map is None:
        return lat_nc, lon_nc

    lat = np.full(stn_ids.shape, np.nan, dtype=np.float64)
    lon = np.full(stn_ids.shape, np.nan, dtype=np.float64)
    for i, sid in enumerate(stn_ids):
        rec = station_latlon_map.get(sid)
        if rec is None:
            continue
        try:
            lat[i] = float(rec["latitude"])
            lon[i] = float(rec["longitude"])
        except Exception:
            continue

    miss = ~np.isfinite(lat) | ~np.isfinite(lon)
    if np.any(miss):
        lat[miss] = lat_nc[miss]
        lon[miss] = lon_nc[miss]
    # Longitudes remain in catalog convention [-180, 180]; interpolation maps to grid.
    return lat, lon


def _to_numpy_1d(da: xr.DataArray) -> np.ndarray:
    arr = np.asarray(da.values)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float64, copy=False)


def _extract_station_target(ds: xr.Dataset, var: str) -> np.ndarray:
    if var in ds.data_vars:
        return _to_numpy_1d(ds[var])

    if var == "t2m":
        if "temperature" in ds.data_vars:
            return _to_numpy_1d(ds["temperature"])
    elif var == "msl":
        if "sea_level_pressure" in ds.data_vars:
            return _to_numpy_1d(ds["sea_level_pressure"])
    elif var in ("u10", "v10"):
        if var in ds.data_vars:
            return _to_numpy_1d(ds[var])
        if "wind_speed" in ds.data_vars and "wind_direction" in ds.data_vars:
            speed = _to_numpy_1d(ds["wind_speed"])
            direction_deg = _to_numpy_1d(ds["wind_direction"])
            direction_rad = np.deg2rad(direction_deg)
            # Meteorological convention: direction is where wind comes FROM.
            u = -speed * np.sin(direction_rad)
            v = -speed * np.cos(direction_rad)
            return u if var == "u10" else v

    raise KeyError(f"station variable mapping missing for {var}")


def _prepare_grid(lat: np.ndarray, lon: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_a = np.asarray(lat, dtype=np.float64)
    lon_a = np.asarray(lon, dtype=np.float64)
    f = np.asarray(field, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"field must be 2D, got {f.shape}")
    if lat_a.ndim != 1 or lon_a.ndim != 1:
        raise ValueError("lat/lon must be 1D")
    if f.shape != (lat_a.size, lon_a.size):
        raise ValueError(f"field shape mismatch {f.shape} vs ({lat_a.size}, {lon_a.size})")

    if lat_a[0] > lat_a[-1]:
        lat_a = lat_a[::-1]
        f = f[::-1, :]
    return lat_a, lon_a, f


def _bilinear_interp_points(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    field: np.ndarray,
    qlat: np.ndarray,
    qlon: np.ndarray,
) -> np.ndarray:
    lat, lon, z = _prepare_grid(lat_grid, lon_grid, field)
    qlat = np.asarray(qlat, dtype=np.float64)
    qlon = np.asarray(qlon, dtype=np.float64)
    out = np.full(qlat.shape, np.nan, dtype=np.float64)

    if lat.size < 2 or lon.size < 2:
        return out

    lon0 = float(lon[0])
    dlon = float(lon[1] - lon[0])
    if not np.isfinite(dlon) or dlon <= 0:
        raise ValueError("invalid longitude grid spacing")

    qlon_use = _normalize_query_lon_for_grid(qlon, lon)

    in_lat = (qlat >= lat[0]) & (qlat <= lat[-1])
    finite_q = np.isfinite(qlat) & np.isfinite(qlon_use) & in_lat
    if not np.any(finite_q):
        return out

    idx = np.where(finite_q)[0]
    qy = qlat[idx]
    qx = qlon_use[idx]

    i = np.searchsorted(lat, qy, side="right") - 1
    i = np.clip(i, 0, lat.size - 2)
    lat0 = lat[i]
    lat1 = lat[i + 1]
    wy = (qy - lat0) / (lat1 - lat0)

    j = np.floor((qx - lon0) / dlon).astype(np.int64)
    j = np.mod(j, lon.size)
    j1 = (j + 1) % lon.size
    lonj = lon0 + j * dlon
    wx = (qx - lonj) / dlon
    wx = np.clip(wx, 0.0, 1.0)

    f00 = z[i, j]
    f01 = z[i, j1]
    f10 = z[i + 1, j]
    f11 = z[i + 1, j1]

    interp = (
        (1.0 - wy) * (1.0 - wx) * f00
        + (1.0 - wy) * wx * f01
        + wy * (1.0 - wx) * f10
        + wy * wx * f11
    )
    out[idx] = interp
    return out


def _to_si_temperature_k(a: np.ndarray) -> np.ndarray:
    """Normalize to kelvin. Station archives use °C (median magnitude « 100). ERA5/model use K."""
    x = np.asarray(a, dtype=np.float64)
    finite = x[np.isfinite(x)]
    med = float(np.nanmedian(finite)) if finite.size else float("nan")
    if np.isfinite(med) and med < 100.0:
        return x + 273.15
    return x


def _convert_units_if_needed(
    var: str,
    pred: np.ndarray,
    gt: np.ndarray,
    clim: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Express obs/forecast/clim in SI before metrics: t2m [K], msl [Pa].

    msl: forecasts and ERA5 climatology are **Pa**. Station values use **hPa → Pa** only when the
    median magnitude looks like hPa (< 5000); otherwise the station field is treated as **already Pa**
    (some processed NC follow CF / Pa).
    """
    p = pred.astype(np.float64, copy=False)
    g = gt.astype(np.float64, copy=False)
    c = clim.astype(np.float64, copy=False) if clim is not None else None

    if var == "t2m":
        p = _to_si_temperature_k(p)
        g = _to_si_temperature_k(g)
        if c is not None:
            c = _to_si_temperature_k(c)
    elif var == "msl":
        # Forecast NetCDF from this repo: sea-level pressure in Pa.
        p = np.asarray(p, dtype=np.float64)
        # Station files may be **hPa** (~960–1050) or already **Pa** (~1e5); ERA5 clim npy is Pa.
        g_raw = np.asarray(g, dtype=np.float64)
        finite_g = g_raw[np.isfinite(g_raw)]
        med_g = float(np.nanmedian(finite_g)) if finite_g.size else float("nan")
        if np.isfinite(med_g) and med_g < 5000.0:
            g = g_raw * 100.0  # hPa → Pa
        else:
            g = g_raw  # already Pa (do not scale)
        if c is not None:
            c = np.asarray(c, dtype=np.float64)

    return p, g, c


def _compute_station_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    clim: np.ndarray | None,
) -> dict[str, float]:
    mask = np.isfinite(pred) & np.isfinite(gt)
    if clim is not None:
        mask = mask & np.isfinite(clim)
    if not np.any(mask):
        return {
            "wrmse": float("nan"),
            "bias": float("nan"),
            "mae": float("nan"),
            "activity": float("nan"),
            "acc": float("nan"),
        }

    p = pred[mask]
    g = gt[mask]
    err = p - g
    wrmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    mae = float(np.mean(np.abs(err)))

    activity = float("nan")
    acc = float("nan")
    if clim is not None:
        c = clim[mask]
        fa = p - c
        oa = g - c
        activity = float(np.sqrt(np.mean(fa**2)))
        denom = math.sqrt(float(np.mean(fa**2)) * float(np.mean(oa**2)))
        if denom > 1e-12:
            acc = float(np.mean(fa * oa) / denom)
    return {
        "wrmse": wrmse,
        "bias": bias,
        "mae": mae,
        "activity": activity,
        "acc": acc,
    }


def _load_station_dataset(path: Path) -> xr.Dataset:
    return xr.open_dataset(path)


def _load_clim_2d(var: str, valid_time: datetime, era5_root: Path) -> np.ndarray | None:
    mmdd = valid_time.strftime("%m-%d")
    layout = Era5NpyLayout(era5_root)
    path = layout.root / "climate_mean_day" / "single" / "1993-2016" / mmdd / f"{var}.npy"
    if not path.exists():
        return None
    return load_npy_2d(path, flip_north_south=False).astype(np.float64)


def _worker_one_file(
    model: str,
    init_str: str,
    init_time_s: str,
    lead: int,
    nc_path_s: str,
    cfg: WorkerConfig,
) -> list[dict]:
    rows: list[dict] = []
    init_time = datetime.strptime(init_time_s, "%Y%m%d%H")
    valid_time = init_time + timedelta(hours=int(lead))
    valid_str = valid_time.strftime("%Y-%m-%d %H:%M:%S")

    station_path = _station_file_path(Path(cfg.station_root), valid_time)
    if not station_path.exists():
        return rows

    stn_map = _load_station_latlon_map(cfg.station_latlon_json)
    with xr.open_dataset(nc_path_s) as ds_pred, _load_station_dataset(station_path) as ds_stn:
        lat_grid = np.asarray(ds_pred["latitude"].values, dtype=np.float64)
        lon_grid = np.asarray(ds_pred["longitude"].values, dtype=np.float64)
        stn_lat, stn_lon = _resolve_station_latlon(ds_stn, stn_map)

        clim_cache: dict[str, np.ndarray | None] = {}
        for var in cfg.eval_vars:
            if var not in ds_pred.data_vars:
                continue
            try:
                gt_raw = _extract_station_target(ds_stn, var)
            except Exception:
                continue

            pred_da = ds_pred[var]
            if "time" in pred_da.dims:
                pred_da = pred_da.isel(time=0)
            pred_2d = np.asarray(pred_da.values, dtype=np.float64)
            pred_stn = _bilinear_interp_points(lat_grid, lon_grid, pred_2d, stn_lat, stn_lon)

            clim_stn: np.ndarray | None = None
            if cfg.use_climatology:
                if var not in clim_cache:
                    clim_cache[var] = _load_clim_2d(var, valid_time, Path(cfg.era5_root))
                clim_2d = clim_cache[var]
                if clim_2d is not None:
                    # Climatology is always on ERA5 0.25deg (721x1440). Regrid it to
                    # stations directly on that native grid; do not assume model grid.
                    clim_stn = _bilinear_interp_points(
                        CLIM_LAT_721,
                        CLIM_LON_1440,
                        clim_2d,
                        stn_lat,
                        stn_lon,
                    )

            pred_stn, gt_stn, clim_stn = _convert_units_if_needed(var, pred_stn, gt_raw, clim_stn)
            m = _compute_station_metrics(pred_stn, gt_stn, clim_stn)
            rows.append(
                {
                    "init_time": init_str,
                    "valid_time": valid_str,
                    "lead_hours": int(lead),
                    "variable": var,
                    "wrmse": m["wrmse"],
                    "bias": m["bias"],
                    "mae": m["mae"],
                    "activity": m["activity"],
                    "acc": m["acc"],
                }
            )
    return rows


def _collect_model_tasks(
    model_root: Path,
    expected_inits: list[datetime],
    max_lead: int,
) -> list[tuple[str, int, Path]]:
    out: list[tuple[str, int, Path]] = []
    for it in expected_inits:
        init_str = it.strftime("%Y%m%d%H")
        init_dir = model_root / init_str
        if not init_dir.exists():
            continue
        pairs: list[tuple[int, Path]] = []
        for p in init_dir.glob("*.nc"):
            lead = _lead_from_name(p.name)
            if lead is None:
                continue
            if lead < 6 or lead > max_lead or lead % 6 != 0:
                continue
            pairs.append((lead, p))
        pairs.sort(key=lambda x: x[0])
        out.extend([(init_str, lead, p) for lead, p in pairs])
    return out


def _collect_model_tasks_from_roots(
    forecast_roots: list[Path],
    model: str,
    expected_inits: list[datetime],
    max_lead: int,
) -> list[tuple[str, int, Path]]:
    """Merge tasks from several roots: first root wins per (init_time, lead_hours)."""
    seen: set[tuple[str, int]] = set()
    out: list[tuple[str, int, Path]] = []
    for root in forecast_roots:
        mr = root / model
        if not mr.exists():
            continue
        for init_str, lead, p in _collect_model_tasks(mr, expected_inits, max_lead):
            k = (init_str, int(lead))
            if k in seen:
                continue
            seen.add(k)
            out.append((init_str, lead, p))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _normalize_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in METRIC_COLS:
        if c not in out.columns:
            out[c] = np.nan
    out = out[METRIC_COLS]
    if not df.empty:
        out = out.drop_duplicates(subset=["init_time", "lead_hours", "variable"], keep="last")
        out = out.sort_values(by=["init_time", "lead_hours", "variable"]).reset_index(drop=True)
    return out


def _load_existing_metrics(output_csv: Path) -> pd.DataFrame:
    if not output_csv.exists():
        return pd.DataFrame(columns=METRIC_COLS)
    try:
        df = pd.read_csv(output_csv)
    except Exception:
        return pd.DataFrame(columns=METRIC_COLS)
    return _normalize_metrics_df(df)


def _completed_sample_keys(df: pd.DataFrame) -> set[tuple[str, int]]:
    if df.empty:
        return set()
    keys = set()
    for _, r in df[["init_time", "lead_hours"]].drop_duplicates().iterrows():
        keys.add((str(r["init_time"]), int(r["lead_hours"])))
    return keys


def _flush_outputs(
    model: str,
    df: pd.DataFrame,
    output_csv: Path,
    by_init_dir: Path | None,
    touched_inits: set[str] | None = None,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if by_init_dir is not None:
        by_init_dir.mkdir(parents=True, exist_ok=True)
        if touched_inits is None:
            iters = [str(x) for x in df["init_time"].drop_duplicates().tolist()]
        else:
            iters = sorted(touched_inits)
        for init_time in iters:
            sub = df[df["init_time"].astype(str) == str(init_time)]
            sub.to_csv(by_init_dir / f"{model}_{init_time}.csv", index=False)


_ARG_EPILOG = """
Examples (from repo root, conda env nwp_unified):
  # Three forecast trees, 7 models, cap ~53 online CPUs, 8 sample workers per model
  export OMP_NUM_THREADS=1
  python scripts/eval_station_metrics_from_saved_ifs_nc.py \\
    --forecasts-roots \\
      nwp_outputs/ifs_monthly_202506_v2/forecasts \\
      /path/to/mirror/ifs_monthly_202506_v2/forecasts \\
      /path/to/mirror/era5_monthly_202506_v2/forecasts \\
    --models aifs aurora fuxi fengwu pangu graphcast stormer \\
    --start 2025-01-01 --end 2025-12-31 --max-lead 240 --max-workers 8 --cpu-cap 53 \\
    --out-root nwp_outputs/ifs_monthly_202506_v2/metrics_station \\
    --by-init-dir nwp_outputs/ifs_monthly_202506_v2/metrics_station/by_init --write-by-init \\
    --flush-every-samples 500 --resume

  # MSL-only replace; 3 models at a time, 10 workers each (3*10<=~53); repeat full --forecasts-roots list
  python scripts/eval_station_metrics_from_saved_ifs_nc.py --replace-vars msl \\
    --forecasts-roots /path/ifs.../forecasts /path/nas_ifs.../forecasts /path/nas_era5.../forecasts \\
    --models fengwu fuxi pangu graphcast aifs aurora stormer \\
    --start 2025-01-01 --end 2025-12-31 --max-workers 10 --model-parallel-jobs 3 --cpu-cap 53 \\
    --out-root nwp_outputs/ifs_monthly_202506_v2/metrics_station --resume --flush-every-samples 500
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate station metrics from saved IFS-driven forecast NetCDF files.",
        epilog=_ARG_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--forecasts-root",
        type=Path,
        default=nwp_outputs_dir() / "ifs_monthly_202506_v2" / "forecasts",
        help="Single forecast tree forecasts/<model>/<init>/*.nc (used if --forecasts-roots not set).",
    )
    ap.add_argument(
        "--forecasts-roots",
        type=Path,
        nargs="+",
        default=None,
        metavar="DIR",
        help=(
            "Multiple forecast directories in decreasing priority: first root wins for each "
            "(init, lead). Use to merge VEPFS IFS + NAS-moved IFS + ERA5-input forecasts. "
            "When set, overrides --forecasts-root."
        ),
    )
    ap.add_argument(
        "--station-root",
        type=Path,
        default=Path(os.environ.get("NWP_STATION_ROOT", "data/stations/processed/2025")),
    )
    ap.add_argument(
        "--era5-root",
        type=Path,
        default=Path("/ecmwf-era5-datasets/era5_np.25"),
        help="Used only for climatology needed by activity/acc.",
    )
    ap.add_argument("--models", type=str, nargs="+", required=True)
    ap.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--init-hours", type=int, nargs="+", default=[0, 6, 12, 18])
    ap.add_argument("--max-lead", type=int, default=240)
    ap.add_argument("--eval-vars", type=str, nargs="+", default=["u10", "v10", "msl", "t2m"])
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument(
        "--station-latlon-json",
        type=Path,
        default=Path(os.environ.get("NWP_STATION_LATLON_JSON", "data/stations/station_latlon_2025.json")),
        help="Station lat/lon lookup JSON keyed by station id.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=nwp_outputs_dir() / "ifs_monthly_202506_v2" / "metrics_station",
    )
    ap.add_argument(
        "--by-init-dir",
        type=Path,
        default=None,
        help="Explicit by_init output dir. Default: <out-root>/by_init",
    )
    ap.add_argument("--write-by-init", dest="write_by_init", action="store_true")
    ap.add_argument("--no-write-by-init", dest="write_by_init", action="store_false")
    ap.add_argument("--disable-climatology", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Resume from existing model output csv.")
    ap.add_argument(
        "--replace-vars",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Drop these variables from existing CSV and recompute them for all (init, lead) samples. "
            "Only those variables are evaluated (same as setting --eval-vars to this list). "
            "Ignores --resume skipping so msl-only reruns are not skipped due to other variables. "
            "Typical: --replace-vars msl"
        ),
    )
    ap.add_argument(
        "--flush-every-samples",
        type=int,
        default=1,
        help="Flush cumulative CSV every N completed samples (default 1).",
    )
    ap.add_argument(
        "--parallel-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run each model in its own process (default on). Use --no-parallel-models to run models one after another.",
    )
    ap.add_argument(
        "--model-parallel-jobs",
        type=int,
        default=None,
        help="Max concurrent model processes when --parallel-models. "
        "Default: min(n_models, usable_cpus//max-workers); usable_cpus from sysfs/cgroup.",
    )
    ap.add_argument(
        "--cpu-cap",
        type=int,
        default=None,
        metavar="N",
        help="Force usable CPU count for default model-parallel sizing (optional). "
        "Also env STATION_EVAL_CPU_CAP.",
    )
    ap.set_defaults(write_by_init=True)
    args = ap.parse_args()

    if args.forecasts_roots:
        forecast_roots_resolved: list[Path] = [Path(p).resolve() for p in args.forecasts_roots]
    else:
        forecast_roots_resolved = [Path(args.forecasts_root).resolve()]
    print(
        "[station-eval] forecast_roots (priority order): "
        + ", ".join(str(p) for p in forecast_roots_resolved)
    )
    _usable = usable_cpu_count(args.cpu_cap)
    print(
        f"[station-eval] usable_cpus={_usable} "
        "(override via --cpu-cap or STATION_EVAL_CPU_CAP; else min(affinity, sysfs online))"
    )

    replace_vars_list = [v.strip().lower() for v in (args.replace_vars or []) if v.strip()]
    if replace_vars_list:
        eval_vars_tuple = tuple(replace_vars_list)
    else:
        eval_vars_tuple = tuple(v.strip() for v in args.eval_vars if v.strip())

    expected_inits = _iter_inits(args.start, args.end, args.init_hours)
    use_clim = not args.disable_climatology
    station_latlon_s = str(args.station_latlon_json) if args.station_latlon_json else None
    by_init_str = str(args.by_init_dir or (args.out_root / "by_init"))

    models_list = [m.strip().lower() for m in args.models if m.strip()]
    jobs: list[StationEvalModelJob] = []
    for model in models_list:
        if not any((r / model).exists() for r in forecast_roots_resolved):
            print(
                f"[station-eval] skip model={model}: no directory under any forecast root "
                f"({model})"
            )
            continue
        jobs.append(
            StationEvalModelJob(
                model=model,
                forecast_roots=tuple(str(p) for p in forecast_roots_resolved),
                out_root=str(args.out_root),
                by_init_dir=by_init_str,
                write_by_init=bool(args.write_by_init),
                station_root=str(args.station_root),
                era5_root=str(args.era5_root),
                eval_vars=eval_vars_tuple,
                use_climatology=use_clim,
                station_latlon_json=station_latlon_s,
                expected_inits=tuple(expected_inits),
                max_lead=int(args.max_lead),
                max_workers=int(args.max_workers),
                resume=bool(args.resume),
                replace_vars_list=tuple(replace_vars_list),
                flush_every_samples=int(args.flush_every_samples),
            )
        )

    if not jobs:
        print("[station-eval] no models to run")
        sys.exit(1)

    mpj = args.model_parallel_jobs
    if mpj is None:
        mpj = _default_model_parallel_jobs(len(jobs), int(args.max_workers), args.cpu_cap)

    run_parallel = bool(args.parallel_models) and len(jobs) > 1
    if run_parallel:
        print(
            f"[station-eval] parallel_models concurrent={mpj} total_models={len(jobs)} "
            f"max_workers_per_model={args.max_workers} usable_cpus={usable_cpu_count(args.cpu_cap)}"
        )
        errors: list[tuple[str, Exception]] = []
        with ProcessPoolExecutor(max_workers=max(1, mpj)) as pool:
            fut_map = {pool.submit(_station_eval_run_one_model, j): j.model for j in jobs}
            for fut in as_completed(fut_map):
                name = fut_map[fut]
                try:
                    fut.result()
                except Exception as e:
                    errors.append((name, e))
                    print(f"[station-eval] model={name} FAILED: {type(e).__name__}: {e}")
        if errors:
            sys.exit(1)
    else:
        if len(jobs) > 1:
            print("[station-eval] parallel_models=off (sequential models)")
        for j in jobs:
            _station_eval_run_one_model(j)


if __name__ == "__main__":
    main()

